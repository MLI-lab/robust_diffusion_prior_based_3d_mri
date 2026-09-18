from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import fastmri
import numpy as np
import torch
from fastmri.data.transforms import normalize_instance, to_tensor

from src.datasets.fastmri_volume_dataset import VolumeDatasetSample
from src.problem_trafos.utils.noncartesian_mri import (
    build_mrinufft_operator,
    complex_gaussian_noise_like,
    ensure_complex_torch,
    ensure_ri_torch,
    generate_radial_trajectory,
    load_noncartesian_trajectory,
    prepare_sens_maps,
)
from src.utils.fftn3d import fft3c, ifft3c

from .base_dataset_trafo import BaseDatasetTrafo
from .volume_preprocess_utils import (
    interpolate_sensmaps as _interpolate_sensmaps,
    interpolate_volume as _interpolate_volume,
    scale_by_kspace_norm as _scale_by_kspace_norm,
)


class FastMRI3DNonCartesianDataTransform(BaseDatasetTrafo[VolumeDatasetSample]):
    def __init__(
        self,
        which_challenge: str,
        mask_enabled: bool,
        mask_type: str,
        mask_accelerations: Tuple[int],
        mask_center_fractions: Tuple[float],
        mask_seed: int,
        trajectory_path: Optional[str] = None,
        trajectory_acs_path: Optional[str] = None,
        trajectory_format: str = "npy",
        trajectory_dwell_time: float = 0.005,
        trajectory_sha256: Optional[str] = None,
        trajectory_source_url: Optional[str] = None,
        trajectory_expected_size: Optional[int] = None,
        trajectory_use_acs: bool = True,
        trajectory_normalize: bool = True,
        trajectory_clip: bool = True,
        trajectory_compute_on_the_fly: bool = False,
        trajectory_num_spokes: int = 8192,
        trajectory_readout_oversample: Optional[float] = None,
        trajectory_img_shape: Optional[Tuple[int, ...]] = None,
        trajectory_gamma: float = 1.0,
        trajectory_acs_size: int = 24,
        nufft_backend: str = "finufft",
        density_compensation: Any = True,
        noise_std: float = 0.0,
        use_seed: bool = True,
        provide_pseudoinverse: bool = False,
        provide_measurement: bool = True,
        use_real_synth_data: bool = False,
        return_magnitude_image: bool = False,
        return_cropped_pseudoinverse: bool = False,
        scale_target_by_kspacenorm: bool = False,
        target_scaling_factor: float = 1.0,
        target_interpolate_by_factor: float = 1.0,
        target_interpolation_method: str = "fourier",
        normalize_target: bool = False,
        target_type: str = "rss",
        pseudoinverse_conv_averaging_shape: Optional[Tuple[int, int]] = None,
        multicoil_reduction_op: str = "sum",
        device: str = "cpu",
        wrapped_2d: bool = False,
        return_pseudoinverse_as_observation: bool = False,
        upsampfac: Optional[float] = None,
        **_: Any,
    ):
        super().__init__(
            provide_measurement=provide_measurement,
            provide_pseudoinverse=provide_pseudoinverse,
        )

        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")
        if wrapped_2d:
            raise NotImplementedError("wrapped_2d is not supported for non-Cartesian 3D MRI.")

        self.which_challenge = which_challenge
        self.mask_enabled = mask_enabled
        self.mask_type = mask_type
        self.mask_accelerations = mask_accelerations
        self.mask_center_fractions = mask_center_fractions
        self.mask_seed = mask_seed
        self.use_seed = use_seed

        self.trajectory_path = trajectory_path
        self.trajectory_acs_path = trajectory_acs_path
        self.trajectory_format = trajectory_format
        self.trajectory_dwell_time = trajectory_dwell_time
        self.trajectory_sha256 = trajectory_sha256
        self.trajectory_source_url = trajectory_source_url
        self.trajectory_expected_size = trajectory_expected_size
        self.trajectory_use_acs = trajectory_use_acs
        self.trajectory_normalize = trajectory_normalize
        self.trajectory_clip = trajectory_clip
        self.trajectory_compute_on_the_fly = trajectory_compute_on_the_fly
        self.trajectory_num_spokes = trajectory_num_spokes
        self.trajectory_readout_oversample = trajectory_readout_oversample
        self.trajectory_img_shape = trajectory_img_shape
        self.trajectory_gamma = trajectory_gamma
        self.trajectory_acs_size = trajectory_acs_size
        self.nufft_backend = nufft_backend
        self.density_compensation = density_compensation
        self.noise_std = noise_std
        self.upsampfac = upsampfac

        self.use_real_synth_data = use_real_synth_data
        self.return_magnitude_image = return_magnitude_image
        self.return_cropped_pseudoinverse = return_cropped_pseudoinverse
        self.scale_target_by_kspacenorm = scale_target_by_kspacenorm
        self.target_scaling_factor = target_scaling_factor
        self.target_interpolate_by_factor = target_interpolate_by_factor
        self.target_interpolation_method = target_interpolation_method
        self.normalize_target = normalize_target
        self.target_type = target_type
        self.pseudoinverse_conv_averaging_shape = pseudoinverse_conv_averaging_shape
        self.multicoil_reduction_op = multicoil_reduction_op
        self.device = device
        self.return_pseudoinverse_as_observation = return_pseudoinverse_as_observation

    def _device(self) -> torch.device:
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def requires_sensmaps(self) -> bool:
        return True

    def _load_target(self, kspace_torch, target, attrs):
        if target is not None:
            if self.target_type == "rss":
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target
                if self.device is not None:
                    target_torch = target_torch.to(self.device)
            elif self.target_type == "mvue":
                target_torch = to_tensor(target) if not torch.is_tensor(target) else target
                if self.device is not None:
                    target_torch = target_torch.to(self.device)
                target_torch = torch.view_as_real(target_torch) if torch.is_complex(target_torch) else target_torch
            elif self.target_type == "fullysampled_rec":
                target_torch = ifft3c(kspace_torch)
                if self.which_challenge == "multicoil":
                    if self.multicoil_reduction_op == "sum":
                        target_torch = target_torch.sum(dim=0)
                    elif self.multicoil_reduction_op == "mean":
                        target_torch = target_torch.mean(dim=0)
                    elif self.multicoil_reduction_op == "norm":
                        target_torch = target_torch.norm(dim=0)
                    elif self.multicoil_reduction_op == "norm_sum_sensmaps":
                        S = prepare_sens_maps(attrs["sens_maps"], device=kspace_torch.device)
                        S_safe = S.clone()
                        nonzero = S_safe != 0
                        if nonzero.any():
                            S_safe[~nonzero] = S_safe[nonzero].abs().min() + 0j
                        norm = torch.abs(S_safe).square().sum(dim=0).sqrt()
                        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
                        target_torch = torch.view_as_real(
                            torch.sum(torch.view_as_complex(target_torch) * torch.conj(S_safe), dim=0) / norm
                        )
                    else:
                        raise NotImplementedError(
                            f"Reduction operation {self.multicoil_reduction_op} not supported"
                        )
            else:
                raise NotImplementedError(f"Target type {self.target_type} not supported")

            if self.scale_target_by_kspacenorm:
                target_torch = _scale_by_kspace_norm(target_torch, attrs["kspace_vol_norm"])

            if self.target_scaling_factor != 1.0:
                target_torch = target_torch * self.target_scaling_factor

            if self.target_interpolate_by_factor != 1.0:
                target_torch = _interpolate_volume(
                    target_torch,
                    self.target_interpolate_by_factor,
                    self.target_interpolation_method,
                )

            if self.normalize_target:
                target_torch, _, _ = normalize_instance(target_torch, eps=1e-11)
                target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.tensor([0.0], device=self._device())

        return target_torch

    def _target_to_coil_images(self, target_torch: torch.Tensor, attrs: Dict[str, Any]):
        if self.which_challenge != "multicoil":
            return ensure_complex_torch(target_torch, device=self._device())

        sens_maps = attrs.get("sens_maps")
        if sens_maps is None:
            raise ValueError("Sensitivity maps are required for non-Cartesian multicoil simulation.")
        S = prepare_sens_maps(sens_maps, device=self._device())
        target_complex = ensure_complex_torch(target_torch, device=S.device)
        return target_complex.unsqueeze(0) * S

    def _combine_coils(self, coil_images: torch.Tensor, attrs: Dict[str, Any]) -> torch.Tensor:
        if self.which_challenge != "multicoil":
            return ensure_ri_torch(coil_images, device=coil_images.device)

        image = ensure_ri_torch(coil_images, device=coil_images.device)
        if self.return_magnitude_image:
            image = fastmri.complex_abs(image)
            image = fastmri.rss(image).unsqueeze(-1)
            return image

        if self.multicoil_reduction_op == "sum":
            return image.sum(dim=0)
        if self.multicoil_reduction_op == "mean":
            return image.mean(dim=0)
        if self.multicoil_reduction_op == "norm":
            return image.norm(dim=0)
        if self.multicoil_reduction_op == "norm_sum_sensmaps":
            S = prepare_sens_maps(attrs["sens_maps"], device=coil_images.device)
            nonzero = S != 0
            S_safe = S.clone()
            if nonzero.any():
                S_safe[~nonzero] = S_safe[nonzero].abs().min() + 0j
            norm = torch.abs(S_safe).square().sum(dim=0).sqrt()
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            return torch.view_as_real(torch.sum(coil_images * torch.conj(S_safe), dim=0) / norm)
        raise NotImplementedError(f"Reduction operation {self.multicoil_reduction_op} not supported")

    def _transform(
        self,
        sample: VolumeDatasetSample,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        kspace, target, attrs = sample.kspace, sample.target, sample.attrs
        work_device = self._device()
        masked_kspace = torch.tensor([0.0], device=work_device)
        image = torch.tensor([0.0], device=work_device)
        masked_kspace_complex: Optional[torch.Tensor] = None
        nufft_op = None
        attrs = dict(attrs)
        attrs["mask"] = None

        kspace_torch = None
        if kspace is not None:
            if torch.is_tensor(kspace):
                kspace_torch = kspace
            else:
                kspace_torch = to_tensor(np.asarray(kspace))
            if self.device is not None:
                kspace_torch = kspace_torch.to(self.device)

        target_torch = self._load_target(kspace_torch, target, attrs)

        if self.provide_pseudoinverse or self.provide_measurement:
            if self.use_real_synth_data:
                if self.target_type == "rss":
                    kspace_torch = fft3c(torch.stack([target_torch, torch.zeros_like(target_torch)], dim=-1))
                elif self.target_type in ("fullysampled_rec", "mvue"):
                    attrs["sens_maps"] = np.ones_like(attrs["sens_maps"])
                    kspace_torch = fft3c(target_torch)
                else:
                    raise NotImplementedError(f"Target type {self.target_type} not supported")

            if self.target_interpolate_by_factor != 1.0:
                logging.info(
                    "Interpolated target requested for non-Cartesian data -> interpolate sensitivities as well."
                )
                attrs["sens_maps"] = _interpolate_sensmaps(
                    attrs["sens_maps"],
                    self.target_interpolate_by_factor,
                    self.target_interpolation_method,
                    device=self.device,
                )

            coil_images = self._target_to_coil_images(target_torch, attrs)
            image_shape = tuple(int(v) for v in coil_images.shape[-3:])
            if self.trajectory_compute_on_the_fly:
                # Use actual data shape; only fall back to config if trajectory_img_shape explicitly set and not None
                img_shape = image_shape if self.trajectory_img_shape is None else tuple(self.trajectory_img_shape)
                trajectory = generate_radial_trajectory(
                    num_spokes=self.trajectory_num_spokes,
                    image_shape=img_shape,
                    readout_oversample=self.trajectory_readout_oversample,
                    gamma=self.trajectory_gamma,
                    use_acs=self.trajectory_use_acs,
                    acs_size=self.trajectory_acs_size if self.trajectory_use_acs else None,
                    normalize_to_unit_box=self.trajectory_normalize,
                    clip=self.trajectory_clip,
                )
            else:
                if self.trajectory_path is None:
                    raise ValueError(
                        "trajectory_path must be set when trajectory_compute_on_the_fly=False."
                    )
                trajectory = load_noncartesian_trajectory(
                    trajectory_path=self.trajectory_path,
                    acs_path=self.trajectory_acs_path,
                    use_acs=self.trajectory_use_acs,
                    image_shape=image_shape,
                    normalize_to_unit_box=self.trajectory_normalize,
                    clip=self.trajectory_clip,
                    trajectory_format=self.trajectory_format,
                    dwell_time=self.trajectory_dwell_time,
                    source_url=self.trajectory_source_url,
                    sha256=self.trajectory_sha256,
                    expected_size=self.trajectory_expected_size,
                )
            logging.info(
                "[NUFFT][dataset_trafo] backend=%s density_compensation=%s image_shape=%s n_coils=%s",
                self.nufft_backend,
                self.density_compensation,
                image_shape,
                coil_images.shape[0] if coil_images.ndim == 4 else 1,
            )
            nufft_op = build_mrinufft_operator(
                samples=trajectory,
                image_shape=image_shape,
                n_coils=coil_images.shape[0] if coil_images.ndim == 4 else 1,
                backend=self.nufft_backend,
                density_compensation=self.density_compensation,
                squeeze_dims=True,
                upsampfac=self.upsampfac,
                device=self.device,
            )

            masked_kspace_complex = ensure_complex_torch(nufft_op.op(coil_images), device=coil_images.device)
            assert masked_kspace_complex is not None
            masked_kspace_complex = masked_kspace_complex + complex_gaussian_noise_like(
                masked_kspace_complex, self.noise_std
            )
            masked_kspace = ensure_ri_torch(masked_kspace_complex, device=coil_images.device)

            # Store trajectory as "mask" to mirror the Cartesian attrs_out convention
            # (same key, always a tensor, collateable by default_collate).
            attrs["mask"] = torch.from_numpy(trajectory)

        if self.provide_pseudoinverse:
            assert nufft_op is not None and masked_kspace_complex is not None
            coil_pseudoinverse = ensure_complex_torch(
                nufft_op.adj_op(masked_kspace_complex), device=work_device
            )
            image = self._combine_coils(coil_pseudoinverse, attrs)

        if self.device != "cpu":
            torch.cuda.empty_cache()

        if self.provide_measurement and self.return_pseudoinverse_as_observation:
            assert self.provide_pseudoinverse, "Pseudoinverse must be provided if it is returned as observation."
            assert image is not None
            masked_kspace = image.clone()

        attrs_out: Dict[str, Any] = {"mask": attrs["mask"]}
        if attrs.get("sens_maps") is not None:
            attrs_out["sens_maps"] = attrs["sens_maps"]

        return masked_kspace, target_torch, image, attrs_out
