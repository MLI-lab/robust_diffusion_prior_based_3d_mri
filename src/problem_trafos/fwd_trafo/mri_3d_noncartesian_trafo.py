from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from .base_fwd_trafo import BaseFwdTrafo
from src.problem_trafos.utils.noncartesian_mri import (
    build_mrinufft_operator,
    ensure_complex_torch,
    ensure_ri_torch,
    generate_radial_trajectory,
    load_noncartesian_trajectory,
    prepare_sens_maps,
)


class NonCartesianMRI3DTrafo(BaseFwdTrafo):
    def __init__(
        self,
        mask_enabled: bool,
        mask_type: Optional[str],
        mask_accelerations: Any,
        mask_center_fractions: Any,
        mask_seed: Any,
        include_sensitivitymaps: bool,
        sensitivitymaps_complex: bool,
        sensitivitymaps_fillouter: bool,
        wrapped_2d_mode: bool,
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
        trajectory_img_shape: Optional[Sequence[int]] = None,
        trajectory_gamma: float = 1.0,
        trajectory_acs_size: int = 24,
        nufft_backend: str = "finufft",
        density_compensation: Any = True,
        upsampfac: Optional[float] = None,
        **_: Any,
    ):
        super().__init__()
        self.mask_enabled = mask_enabled
        self.mask_type = mask_type
        self.mask_accelerations = mask_accelerations
        self.mask_center_fractions = mask_center_fractions
        self.mask_seed = mask_seed

        self.include_sensitivitymaps = include_sensitivitymaps
        self.sensitivitymaps_complex = sensitivitymaps_complex
        self.sensitivitymaps_fillouter = sensitivitymaps_fillouter
        self.wrapped_2d_mode = wrapped_2d_mode

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
        self.upsampfac = upsampfac

        self.sense_matrix: Optional[torch.Tensor] = None
        self.sense_matrix_normalization_constant: Optional[torch.Tensor] = None
        self.image_shape: Optional[Sequence[int]] = None
        self.trajectory = None
        self.nufft_operator = None

        if wrapped_2d_mode:
            raise NotImplementedError("wrapped_2d_mode is not supported for non-Cartesian 3D MRI.")

    def calibrate(self, y: Tensor, calib_params) -> None:
        if not self.include_sensitivitymaps:
            raise NotImplementedError("Non-Cartesian MRI3D currently requires sensitivity maps.")

        device = y.device
        sens_maps = calib_params.get("sens_maps")
        if sens_maps is None:
            raise ValueError("calib_params['sens_maps'] is required for non-Cartesian MRI3D.")

        self.sense_matrix = prepare_sens_maps(sens_maps, device=device)
        self.image_shape = tuple(int(v) for v in self.sense_matrix.shape[-3:])

        self.sense_matrix_normalization_constant = (
            self.sense_matrix.abs().square().sum(dim=0).sqrt().to(device)
        )

        if self.sensitivitymaps_fillouter:
            zero_mask = self.sense_matrix_normalization_constant == 0
            if zero_mask.any():
                fill_value = 1.0 / (self.sense_matrix.shape[0] ** 0.5)
                self.sense_matrix = torch.where(
                    zero_mask.unsqueeze(0),
                    torch.full_like(self.sense_matrix, fill_value),
                    self.sense_matrix,
                )
                self.sense_matrix_normalization_constant[zero_mask] = 1.0

        # "mask" carries the trajectory (dataset trafo convention); fall back to
        # "trajectory" for backward compatibility with pre-loaded trajectory paths.
        trajectory = calib_params.get("mask")
        if trajectory is None:
            trajectory = calib_params.get("trajectory")
        if trajectory is None:
            if self.trajectory_compute_on_the_fly:
                img_shape = tuple(self.trajectory_img_shape) if self.trajectory_img_shape is not None else self.image_shape
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
                    image_shape=self.image_shape,
                    normalize_to_unit_box=self.trajectory_normalize,
                    clip=self.trajectory_clip,
                    trajectory_format=self.trajectory_format,
                    dwell_time=self.trajectory_dwell_time,
                    source_url=self.trajectory_source_url,
                    sha256=self.trajectory_sha256,
                    expected_size=self.trajectory_expected_size,
                )
        else:
            if torch.is_tensor(trajectory):
                trajectory = trajectory.detach().cpu().numpy()

        self.trajectory = trajectory
        logging.info(
            "[NUFFT][fwd_trafo] backend=%s density_compensation=%s image_shape=%s n_coils=%s",
            self.nufft_backend,
            self.density_compensation,
            self.image_shape,
            self.sense_matrix.shape[0],
        )
        self.nufft_operator = build_mrinufft_operator(
            samples=self.trajectory,
            image_shape=self.image_shape,
            n_coils=self.sense_matrix.shape[0],
            backend=self.nufft_backend,
            density_compensation=self.density_compensation,
            squeeze_dims=True,
            upsampfac=self.upsampfac,
            autograd=True,
            device=self.sense_matrix.device,
        )

    def trafo(
        self,
        x: Tensor,
        slice_inds: Optional[Tensor] = None,
        slice_axis: Optional[int] = None,
    ) -> Tensor:
        if slice_inds is not None or slice_axis is not None:
            raise NotImplementedError("Slice-wise non-Cartesian data consistency is not implemented.")
        if self.sense_matrix is None or self.nufft_operator is None:
            raise RuntimeError("Call calibrate() before trafo() for non-Cartesian MRI3D.")

        x_complex = ensure_complex_torch(x, device=self.sense_matrix.device)
        coil_images = x_complex.unsqueeze(0) * self.sense_matrix

        squeeze_dims = getattr(self.nufft_operator, "squeeze_dims", True)
        if not squeeze_dims:
            coil_images_input = coil_images.unsqueeze(0)
            y_raw = self.nufft_operator.op(coil_images_input)
            y_complex = ensure_complex_torch(y_raw, device=self.sense_matrix.device).squeeze(0)
        else:
            y_complex = ensure_complex_torch(self.nufft_operator.op(coil_images), device=self.sense_matrix.device)

        return ensure_ri_torch(y_complex, device=x.device)

    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        if self.sense_matrix is None or self.nufft_operator is None:
            raise RuntimeError("Call calibrate() before trafo_adjoint() for non-Cartesian MRI3D.")

        y_complex = ensure_complex_torch(observation, device=self.sense_matrix.device)

        squeeze_dims = getattr(self.nufft_operator, "squeeze_dims", True)
        if not squeeze_dims:
            y_complex_input = y_complex.unsqueeze(0)
            coil_images_raw = self.nufft_operator.adj_op(y_complex_input)
            coil_images = ensure_complex_torch(coil_images_raw, device=self.sense_matrix.device).squeeze(0)
        else:
            coil_images = ensure_complex_torch(
                self.nufft_operator.adj_op(y_complex), device=self.sense_matrix.device
            )

        image = torch.sum(torch.conj(self.sense_matrix) * coil_images, dim=0)

        if self.sense_matrix_normalization_constant is not None:
            denom = self.sense_matrix_normalization_constant
            safe = torch.where(denom == 0, torch.ones_like(denom), denom)
            image = image / safe

        return ensure_ri_torch(image, device=observation.device)
