from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from src.datasets.fastmri_volume_dataset import VolumeDatasetSample

from .base_dataset_trafo import BaseDatasetTrafo


class PreparedMRIReconDataTransform(BaseDatasetTrafo[VolumeDatasetSample]):
    """Transform for test datasets whose measurement has already been prepared."""

    def __init__(
        self,
        sampling_kind: str = "cartesian",
        which_challenge: str = "multicoil",
        multicoil_reduction_op: str = "norm_sum_sensmaps",
        return_magnitude_image: bool = False,
        nufft_backend: str = "pytorch",
        density_compensation: Any = False,
        upsampfac: Optional[float] = None,
        device: str = "cpu",
        provide_pseudoinverse: bool = False,
        provide_measurement: bool = True,
        return_pseudoinverse_as_observation: bool = False,
        **_: Any,
    ):
        super().__init__(
            provide_measurement=provide_measurement,
            provide_pseudoinverse=provide_pseudoinverse,
        )
        if sampling_kind not in ("cartesian", "noncartesian"):
            raise ValueError(f"sampling_kind must be cartesian or noncartesian, got {sampling_kind}.")
        self.sampling_kind = sampling_kind
        self.which_challenge = which_challenge
        self.multicoil_reduction_op = multicoil_reduction_op
        self.return_magnitude_image = return_magnitude_image
        self.nufft_backend = nufft_backend
        self.density_compensation = density_compensation
        self.upsampfac = upsampfac
        self.device = device
        self.return_pseudoinverse_as_observation = return_pseudoinverse_as_observation

    def requires_sensmaps(self) -> bool:
        return self.which_challenge == "multicoil" and self.multicoil_reduction_op == "norm_sum_sensmaps"

    def _device(self) -> torch.device:
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)

    def _transform(
        self,
        sample: VolumeDatasetSample,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        attrs = dict(sample.attrs)
        if sample.kspace is None:
            raise ValueError("Prepared MRI recon data must contain observed kspace.")

        observation = sample.kspace if torch.is_tensor(sample.kspace) else torch.from_numpy(sample.kspace)
        observation = observation.to(self._device())

        if sample.target is None:
            target = torch.tensor([0.0], device=self._device())
        else:
            target = sample.target if torch.is_tensor(sample.target) else torch.from_numpy(sample.target)
            target = target.to(self._device())
            if torch.is_complex(target):
                target = torch.view_as_real(target.contiguous())

        image = torch.tensor([0.0], device=self._device())
        if self.provide_pseudoinverse:
            stored_pseudoinverse = attrs.get("pseudoinverse")
            if stored_pseudoinverse is None:
                raise ValueError("Prepared MRI recon data must contain a stored pseudoinverse.")
            image = stored_pseudoinverse if torch.is_tensor(stored_pseudoinverse) else torch.from_numpy(stored_pseudoinverse)
            image = image.to(self._device())
            if torch.is_complex(image):
                image = torch.view_as_real(image.contiguous())

        if self.provide_measurement and self.return_pseudoinverse_as_observation:
            if not self.provide_pseudoinverse:
                raise ValueError("Pseudoinverse must be provided when returned as observation.")
            observation = image.clone()

        attrs_out: Dict[str, Any] = {
            "mask": attrs.get("trajectory", attrs.get("mask", None)),
        }
        for key in (
            "prepared_recon_dataset",
            "sampling_kind",
            "sensmaps_from_observed_data",
            "observation_scaling_factor",
            "observation_scaling_rep_numel",
            "source_file",
            "image_shape",
        ):
            if key in attrs:
                attrs_out[key] = attrs[key]
        for key, value in attrs.items():
            if str(key).startswith("trajectory_"):
                attrs_out[key] = value
        if "trajectory" in attrs:
            attrs_out["trajectory"] = attrs["trajectory"]
        if "sens_maps" in attrs:
            attrs_out["sens_maps"] = attrs["sens_maps"]
        return observation, target, image, attrs_out
