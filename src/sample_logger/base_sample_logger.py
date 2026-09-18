from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

import math

import torch

from src.representations.base_coord_based_representation import CoordBasedRepresentation

class BaseSampleLogger(ABC):
    """Models logging mechanisms for the reconstruction pipeline.
    It is based on the assumption that within a run N samples are reconstruction, with e.g. 1000 iterations per sample.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _to_python_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().cpu().reshape(-1)[0].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_python_int(value: Any) -> Optional[int]:
        float_value = BaseSampleLogger._to_python_float(value)
        if float_value is None:
            return None
        int_value = int(float_value)
        return int_value if int_value > 0 else None

    @staticmethod
    def _looks_like_trajectory(x: Any) -> bool:
        if x is None:
            return False
        try:
            t = torch.as_tensor(x)
        except (TypeError, ValueError):
            return False
        return t.ndim >= 2 and t.shape[-1] in (2, 3)

    @staticmethod
    def _image_shape_values(image_shape: Any) -> Optional[list[int]]:
        if image_shape is None:
            return None
        try:
            image_shape_t = torch.as_tensor(image_shape)
        except (TypeError, ValueError):
            return None
        if image_shape_t.numel() == 0:
            return None
        values = [int(v) for v in image_shape_t.detach().cpu().reshape(-1).tolist()]
        values = [v for v in values if v > 0]
        return values or None

    @staticmethod
    def _attrs_sampling_kind_is_noncartesian(attrs: Optional[Dict[str, Any]] = None) -> bool:
        if not isinstance(attrs, dict):
            return False
        return str(attrs.get("sampling_kind", "")).lower() == "noncartesian"

    def _fwd_trafo_is_noncartesian(self) -> bool:
        fwd_trafo = getattr(self, "fwd_trafo", None)
        if fwd_trafo is None:
            return False
        if self._looks_like_trajectory(getattr(fwd_trafo, "trajectory", None)):
            return True
        return "noncartesian" in fwd_trafo.__class__.__name__.lower()

    def _mask_value_is_trajectory(self, attrs: Optional[Dict[str, Any]], mask_val: Any) -> bool:
        if not self._looks_like_trajectory(mask_val):
            return False
        return self._attrs_sampling_kind_is_noncartesian(attrs) or self._fwd_trafo_is_noncartesian()

    def _image_shape_from_attrs_or_trafo(self, attrs: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if isinstance(attrs, dict) and attrs.get("image_shape") is not None:
            return attrs.get("image_shape")
        fwd_trafo = getattr(self, "fwd_trafo", None)
        return getattr(fwd_trafo, "image_shape", None)

    @staticmethod
    def _effective_acceleration_from_mask(mask: Any) -> Optional[float]:
        if mask is None:
            return None
        mask_t = torch.as_tensor(mask)
        if mask_t.numel() == 0:
            return None

        sampled = int(torch.count_nonzero(mask_t).item())
        total = int(mask_t.numel())
        if sampled <= 0 or total <= 0:
            return None
        return float(total) / float(sampled)

    @staticmethod
    def _trajectory_sample_count(trajectory: Any) -> Optional[int]:
        if trajectory is None:
            return None
        try:
            trajectory_t = torch.as_tensor(trajectory)
        except (TypeError, ValueError):
            return None
        if trajectory_t.ndim == 0 or trajectory_t.numel() == 0:
            return None
        if trajectory_t.ndim >= 2 and trajectory_t.shape[-1] in (2, 3):
            n_samples = int(math.prod(int(v) for v in trajectory_t.shape[:-1]))
        else:
            n_samples = int(trajectory_t.shape[0])
        return n_samples if n_samples > 0 else None

    @staticmethod
    def _trajectory_shot_count(trajectory: Any, attrs: Optional[Dict[str, Any]] = None) -> Optional[int]:
        if isinstance(attrs, dict):
            for key in ("trajectory_num_shots", "num_shots"):
                value = BaseSampleLogger._to_python_int(attrs.get(key))
                if value is not None:
                    return value
            samples_per_shot = BaseSampleLogger._to_python_int(attrs.get("trajectory_samples_per_shot"))
            n_samples = BaseSampleLogger._to_python_int(attrs.get("trajectory_num_samples_without_acs"))
            if n_samples is None:
                n_samples = BaseSampleLogger._to_python_int(attrs.get("trajectory_num_samples"))
            if samples_per_shot is not None and n_samples is not None and n_samples % samples_per_shot == 0:
                return n_samples // samples_per_shot

        if trajectory is None:
            return None
        try:
            trajectory_t = torch.as_tensor(trajectory)
        except (TypeError, ValueError):
            return None
        if trajectory_t.ndim < 3 or trajectory_t.shape[-1] not in (2, 3):
            return None
        shape = tuple(int(v) for v in trajectory_t.shape)
        # Avoid interpreting a batched flattened trajectory [1, N, D] as one shot.
        if trajectory_t.ndim == 3 and shape[0] == 1:
            return None
        n_shots = int(math.prod(shape[:-2]))
        return n_shots if n_shots > 0 else None

    @staticmethod
    def _effective_acceleration_from_trajectory(
        trajectory: Any,
        image_shape: Any,
    ) -> Optional[float]:
        n_samples = BaseSampleLogger._trajectory_sample_count(trajectory)
        image_shape_values = BaseSampleLogger._image_shape_values(image_shape)
        if n_samples is None or image_shape_values is None:
            return None
        n_voxels = int(math.prod(image_shape_values))
        if n_voxels <= 0:
            return None
        return float(n_voxels) / float(n_samples)

    @staticmethod
    def _effective_acceleration_from_shots(
        image_shape: Any,
        n_shots: Optional[int],
    ) -> Optional[float]:
        image_shape_values = BaseSampleLogger._image_shape_values(image_shape)
        if image_shape_values is None or n_shots is None or n_shots <= 0:
            return None
        encoded_lines = int(math.prod(image_shape_values[1:] if len(image_shape_values) >= 3 else image_shape_values))
        if encoded_lines <= 0:
            return None
        return float(encoded_lines) / float(n_shots)

    def _trajectory_from_attrs_or_trafo(self, attrs: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if isinstance(attrs, dict):
            if self._looks_like_trajectory(attrs.get("trajectory")):
                return attrs.get("trajectory")
            if self._mask_value_is_trajectory(attrs, attrs.get("mask")):
                return attrs.get("mask")
        fwd_trafo = getattr(self, "fwd_trafo", None)
        trajectory = getattr(fwd_trafo, "trajectory", None)
        if self._looks_like_trajectory(trajectory):
            return trajectory
        return None

    def _extract_legacy_effective_acceleration(self, attrs: Optional[Dict[str, Any]] = None) -> Optional[float]:
        if isinstance(attrs, dict):
            mask_val = attrs.get("mask")
            # Non-Cartesian trajectories are shaped [..., N_samples, D] with D in {2,3},
            # and can be batched by DataLoader (e.g. [1, N, 3]).
            if self._mask_value_is_trajectory(attrs, mask_val):
                image_shape = self._image_shape_from_attrs_or_trafo(attrs)
                reff = self._effective_acceleration_from_trajectory(mask_val, image_shape)
            else:
                reff = self._effective_acceleration_from_mask(mask_val)
            if reff is not None:
                return reff

            reff = self._effective_acceleration_from_trajectory(
                attrs.get("trajectory"),
                self._image_shape_from_attrs_or_trafo(attrs),
            )
            if reff is not None:
                return reff

            reff = self._to_python_float(attrs.get("effective_acceleration"))
            if reff is not None:
                return reff

        fwd_trafo = getattr(self, "fwd_trafo", None)
        if fwd_trafo is None:
            return None

        reff = self._effective_acceleration_from_mask(getattr(fwd_trafo, "mask", None))
        if reff is not None:
            return reff

        return self._effective_acceleration_from_trajectory(
            getattr(fwd_trafo, "trajectory", None),
            getattr(fwd_trafo, "image_shape", None),
        )

    def _extract_effective_acceleration(self, attrs: Optional[Dict[str, Any]] = None) -> Optional[float]:
        return self._extract_legacy_effective_acceleration(attrs)

    def _extract_effective_acceleration_metrics(self, attrs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        reff = self._extract_effective_acceleration(attrs)
        if reff is not None:
            metrics["Reff"] = float(reff)

        trajectory = self._trajectory_from_attrs_or_trafo(attrs)
        if trajectory is None:
            return metrics

        image_shape = self._image_shape_from_attrs_or_trafo(attrs)
        reff_samples = self._effective_acceleration_from_trajectory(trajectory, image_shape)
        if reff_samples is not None:
            metrics["Reff_samples"] = float(reff_samples)
            metrics.setdefault("Reff", float(reff_samples))

        n_samples = self._trajectory_sample_count(trajectory)
        if n_samples is not None:
            metrics["num_trajectory_samples"] = int(n_samples)

        n_shots = self._trajectory_shot_count(trajectory, attrs)
        if n_shots is not None:
            metrics["num_trajectory_shots"] = int(n_shots)
            reff_shots = self._effective_acceleration_from_shots(image_shape, n_shots)
            if reff_shots is not None:
                metrics["Reff_shots"] = float(reff_shots)

        return metrics

    @abstractmethod
    def init_run(self, **kwargs):
        """
        This method is called once before starting reconstruction.
        """
        pass

    @abstractmethod
    def init_sample_log(self, **kwargs):
        """
        Called once before reconstruction of a sample (but can called multiple times in one run)
        """
        pass

    @abstractmethod
    def __call__(self, representation: CoordBasedRepresentation, step: int, **kwargs):
        """
        Called during reconstruction ()
        """

    @abstractmethod
    def close_sample_log(self, representation: CoordBasedRepresentation):
        """
        Called once after reconstruction of a sample (but can called multiple times in one run)
        """
        pass

    @abstractmethod
    def close_run(self):
        """
        Called once after the reconstruction is finished.
        """
        pass

    @abstractmethod
    def get_final_stats(self) -> Dict[str, Any]:
        """
        Returns a dictionary of final statistics after the run is finished.
        """
        pass