from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import logging
import numpy as np
import torch

from src.problem_trafos.utils.trajectory_assets import ensure_trajectory_asset


ArrayLike = Union[np.ndarray, torch.Tensor]


def import_mrinufft_get_operator():
    try:
        from mrinufft import get_operator  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "mri-nufft is required for the non-Cartesian MRI pipeline. "
            "Install it with `pip install mri-nufft[finufft,autodiff]`."
        ) from exc
    return get_operator


def ensure_complex_torch(x: ArrayLike, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if device is not None:
        x = x.to(device)
    if torch.is_complex(x):
        return x.contiguous()
    if x.shape[-1] == 2:
        return torch.view_as_complex(x.contiguous())
    return torch.complex(x, torch.zeros_like(x)).contiguous()


def ensure_ri_torch(x: ArrayLike, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if device is not None:
        x = x.to(device)
    if torch.is_complex(x):
        return torch.view_as_real(x.contiguous())
    if x.shape[-1] == 2:
        return x
    return torch.stack((x, torch.zeros_like(x)), dim=-1)


def ensure_numpy_complex(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if torch.is_complex(x):
            return x.numpy()
        if x.shape[-1] == 2:
            return torch.view_as_complex(x.contiguous()).numpy()
        return torch.complex(x, torch.zeros_like(x)).numpy()
    if np.iscomplexobj(x):
        return x.astype(np.complex64, copy=False)
    if x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]
    return x.astype(np.complex64, copy=False)


def parse_metadata_file(metadata_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    if metadata_path is None:
        return {}

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}

    meta: Dict[str, Any] = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def generate_radial_trajectory(
    num_spokes: int,
    image_shape: Sequence[int],
    readout_oversample: Optional[float] = None,
    gamma: float = 1.0,
    golden: bool = True,
    acs_size: Optional[int] = None,
    use_acs: bool = True,
    normalize_to_unit_box: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """Generate a 3-D golden-angle radial trajectory on-the-fly using sigpy."""
    try:
        from sigpy.mri import radial as sigpy_radial  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "sigpy is required for on-the-fly trajectory generation. "
            "Install it with: pip install sigpy"
        ) from exc

    readout_points = int(max(image_shape) * float(readout_oversample)) if readout_oversample else max(image_shape)
    coord = sigpy_radial(
        coord_shape=(num_spokes, readout_points, len(image_shape)),
        img_shape=tuple(image_shape),
        golden=golden,
    ).astype(np.float32)

    if not np.isclose(gamma, 1.0):
        r = np.linalg.norm(coord, axis=-1, keepdims=True)
        r_max = float(np.max(r))
        eps = 1e-8
        direction = coord / np.maximum(r, eps)
        r_norm = r / r_max
        r_new = r_max * (r_norm ** gamma)
        coord = (direction * r_new).astype(np.float32)

    trajectory = coord.reshape(-1, len(image_shape))

    if use_acs and acs_size is not None and acs_size > 0:
        ranges = [np.arange(-acs_size // 2, acs_size // 2) for _ in image_shape]
        grids = np.meshgrid(*ranges, indexing="ij")
        acs = np.stack([g.ravel() for g in grids], axis=-1).astype(np.float32)
        trajectory = np.concatenate([trajectory, acs], axis=0)

    if normalize_to_unit_box:
        trajectory = normalize_trajectory_samples(trajectory, image_shape=image_shape)

    if clip:
        trajectory = np.clip(trajectory, -0.5, 0.5)

    return trajectory.astype(np.float32, copy=False)


def normalize_trajectory_samples(
    samples: np.ndarray,
    image_shape: Sequence[int],
    assume_sigpy_grid_if_needed: bool = True,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[-1] != len(image_shape):
        raise ValueError(
            f"Trajectory must have shape (N, {len(image_shape)}), got {samples.shape}."
        )

    if not assume_sigpy_grid_if_needed:
        return samples

    if float(np.max(np.abs(samples))) <= 0.5 + 1e-6:
        return samples

    scale = np.asarray(image_shape, dtype=np.float32).reshape(1, -1)
    return samples / scale


def load_noncartesian_trajectory(
    trajectory_path: Union[str, Path],
    image_shape: Sequence[int],
    acs_path: Optional[Union[str, Path]] = None,
    use_acs: bool = True,
    normalize_to_unit_box: bool = True,
    clip: bool = True,
    trajectory_format: str = "npy",
    dwell_time: float = 0.005,
    source_url: Optional[str] = None,
    sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    return_metadata: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    trajectory_path = Path(ensure_trajectory_asset(
        trajectory_path,
        source_url=source_url,
        sha256=sha256,
        expected_size=expected_size,
    ))
    trajectory_format = str(trajectory_format or "npy").lower()
    metadata: Dict[str, Any] = {}
    if trajectory_format in ("npy", "numpy"):
        trajectory = np.load(trajectory_path).astype(np.float32)
    elif trajectory_format in ("mrinufft_bin", "bin"):
        try:
            from mrinufft.io import read_trajectory  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "mri-nufft is required to read mrinufft_bin trajectory files. "
                "Install it with `pip install mri-nufft[finufft,autodiff]`."
            ) from exc
        trajectory, params = read_trajectory(str(trajectory_path), dwell_time=float(dwell_time))
        dim = int(params.get("dimension", trajectory.shape[-1]))
        if dim != len(image_shape):
            raise ValueError(
                f"Trajectory dimension from {trajectory_path} is {dim}, expected {len(image_shape)} "
                f"for image_shape={tuple(image_shape)}."
            )
        trajectory_shape = tuple(int(v) for v in np.asarray(trajectory).shape)
        metadata["trajectory_dimension"] = dim
        metadata["trajectory_raw_shape"] = trajectory_shape
        if params.get("num_shots") is not None:
            metadata["trajectory_num_shots"] = int(params["num_shots"])
        if len(trajectory_shape) >= 2 and trajectory_shape[-1] == dim:
            metadata["trajectory_samples_per_shot"] = int(np.prod(trajectory_shape[1:-1]))
        if params.get("num_samples_per_shot") is not None:
            metadata["trajectory_header_num_samples_per_shot"] = int(params["num_samples_per_shot"])
        if params.get("img_size") is not None:
            metadata["trajectory_img_size"] = np.asarray(params["img_size"], dtype=np.int64)
        trajectory = np.asarray(trajectory, dtype=np.float32).copy()
    else:
        raise ValueError(
            f"Unsupported trajectory_format={trajectory_format!r}. Expected 'npy' or 'mrinufft_bin'."
        )

    trajectory = trajectory.reshape(-1, len(image_shape))
    metadata["trajectory_num_samples_without_acs"] = int(trajectory.shape[0])

    if use_acs and acs_path is not None:
        acs = np.load(Path(acs_path)).astype(np.float32).reshape(-1, len(image_shape))
        metadata["trajectory_acs_num_samples"] = int(acs.shape[0])
        trajectory = np.concatenate([trajectory, acs], axis=0)

    if normalize_to_unit_box:
        trajectory = normalize_trajectory_samples(trajectory, image_shape=image_shape)

    if clip:
        trajectory = np.clip(trajectory, -0.5, 0.5)

    trajectory = trajectory.astype(np.float32, copy=False)
    metadata["trajectory_num_samples"] = int(trajectory.shape[0])
    if return_metadata:
        return trajectory, metadata
    return trajectory


def build_mrinufft_operator(
    samples: np.ndarray,
    image_shape: Sequence[int],
    n_coils: int,
    backend: str,
    density_compensation: Any = True,
    squeeze_dims: bool = True,
    upsampfac: Optional[float] = None,
    autograd: bool = False,
    device: Optional[Union[str, torch.device]] = None,
):
    get_operator = import_mrinufft_get_operator()
    requested_backend = backend
    if backend == "pytorch" or backend == "torchkbnufft":
        # Select device-specific torchkbnufft backend based on sample/operator requirements
        if device is not None:
            dev_str = str(device)
        else:
            dev_str = str(getattr(samples, "device", "cpu"))
        backend = "torchkbnufft-gpu" if "cuda" in dev_str else "torchkbnufft-cpu"
    else:
        dev_str = str(device) if device is not None else str(getattr(samples, "device", "cpu"))

    logging.info(
        "[NUFFT] build operator requested_backend=%s resolved_backend=%s density_compensation=%s autograd=%s device=%s",
        requested_backend,
        backend,
        density_compensation,
        autograd,
        dev_str,
    )

    is_torchkbnufft = backend.startswith("torchkbnufft")

    density_value = density_compensation
    if density_compensation is True and backend == "cufinufft":
        density_value = "voronoi"
        logging.warning(
            "[NUFFT] density_compensation=True maps to PIPE in mri-nufft, which is unsupported for cufinufft; "
            "falling back to density='voronoi'."
        )

    # If autograd is requested and we are not using torchkbnufft, we must use squeeze_dims=False
    if autograd and not is_torchkbnufft:
        squeeze_dims = False

    op_factory = get_operator(backend)
    kwargs: Dict[str, Any] = dict(
        shape=tuple(int(v) for v in image_shape),
        n_coils=int(n_coils),
        density=density_value,
        squeeze_dims=squeeze_dims,
    )
    if upsampfac is not None:
        kwargs["upsampfac"] = upsampfac

    op = op_factory(samples, **kwargs)

    if autograd and not is_torchkbnufft:
        if hasattr(op, "make_autograd"):
            op = op.make_autograd()

    return op


def complex_gaussian_noise_like(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return torch.zeros_like(x)
    scale = std / np.sqrt(2.0)
    return scale * (torch.randn_like(x) + 1j * torch.randn_like(x))


def prepare_sens_maps(
    sens_maps: ArrayLike,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(sens_maps, np.ndarray):
        sens_maps = torch.from_numpy(sens_maps)
    if device is not None:
        sens_maps = sens_maps.to(device)

    # Handle batch dimension: squeeze if first dim is 1 (common from dataset loaders)
    while sens_maps.ndim >= 5 and sens_maps.shape[0] == 1:
        sens_maps = sens_maps.squeeze(0)

    # Convert to complex
    if torch.is_complex(sens_maps):
        smaps_complex = sens_maps
    elif sens_maps.shape[-1] == 2:
        # Real/imag pair in last dimension: view as complex
        smaps_complex = torch.view_as_complex(sens_maps.contiguous())
    else:
        # Not explicitly real/imag; assume already conceptually complex (coils, z, y, x)
        smaps_complex = torch.complex(sens_maps, torch.zeros_like(sens_maps))

    # After complex conversion, should be [coils, z, y, x]
    if smaps_complex.ndim != 4:
        raise ValueError(
            f"Sensitivity maps must have 4 dims after complex conversion, got {smaps_complex.shape}."
        )

    # Move coil dimension to the front if not already there
    return smaps_complex.movedim(-1, 0).contiguous()
