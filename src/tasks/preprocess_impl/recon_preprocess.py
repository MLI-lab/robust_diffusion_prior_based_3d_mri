from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf, open_dict
from tqdm import tqdm

from src.prefect.caching import CacheableDictConfig
from src.problem_trafos.dataset_trafo.mask_utils import get_gaussian_2d_mask_rej, _normalize_calib
from src.problem_trafos.dataset_trafo.volume_preprocess_utils import (
    interpolate_volume,
    scale_by_kspace_norm,
)
from src.problem_trafos.utils.bart_utils import coil_compress_kspace_3d, compute_sens_maps_3d, import_bart
from src.problem_trafos.utils.noncartesian_mri import (
    build_mrinufft_operator,
    complex_gaussian_noise_like,
    ensure_complex_torch,
    generate_radial_trajectory,
    load_noncartesian_trajectory,
)
from src.problem_trafos.utils.scaling_factors import get_scaling_factor
from src.utils.fftn3d import fft3c, fftshift, ifft3c, ifftshift
from src.tasks.dataset_pipeline_utils import cache_base_from_subfolder, concrete_path_resolver


def _resolve(raw_cfg: Any, path_resolver: Callable) -> Any:
    if raw_cfg is None:
        return None
    return path_resolver(raw_cfg)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _interp_factor_is_spec(value: Any) -> bool:
    """True when target_interpolate_by_factor is a list/interval, not a scalar."""
    return isinstance(value, (list, tuple, ListConfig))


def _resolve_interp_factor(cfg: Any, fname_name: str) -> float:
    """Per-file target interpolation factor."""
    value = getattr(cfg, "target_interpolate_by_factor", 1.0)
    if not _interp_factor_is_spec(value):
        return float(value)

    from src.tasks.preprocess_impl.generic_h5_preprocess import _resolve_factors_for_file

    return _resolve_factors_for_file(
        fname_name=fname_name,
        base_seed=int(getattr(cfg, "target_interpolate_factor_seed", 1234) or 1234),
        count=1,
        target_interpolate_by_factor=value,
        target_interpolate_factor_is_interval=bool(
            getattr(cfg, "target_interpolate_factor_is_interval", False)
        ),
        target_interpolate_factor_step=getattr(cfg, "target_interpolate_factor_step", None),
        target_interpolate_factor_weights=getattr(cfg, "target_interpolate_factor_weights", None),
        target_interpolate_factor_with_replacement=getattr(
            cfg, "target_interpolate_factor_with_replacement", None
        ),
        target_interpolate_factor_space=str(getattr(cfg, "target_interpolate_factor_space", "factor") or "factor"),
    )[0]


def _squeeze_bart_sensmaps(sens_maps_np: np.ndarray) -> np.ndarray:
    while sens_maps_np.ndim > 4:
        singleton_axes = [axis for axis, size in enumerate(sens_maps_np.shape) if size == 1]
        if not singleton_axes:
            break
        sens_maps_np = np.squeeze(sens_maps_np, axis=singleton_axes[0])
    return sens_maps_np


def _center_crop_kspace_torch_czyx(kspace: torch.Tensor, scale_factor: float) -> torch.Tensor:
    if scale_factor == 1.0:
        return kspace
    if scale_factor <= 0.0 or scale_factor > 1.0:
        raise ValueError(f"k-space crop scale_factor must be in (0, 1], got {scale_factor}.")
    _coils, z, y, x, _ri = kspace.shape
    nz = int(round(z * scale_factor))
    ny = int(round(y * scale_factor))
    nx = int(round(x * scale_factor))
    sz = (z - nz) // 2
    sy = (y - ny) // 2
    sx = (x - nx) // 2
    return kspace[:, sz:sz + nz, sy:sy + ny, sx:sx + nx, :].contiguous()


def _fft1c(data: torch.Tensor, dim: int, shifts_enable=(True, True)) -> torch.Tensor:
    if shifts_enable[0]:
        data = ifftshift(data, dim=[dim])
    data = torch.view_as_real(torch.fft.fftn(torch.view_as_complex(data), dim=[dim], norm="ortho"))
    if shifts_enable[1]:
        data = fftshift(data, dim=[dim])
    return data


def _load_external_sensmaps(sensmap_dir: Optional[str], fname: Path) -> Optional[np.ndarray]:
    if sensmap_dir is None:
        return None
    candidates = [
        Path(sensmap_dir) / (fname.stem + "_sensmap" + fname.suffix),
        Path(sensmap_dir) / fname.name,
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        with h5py.File(cand, "r", locking=False) as hf:
            for key in ("sensitivity_maps", "sens_maps"):
                if key in hf:
                    return hf[key][()]
            for key in hf.keys():
                obj = hf[key]
                if isinstance(obj, h5py.Dataset) and getattr(obj, "ndim", 0) >= 3:
                    return obj[()]
    logging.warning("No external sensmap found for %s in %s", fname.name, sensmap_dir)
    return None


def _match_sensmaps_to_kspace(sens_maps_np: np.ndarray, n_coils: int, spatial_shape: tuple[int, int, int]) -> np.ndarray:
    import itertools

    if sens_maps_np is None:
        raise ValueError("Sensitivity maps are required to build prepared test targets.")
    candidate_coil_axes = [axis for axis, size in enumerate(sens_maps_np.shape) if size == n_coils]
    for coil_axis in candidate_coil_axes:
        cfirst = np.moveaxis(sens_maps_np, coil_axis, 0)
        sm_spatial = cfirst.shape[1:]
        for perm in itertools.permutations(range(3)):
            if tuple(sm_spatial[i] for i in perm) == tuple(spatial_shape):
                return np.transpose(cfirst, (0, perm[0] + 1, perm[1] + 1, perm[2] + 1)).astype(np.complex64)
    raise ValueError(
        f"Cannot align sensmaps shape {sens_maps_np.shape} to n_coils={n_coils}, spatial_shape={spatial_shape}."
    )


def _sensmaps_coil_last(sens_maps_np: np.ndarray, n_coils: int, spatial_shape: tuple[int, int, int]) -> np.ndarray:
    return np.moveaxis(_match_sensmaps_to_kspace(sens_maps_np, n_coils, spatial_shape), 0, -1).astype(np.complex64)


def _combine_mvue(coil_images_ri: torch.Tensor, sens_maps_np: np.ndarray, device: str) -> torch.Tensor:
    coil_images = torch.view_as_complex(coil_images_ri.contiguous())
    n_coils = int(coil_images.shape[0])
    sens = _match_sensmaps_to_kspace(sens_maps_np, n_coils, tuple(coil_images.shape[-3:]))
    sens_t = torch.from_numpy(sens).to(device)
    nonzero = sens_t != 0
    sens_safe = sens_t.clone()
    if nonzero.any():
        sens_safe[~nonzero] = sens_safe[nonzero].abs().min() + 0j
    norm = torch.abs(sens_safe).square().sum(dim=0).sqrt()
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return torch.view_as_real((coil_images * sens_safe.conj()).sum(dim=0) / norm).float()




def _estimate_noncartesian_sensmaps_from_observed(
    observed_complex: torch.Tensor,
    op: Any,
    ecalib_calib_size: Optional[int],
) -> np.ndarray:
    coil_images = ensure_complex_torch(op.adj_op(observed_complex), device=observed_complex.device)
    gridded_kspace_ri = fft3c(torch.view_as_real(coil_images.contiguous()))
    return _squeeze_bart_sensmaps(
        compute_sens_maps_3d(gridded_kspace_ri.contiguous(), ecalib_calib_size=ecalib_calib_size)
    )


def _compute_prepared_observation_scaling_factor(
    *,
    cfg: Any,
    observation: torch.Tensor,
    rep_shape: tuple[int, ...],
    sens_maps: Optional[np.ndarray] = None,
    trajectory: Optional[np.ndarray] = None,
    image_shape: Optional[tuple[int, int, int]] = None,
) -> float:
    mode = getattr(cfg, "observation_scaling_mode", "default")
    if mode is None or str(mode).lower() in ("none", "false", "off"):
        return 1.0

    fwd_trafo = None
    if str(mode) != "default":
        if sens_maps is None or trajectory is None or image_shape is None:
            raise ValueError(
                f"observation_scaling_mode={mode!r} requires non-Cartesian sens_maps, trajectory, and image_shape."
            )
        from src.problem_trafos.fwd_trafo.mri_3d_noncartesian_trafo import NonCartesianMRI3DTrafo

        fwd_trafo = NonCartesianMRI3DTrafo(
            mask_enabled=False,
            mask_type=None,
            mask_accelerations=None,
            mask_center_fractions=None,
            mask_seed=getattr(cfg, "mask_seed", 1234),
            include_sensitivitymaps=True,
            sensitivitymaps_complex=True,
            sensitivitymaps_fillouter=bool(getattr(cfg, "sensitivitymaps_fillouter", True)),
            wrapped_2d_mode=False,
            nufft_backend=str(getattr(cfg, "nufft_backend", "pytorch")),
            density_compensation=getattr(cfg, "density_compensation", False),
            upsampfac=getattr(cfg, "upsampfac", None),
        )
        fwd_trafo.calibrate(
            observation,
            {
                "sens_maps": sens_maps,
                "trajectory": trajectory,
                "image_shape": np.asarray(image_shape, dtype=np.int64),
            },
        )
    scaling_factor = get_scaling_factor(
        mode=str(mode),
        fwd_trafo=fwd_trafo,
        observation=observation,
        rep_shape=rep_shape,
        show_tqdm=bool(getattr(cfg, "observation_scaling_show_tqdm", False)),
        max_iter=int(getattr(cfg, "observation_scaling_max_iter", 10)),
        rtol=float(getattr(cfg, "observation_scaling_rtol", 1e-2)),
        lam=float(getattr(cfg, "observation_scaling_lam", 1e-6)),
    )
    scaling_factor *= float(getattr(cfg, "observation_scaling_constant", 1.0))
    return float(scaling_factor)


def _observation_scaling_reference_numel(cfg: Any, rep_shape: tuple[int, ...]) -> Optional[int]:
    """Element count the stored scaling factor is tied to, or None if it is shape-free."""
    mode = getattr(cfg, "observation_scaling_mode", "default")
    if mode is None or str(mode).lower() in ("none", "false", "off"):
        return None
    if str(mode) != "default":
        return None
    return int(np.prod(rep_shape))

def _trajectory_provenance_attrs(cfg: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    string_keys = (
        "trajectory_path",
        "trajectory_format",
        "trajectory_source_url",
        "trajectory_sha256",
    )
    for key in string_keys:
        value = getattr(cfg, key, None)
        if value is not None:
            attrs[key] = str(value)

    value = getattr(cfg, "trajectory_dwell_time", None)
    if value is not None:
        attrs["trajectory_dwell_time"] = float(value)

    value = getattr(cfg, "trajectory_expected_size", None)
    if value is not None:
        attrs["trajectory_expected_size"] = int(value)

    return attrs


def _trajectory_metadata_attrs(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metadata:
        return {}
    attrs: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            attrs[key] = value
        elif isinstance(value, (list, tuple)):
            attrs[key] = np.asarray(value)
        elif isinstance(value, (np.integer, int)):
            attrs[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            attrs[key] = float(value)
        elif isinstance(value, (str, bytes)):
            attrs[key] = value
        else:
            attrs[key] = str(value)
    return attrs


def _resolve_configured_trajectory_path(cfg: Any, path_resolver: Callable) -> None:
    if str(getattr(cfg, "sampling_kind", "cartesian")) != "noncartesian":
        return
    if bool(getattr(cfg, "trajectory_compute_on_the_fly", True)):
        return
    trajectory_path = getattr(cfg, "trajectory_path", None)
    if trajectory_path is None:
        return
    resolved = _resolve(trajectory_path, path_resolver)
    try:
        with open_dict(cfg):
            cfg.trajectory_path = str(resolved)
    except TypeError:
        setattr(cfg, "trajectory_path", str(resolved))


def _normalize_accelerations(acc: Any) -> tuple[int, ...]:
    if isinstance(acc, (list, tuple, ListConfig)):
        values = tuple(int(round(float(v))) for v in acc)
    else:
        values = (int(round(float(acc))),)
    if any(v <= 0 for v in values):
        raise ValueError(f"mask_accelerations entries must be positive, got {acc}.")
    return values


def _center_calib_slices(shape: tuple[int, int], calib: Optional[Sequence[int]]) -> tuple[slice, slice]:
    if calib is None:
        return slice(0, 0), slice(0, 0)
    calib_y, calib_x = _normalize_calib(calib)
    calib_y = min(calib_y, shape[0])
    calib_x = min(calib_x, shape[1])
    y0 = (shape[0] - calib_y) // 2
    x0 = (shape[1] - calib_x) // 2
    return slice(y0, y0 + calib_y), slice(x0, x0 + calib_x)


def _apply_center_calib(mask: np.ndarray, calib: Optional[Sequence[int]]) -> None:
    ys, xs = _center_calib_slices(mask.shape, calib)
    mask[ys, xs] = 1.0


def _factor_2d_acceleration(acc: tuple[int, ...]) -> tuple[int, int]:
    if len(acc) >= 2:
        return acc[0], acc[1]
    value = acc[0]
    first = int(np.floor(np.sqrt(value)))
    while first > 1 and value % first != 0:
        first -= 1
    return first, int(np.ceil(value / first))


def _target_sample_count(shape: tuple[int, int], acc: tuple[int, ...]) -> int:
    total_acc = float(np.prod(acc))
    return max(1, int(round(float(np.prod(shape)) / total_acc)))


def _assert_calib_not_oversampled(mask: np.ndarray, target_count: int, mask_type: str) -> None:
    calib_count = int(mask.sum())
    if calib_count > target_count:
        raise ValueError(
            f"{mask_type} calibration region selects {calib_count} samples, "
            f"which exceeds the requested total sample count {target_count}."
        )


def _uniform_random_points_2d(shape: tuple[int, int], acc: tuple[int, ...], calib: Optional[Sequence[int]], seed: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    _apply_center_calib(mask, calib)
    target_count = _target_sample_count(shape, acc)
    _assert_calib_not_oversampled(mask, target_count, "UniformRandomPoints2D")
    remaining = target_count - int(mask.sum())
    if remaining <= 0:
        return mask
    candidates = np.argwhere(mask == 0)
    rng = np.random.default_rng(seed)
    chosen = candidates[rng.choice(len(candidates), size=remaining, replace=False)]
    mask[chosen[:, 0], chosen[:, 1]] = 1.0
    return mask


def _line_calib_indices(shape: tuple[int, int], axis: int, calib: Optional[Sequence[int]]) -> set[int]:
    ys, xs = _center_calib_slices(shape, calib)
    selected_slice = ys if axis == 0 else xs
    return set(range(selected_slice.start, selected_slice.stop))


def _uniform_random_lines_2d(
    shape: tuple[int, int],
    acc: tuple[int, ...],
    calib: Optional[Sequence[int]],
    seed: int,
    axis: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    _apply_center_calib(mask, calib)
    target_count = _target_sample_count(shape, acc)
    _assert_calib_not_oversampled(mask, target_count, "UniformRandomLines2D")

    num_lines = shape[axis]
    line_width = shape[1 - axis]
    remaining_samples = target_count - int(mask.sum())
    remaining_lines = max(0, int(round(remaining_samples / float(line_width))))
    if remaining_lines <= 0:
        return mask

    calib_line_indices = _line_calib_indices(shape, axis, calib)
    candidates = np.asarray([idx for idx in range(num_lines) if idx not in calib_line_indices], dtype=np.int64)
    remaining_lines = min(remaining_lines, len(candidates))
    rng = np.random.default_rng(seed)
    line_idx = sorted(candidates[rng.choice(len(candidates), size=remaining_lines, replace=False)].tolist())
    if axis == 0:
        mask[line_idx, :] = 1.0
    else:
        mask[:, line_idx] = 1.0
    _apply_center_calib(mask, calib)
    return mask


def _equispaced_points_2d(shape: tuple[int, int], acc: tuple[int, ...], calib: Optional[Sequence[int]], offset: tuple[int, int]) -> np.ndarray:
    step_y, step_x = _factor_2d_acceleration(acc)
    mask = np.zeros(shape, dtype=np.float32)
    mask[offset[0] % step_y::step_y, offset[1] % step_x::step_x] = 1.0
    _apply_center_calib(mask, calib)
    return mask


def _equispaced_lines_2d(shape: tuple[int, int], acc: tuple[int, ...], calib: Optional[Sequence[int]], offset: int, axis: int) -> np.ndarray:
    step = acc[0]
    mask = np.zeros(shape, dtype=np.float32)
    if axis == 0:
        mask[offset % step::step, :] = 1.0
    else:
        mask[:, offset % step::step] = 1.0
    _apply_center_calib(mask, calib)
    return mask


def _caipirinha_2d(
    shape: tuple[int, int],
    acc: tuple[int, ...],
    calib: Optional[Sequence[int]],
    offset: tuple[int, int],
    shift: int,
) -> np.ndarray:
    step_y, step_x = _factor_2d_acceleration(acc)
    mask = np.zeros(shape, dtype=np.float32)
    for y in range(offset[0] % step_y, shape[0], step_y):
        block = (y - (offset[0] % step_y)) // step_y
        x_offset = (offset[1] + block * shift) % step_x
        mask[y, x_offset::step_x] = 1.0
    _apply_center_calib(mask, calib)
    return mask


def _cartesian_mask(shape: tuple[int, int], cfg: Any, seed: int, device: str) -> torch.Tensor:
    mask_type = str(getattr(cfg, "mask_type", "Poisson2D"))
    acc = getattr(cfg, "mask_accelerations", 4.0)
    acc_values = _normalize_accelerations(acc)
    calib = _normalize_calib(getattr(cfg, "mask_calib", None))
    if mask_type == "Poisson2D":
        from sigpy.mri.samp import poisson
        if calib is None:
            mask_np = poisson(list(shape), acc, seed=seed)
        else:
            mask_np = poisson(list(shape), acc, calib=calib, seed=seed)
        return torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(-1).to(device)
    if mask_type == "Gaussian2D":
        return get_gaussian_2d_mask_rej(list(shape), acc_factor=acc, seed=seed, calib=calib).unsqueeze(-1).to(device)

    line_axis = int(getattr(cfg, "mask_line_axis", 0))
    if line_axis not in (0, 1):
        raise ValueError(f"mask_line_axis must be 0 or 1, got {line_axis}.")
    raw_offset = getattr(cfg, "mask_offset", 0)
    if isinstance(raw_offset, (list, tuple, ListConfig)):
        offset = (int(raw_offset[0]), int(raw_offset[1] if len(raw_offset) > 1 else raw_offset[0]))
    else:
        offset = (int(raw_offset), int(raw_offset))
    mask_builders = {
        "UniformRandomPoints2D": lambda: _uniform_random_points_2d(shape, acc_values, calib, seed),
        "UniformRandomLines2D": lambda: _uniform_random_lines_2d(shape, acc_values, calib, seed, line_axis),
        "EquispacedPoints2D": lambda: _equispaced_points_2d(shape, acc_values, calib, offset),
        "EquispacedLines2D": lambda: _equispaced_lines_2d(shape, acc_values, calib, offset[line_axis], line_axis),
        "CAIPIRINHA2D": lambda: _caipirinha_2d(
            shape,
            acc_values,
            calib,
            offset,
            int(getattr(cfg, "mask_caipirinha_shift", 1)),
        ),
    }
    if mask_type not in mask_builders:
        raise ValueError(
            f"Unknown mask_type={mask_type!r}. Supported Cartesian masks are Poisson2D, Gaussian2D, "
            "UniformRandomPoints2D, UniformRandomLines2D, EquispacedPoints2D, "
            "EquispacedLines2D, and CAIPIRINHA2D."
        )
    mask_np = mask_builders[mask_type]()
    return torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(-1).to(device)


def _write_h5(path: Path, datasets: Dict[str, Any], attrs: Dict[str, Any]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.h5")
    try:
        with h5py.File(tmp, "w") as hf:
            for key, value in datasets.items():
                if value is None:
                    continue
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                hf.create_dataset(key, data=value)
            for key, value in attrs.items():
                try:
                    hf.attrs[key] = value
                except Exception:
                    pass
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


@dataclass
class _SourceReconVolume:
    attrs: Dict[str, Any]
    kspace_np: Optional[np.ndarray]
    source_target: Optional[np.ndarray]
    source_sens: Optional[np.ndarray]


@dataclass
class _FullySampledGroundTruth:
    target: torch.Tensor
    kspace: torch.Tensor
    source_sens: Optional[np.ndarray]
    kspace_vol_norm: float
    image_shape: tuple[int, int, int]
    target_type: str
    coil_compression_virtual_coils: Optional[int]


@dataclass
class _UndersampledObservation:
    sampling_kind: str
    observation_ri: torch.Tensor
    kspace_to_store: torch.Tensor
    mask: Optional[torch.Tensor] = None
    trajectory: Optional[np.ndarray] = None
    trajectory_metadata: Optional[Dict[str, Any]] = None


@dataclass
class _ReconCache:
    pseudoinverse: torch.Tensor
    scaling_factor: float
    sens_maps: Optional[np.ndarray]
    sens_maps_from_observed: bool
    # numel `scaling_factor` was normalised against, or None when the mode makes it
    # shape-independent. See _observation_scaling_reference_numel.
    scaling_reference_numel: Optional[int] = None


def _load_source_recon_volume(fname: Path, sensmap_dir: Optional[str], cfg: Any) -> _SourceReconVolume:
    with h5py.File(fname, "r", locking=False) as hf:
        kspace_np = hf["kspace"][:] if "kspace" in hf else None
        attrs = dict(hf.attrs)
        source_sens = None
        sens_key = getattr(cfg, "source_sensmaps_key", "sensitivity_maps")
        if sens_key and sens_key in hf:
            source_sens = hf[sens_key][()]
        source_target = None
        recons_key = getattr(cfg, "source_recons_key", "reconstruction_mvue")
        if recons_key and recons_key in hf:
            source_target = hf[recons_key][:]

    external_sens = _load_external_sensmaps(sensmap_dir, fname)
    if external_sens is not None:
        source_sens = external_sens

    return _SourceReconVolume(
        attrs=attrs,
        kspace_np=kspace_np,
        source_target=source_target,
        source_sens=source_sens,
    )


def _source_kspace_for_ground_truth(
    source: _SourceReconVolume,
    fname: Path,
    cfg: Any,
    dataset_cfg: Any,
    device: str,
) -> torch.Tensor:
    if source.kspace_np is not None:
        kspace = torch.view_as_real(torch.from_numpy(source.kspace_np.astype(np.complex64))).to(device)
        if bool(getattr(dataset_cfg, "dataset_is_3d", True)):
            kspace = kspace.movedim(1, 0)  # (Coils, Z, Y, X, 2)
            if bool(getattr(dataset_cfg, "apply_fft1c_on_readout_dim", False)):
                kspace = _fft1c(
                    kspace,
                    dim=1,
                    shifts_enable=tuple(getattr(dataset_cfg, "apply_fft1c_on_readout_dim_shifts", (True, True))),
                )
        return kspace

    if source.source_target is None:
        raise KeyError(
            f"{fname} contains neither 'kspace' nor source_recons_key={getattr(cfg, 'source_recons_key', 'reconstruction_mvue')!r}."
        )
    target_for_kspace = torch.from_numpy(source.source_target).to(device)
    if torch.is_complex(target_for_kspace):
        target_for_kspace = torch.view_as_real(target_for_kspace.contiguous())
    elif target_for_kspace.ndim >= 4 and target_for_kspace.shape[-1] == 1:
        target_for_kspace = target_for_kspace.squeeze(-1)
    if target_for_kspace.ndim == 3:
        target_for_kspace = torch.stack([target_for_kspace, torch.zeros_like(target_for_kspace)], dim=-1)
    if not (target_for_kspace.ndim == 4 and target_for_kspace.shape[-1] == 2):
        raise ValueError(
            f"Cannot synthesize kspace from source target shape {tuple(target_for_kspace.shape)} in {fname}. "
            "Expected (Z, Y, X), (Z, Y, X, 1), or complex/(Z, Y, X, 2)."
        )
    # Magnitude-only Lüsebrink data has no measured k-space. Use the
    # zero-phase single-coil image implied by reconstruction_rss.
    return fft3c(target_for_kspace.unsqueeze(0).contiguous())


def _prepare_fullysampled_ground_truth(
    source: _SourceReconVolume,
    fname: Path,
    cfg: Any,
    dataset_cfg: Any,
    device: str,
) -> _FullySampledGroundTruth:
    """Prepare only the fully sampled reference target and its matching k-space."""
    kspace = _source_kspace_for_ground_truth(source, fname, cfg, dataset_cfg, device)

    interp_factor = _resolve_interp_factor(cfg, fname.name)
    interp_method = str(getattr(cfg, "target_interpolation_method", "fourier"))
    if interp_factor != 1.0:
        if interp_method != "fourier":
            raise ValueError(
                "Prepared recon interpolation must be Fourier/k-space cropping; "
                f"got target_interpolation_method={interp_method!r}."
            )
        kspace = _center_crop_kspace_torch_czyx(kspace, interp_factor)

    coil_compression_virtual_coils = getattr(cfg, "coil_compression_virtual_coils", None)
    kspace = coil_compress_kspace_3d(kspace, coil_compression_virtual_coils)

    kspace_vol_norm = float(torch.linalg.norm(torch.view_as_complex(kspace.contiguous())).detach().cpu())
    target_type = str(getattr(cfg, "target_type", "fullysampled_rec"))
    full_sensmap_mode = str(getattr(cfg, "full_sensmap_mode", "source"))
    ecalib_calib_size = getattr(cfg, "ecalib_calib_size", None)
    source_sens = source.source_sens
    if target_type in ("fullysampled_rec", "mvue") and full_sensmap_mode == "bart_from_kspace":
        source_sens = _squeeze_bart_sensmaps(compute_sens_maps_3d(kspace, ecalib_calib_size=ecalib_calib_size))
    elif source_sens is None and target_type in ("fullysampled_rec", "mvue"):
        raise ValueError("source sensitivity maps are required unless full_sensmap_mode=bart_from_kspace.")

    if source.source_target is not None and target_type != "fullysampled_rec":
        target = torch.from_numpy(source.source_target).to(device)
        if torch.is_complex(target):
            target = torch.view_as_real(target.contiguous())
        if interp_factor != 1.0:
            target = interpolate_volume(target, interp_factor, interp_method)
    else:
        target = _combine_mvue(ifft3c(kspace), source_sens, device)

    if bool(getattr(cfg, "scale_target_by_kspacenorm", False)):
        target = scale_by_kspace_norm(target, kspace_vol_norm)
    scaling = float(getattr(cfg, "target_scaling_factor", 1.0))
    if scaling != 1.0:
        target = target * scaling

    return _FullySampledGroundTruth(
        target=target,
        kspace=kspace,
        source_sens=source_sens,
        kspace_vol_norm=kspace_vol_norm,
        image_shape=tuple(int(v) for v in target.shape[:3]),
        target_type=target_type,
        coil_compression_virtual_coils=coil_compression_virtual_coils,
    )


def _create_undersampled_observation(
    ground_truth: _FullySampledGroundTruth,
    cfg: Any,
    device: str,
) -> _UndersampledObservation:
    """Create the measurement that reconstruction will receive."""
    sampling_kind = str(getattr(cfg, "sampling_kind", "cartesian"))
    seed = int(getattr(cfg, "mask_seed", getattr(cfg, "seed", 1234)))

    if sampling_kind == "cartesian":
        mask = _cartesian_mask((int(ground_truth.kspace.shape[-3]), int(ground_truth.kspace.shape[-2])), cfg, seed, device)
        observed = ground_truth.kspace * mask.to(ground_truth.kspace.device) + 0.0
        kspace_to_store = torch.view_as_complex(observed.movedim(0, 1).contiguous())
        return _UndersampledObservation(
            sampling_kind=sampling_kind,
            observation_ri=torch.view_as_real(kspace_to_store.contiguous()),
            kspace_to_store=kspace_to_store,
            mask=mask,
        )

    if sampling_kind == "noncartesian":
        trajectory_metadata: Dict[str, Any] = {}
        if bool(getattr(cfg, "trajectory_compute_on_the_fly", True)):
            num_spokes = int(getattr(cfg, "trajectory_num_spokes", 8192))
            trajectory = generate_radial_trajectory(
                num_spokes=num_spokes,
                image_shape=ground_truth.image_shape,
                readout_oversample=getattr(cfg, "trajectory_readout_oversample", None),
                gamma=float(getattr(cfg, "trajectory_gamma", 1.0)),
                use_acs=bool(getattr(cfg, "trajectory_use_acs", True)),
                acs_size=int(getattr(cfg, "trajectory_acs_size", 24)) if bool(getattr(cfg, "trajectory_use_acs", True)) else None,
                normalize_to_unit_box=bool(getattr(cfg, "trajectory_normalize", True)),
                clip=bool(getattr(cfg, "trajectory_clip", True)),
            )
            trajectory_metadata["trajectory_num_shots"] = num_spokes
            trajectory_metadata["trajectory_num_samples"] = int(trajectory.shape[0])
            if num_spokes > 0 and int(trajectory.shape[0]) % num_spokes == 0:
                trajectory_metadata["trajectory_samples_per_shot"] = int(trajectory.shape[0]) // num_spokes
        else:
            trajectory, trajectory_metadata = load_noncartesian_trajectory(
                trajectory_path=getattr(cfg, "trajectory_path"),
                acs_path=getattr(cfg, "trajectory_acs_path", None),
                use_acs=bool(getattr(cfg, "trajectory_use_acs", True)),
                image_shape=ground_truth.image_shape,
                normalize_to_unit_box=bool(getattr(cfg, "trajectory_normalize", True)),
                clip=bool(getattr(cfg, "trajectory_clip", True)),
                trajectory_format=str(getattr(cfg, "trajectory_format", "npy")),
                dwell_time=float(getattr(cfg, "trajectory_dwell_time", 0.005)),
                source_url=getattr(cfg, "trajectory_source_url", None),
                sha256=getattr(cfg, "trajectory_sha256", None),
                expected_size=getattr(cfg, "trajectory_expected_size", None),
                return_metadata=True,
            )
        sens = _match_sensmaps_to_kspace(ground_truth.source_sens, int(ground_truth.kspace.shape[0]), ground_truth.image_shape)
        sens_t = torch.from_numpy(sens).to(device)
        coil_images = torch.view_as_complex(ground_truth.target.unsqueeze(0).contiguous()) * sens_t
        op = build_mrinufft_operator(
            samples=trajectory,
            image_shape=ground_truth.image_shape,
            n_coils=int(ground_truth.kspace.shape[0]),
            backend=str(getattr(cfg, "nufft_backend", "pytorch")),
            density_compensation=getattr(cfg, "density_compensation", False),
            squeeze_dims=True,
            upsampfac=getattr(cfg, "upsampfac", None),
            device=device,
        )
        observed_complex = ensure_complex_torch(op.op(coil_images), device=torch.device(device))
        observed_complex = observed_complex + complex_gaussian_noise_like(observed_complex, float(getattr(cfg, "noise_std", 0.0)))
        return _UndersampledObservation(
            sampling_kind=sampling_kind,
            observation_ri=torch.view_as_real(observed_complex.contiguous()),
            kspace_to_store=observed_complex.contiguous(),
            trajectory=trajectory,
            trajectory_metadata=trajectory_metadata,
        )

    raise ValueError(f"Unknown sampling_kind={sampling_kind}.")


def _build_reconstruction_cache_from_observation(
    observation: _UndersampledObservation,
    *,
    cfg: Any,
    image_shape: tuple[int, int, int],
    target_type: str,
    full_sens_maps: Optional[np.ndarray],
    device: str,
) -> _ReconCache:
    """Cache reconstruction helpers from reconstruction-time inputs only."""
    ecalib_calib_size = getattr(cfg, "ecalib_calib_size", None)
    rep_shape = (*image_shape, 2)
    sensmap_mode = str(getattr(cfg, "sensmap_mode", "bart_from_observed"))
    full_sensmap_modes = {"full", "full_acquisition", "fullysampled", "fullysampled_acquisition", "source"}

    if observation.sampling_kind == "cartesian":
        coil_images = ifft3c(observation.observation_ri.movedim(1, 0).contiguous())
        if target_type == "rss" or int(observation.observation_ri.shape[1]) == 1:
            sens_maps = None
            pseudoinverse = coil_images[0]
            sens_maps_from_observed = False
        else:
            if sensmap_mode == "bart_from_observed":
                sens_maps = _squeeze_bart_sensmaps(compute_sens_maps_3d(
                    observation.observation_ri.movedim(1, 0).contiguous(),
                    ecalib_calib_size=ecalib_calib_size,
                ))
                sens_maps_from_observed = True
            elif sensmap_mode in full_sensmap_modes:
                if full_sens_maps is None:
                    raise ValueError(f"sensmap_mode={sensmap_mode!r} requires full-acquisition sensitivity maps.")
                sens_maps = _sensmaps_coil_last(full_sens_maps, int(observation.observation_ri.shape[1]), image_shape)
                sens_maps_from_observed = False
            else:
                raise ValueError(
                    "Prepared Cartesian recon cache supports sensmap_mode='bart_from_observed' "
                    f"or a full-acquisition mode; got {sensmap_mode!r}."
                )
            pseudoinverse = _combine_mvue(coil_images, sens_maps, device)
        scaling_factor = _compute_prepared_observation_scaling_factor(
            cfg=cfg,
            observation=observation.observation_ri,
            rep_shape=rep_shape,
        )
        return _ReconCache(
            pseudoinverse=pseudoinverse,
            scaling_factor=scaling_factor,
            sens_maps=sens_maps,
            sens_maps_from_observed=sens_maps_from_observed,
            scaling_reference_numel=_observation_scaling_reference_numel(cfg, rep_shape),
        )

    if observation.sampling_kind == "noncartesian":
        if observation.trajectory is None:
            raise ValueError("Prepared non-Cartesian observations require a trajectory for cache computation.")
        n_coils = int(observation.observation_ri.shape[0]) if observation.observation_ri.ndim >= 2 else 1
        op = build_mrinufft_operator(
            samples=observation.trajectory,
            image_shape=image_shape,
            n_coils=n_coils,
            backend=str(getattr(cfg, "nufft_backend", "pytorch")),
            density_compensation=getattr(cfg, "density_compensation", False),
            squeeze_dims=True,
            upsampfac=getattr(cfg, "upsampfac", None),
            device=device,
        )
        observed_complex = ensure_complex_torch(observation.observation_ri, device=torch.device(device))
        if sensmap_mode == "bart_from_observed":
            sens_maps = _estimate_noncartesian_sensmaps_from_observed(
                observed_complex=observed_complex,
                op=op,
                ecalib_calib_size=ecalib_calib_size,
            )
            sens_maps_from_observed = True
        elif sensmap_mode in full_sensmap_modes:
            if full_sens_maps is None:
                raise ValueError(f"sensmap_mode={sensmap_mode!r} requires full-acquisition sensitivity maps.")
            sens_maps = _sensmaps_coil_last(full_sens_maps, n_coils, image_shape)
            sens_maps_from_observed = False
        else:
            raise ValueError(
                "Prepared non-Cartesian recon cache supports sensmap_mode='bart_from_observed' "
                f"or a full-acquisition mode; got {sensmap_mode!r}."
            )
        coil_pseudoinverse = ensure_complex_torch(op.adj_op(observed_complex), device=torch.device(device))
        pseudoinverse = _combine_mvue(torch.view_as_real(coil_pseudoinverse.contiguous()), sens_maps, device)
        scaling_factor = _compute_prepared_observation_scaling_factor(
            cfg=cfg,
            observation=observation.observation_ri,
            rep_shape=rep_shape,
            sens_maps=sens_maps,
            trajectory=observation.trajectory,
            image_shape=image_shape,
        )
        return _ReconCache(
            pseudoinverse=pseudoinverse,
            scaling_factor=scaling_factor,
            sens_maps=sens_maps,
            sens_maps_from_observed=sens_maps_from_observed,
            scaling_reference_numel=_observation_scaling_reference_numel(cfg, rep_shape),
        )

    raise ValueError(f"Unknown sampling_kind={observation.sampling_kind}.")


def _prepare_one_file(
    fname: Path,
    out_path: Path,
    sensmap_dir: Optional[str],
    cfg: Any,
    dataset_cfg: Any,
    device: str,
    exists_ok: bool,
) -> None:
    if exists_ok and out_path.exists():
        try:
            with h5py.File(out_path, "r", locking=False):
                return
        except OSError:
            out_path.unlink(missing_ok=True)

    source = _load_source_recon_volume(fname, sensmap_dir, cfg)
    ground_truth = _prepare_fullysampled_ground_truth(source, fname, cfg, dataset_cfg, device)
    observation = _create_undersampled_observation(ground_truth, cfg, device)
    recon_cache = _build_reconstruction_cache_from_observation(
        observation,
        cfg=cfg,
        image_shape=ground_truth.image_shape,
        target_type=str(getattr(cfg, "target_type", "fullysampled_rec")),
        full_sens_maps=ground_truth.source_sens,
        device=device,
    )

    datasets: Dict[str, Any] = {
        "kspace": observation.kspace_to_store,
        getattr(cfg, "output_recons_key", "reconstruction_mvue"): (
            torch.view_as_complex(ground_truth.target.contiguous())
            if ground_truth.target.shape[-1] == 2 else ground_truth.target
        ),
    }
    if observation.mask is not None:
        datasets["mask"] = observation.mask
    if observation.trajectory is not None:
        datasets["trajectory"] = observation.trajectory

    output_pseudoinverse_key = getattr(cfg, "output_pseudoinverse_key", "pseudoinverse")
    if output_pseudoinverse_key is not None:
        datasets[output_pseudoinverse_key] = (
            torch.view_as_complex(recon_cache.pseudoinverse.contiguous())
            if recon_cache.pseudoinverse.shape[-1] == 2 else recon_cache.pseudoinverse
        )
    output_sensmaps_key = getattr(cfg, "output_sensmaps_key", "sens_maps")
    if output_sensmaps_key is not None and recon_cache.sens_maps is not None:
        datasets[output_sensmaps_key] = recon_cache.sens_maps

    attrs = dict(source.attrs)
    attrs.update(
        {
            "kspace_vol_norm": 1.0,
            "prepared_recon_dataset": True,
            "sensmaps_from_observed_data": recon_cache.sens_maps_from_observed,
            "sampling_kind": observation.sampling_kind,
            "image_shape": np.asarray(ground_truth.image_shape, dtype=np.int64),
            "source_file": fname.name,
            "observation_scaling_factor": recon_cache.scaling_factor,
            "coil_compression_virtual_coils": -1
            if ground_truth.coil_compression_virtual_coils is None
            else int(ground_truth.coil_compression_virtual_coils),
        }
    )
    if recon_cache.scaling_reference_numel is not None:
        # Read back by recon_task to rescale observation_scaling_factor to the shape
        # of the variable actually being optimised. Absent => leave the factor as is.
        attrs["observation_scaling_rep_numel"] = int(recon_cache.scaling_reference_numel)
    attrs.update(_trajectory_provenance_attrs(cfg))
    attrs.update(_trajectory_metadata_attrs(observation.trajectory_metadata))
    _write_h5(out_path, datasets, attrs)


def _prepare_directory(input_dir: str, output_dir: str, sensmap_dir: Optional[str], cfg: Any, dataset_cfg: Any, device: str) -> None:
    import re

    volume_filter = str(getattr(cfg, "volume_filter", ".*\\.h5"))
    limit = getattr(cfg, "volume_limit", None)
    exists_ok = bool(getattr(cfg, "exists_ok", True))
    pattern = re.compile(volume_filter)
    input_path = Path(input_dir)
    files = sorted(f for f in input_path.iterdir() if pattern.match(f.name))
    if limit is not None and int(limit) > 0:
        files = files[: int(limit)]
    if not files:
        available = sorted(p.name for p in input_path.iterdir())
        raise FileNotFoundError(
            f"Recon preprocessing found no input H5 files in {input_path} matching volume_filter={volume_filter!r}. "
            f"Available entries: {available[:10]}{'...' if len(available) > 10 else ''}. "
            "If this is a converted dataset, inspect conversion_meta.json and the fold manifest; "
            "the selected recon split may be empty."
        )
    os.makedirs(output_dir, exist_ok=True)
    for fname in tqdm(files, desc=f"Preparing recon data {input_dir}"):
        _prepare_one_file(fname, Path(output_dir) / fname.name, sensmap_dir, cfg, dataset_cfg, device, exists_ok)

    meta = {
        "prepared_recon_dataset": True,
        "recons_key": str(getattr(cfg, "output_recons_key", "reconstruction_mvue")),
        "observation_key": "kspace",
        "sensmaps_key": None if getattr(cfg, "output_sensmaps_key", "sens_maps") is None else str(getattr(cfg, "output_sensmaps_key", "sens_maps")),
        "pseudoinverse_key": None if getattr(cfg, "output_pseudoinverse_key", "pseudoinverse") is None else str(getattr(cfg, "output_pseudoinverse_key", "pseudoinverse")),
        "sensmap_coil_dim_nr": -1,
        "sampling_kind": str(getattr(cfg, "sampling_kind", "cartesian")),
        "mask_key": "mask" if str(getattr(cfg, "sampling_kind", "cartesian")) == "cartesian" else None,
        "trajectory_key": "trajectory" if str(getattr(cfg, "sampling_kind", "cartesian")) == "noncartesian" else None,
        "sensmaps_from_observed_data": (
            getattr(cfg, "output_sensmaps_key", "sens_maps") is not None
            and str(getattr(cfg, "target_type", "fullysampled_rec")) != "rss"
            and str(getattr(cfg, "sensmap_mode", "bart_from_observed")) == "bart_from_observed"
        ),
        # A list/interval spec varies per volume, so "baked in" means any of its
        # values moves the resolution; the per-file factor is not recorded here.
        "interpolation_baked_in": any(
            float(v) != 1.0 for v in _as_list(getattr(cfg, "target_interpolate_by_factor", 1.0))
        ),
        "interpolation_domain": "kspace",
        "scaling_baked_in": (
            bool(getattr(cfg, "scale_target_by_kspacenorm", False))
            or float(getattr(cfg, "target_scaling_factor", 1.0)) != 1.0
        ),
        "observation_scaling_mode": None
        if getattr(cfg, "observation_scaling_mode", "default") is None
        else str(getattr(cfg, "observation_scaling_mode", "default")),
        "coil_compression_virtual_coils": getattr(cfg, "coil_compression_virtual_coils", None),
    }
    meta.update(_trajectory_provenance_attrs(cfg))
    with open(Path(output_dir) / "_preprocess_recon_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_recon_observation(
    input_cfg: CacheableDictConfig,
    output_cache_subfolder: str,
    preprocess_cfg: CacheableDictConfig,
    dataset_cfg: CacheableDictConfig,
    local_cache_path: str,
    bart_path: Optional[str] = None,
) -> Dict[str, Any]:
    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_resolver = concrete_path_resolver(local_cache_path)
    if bart_path is not None:
        import_bart(bart_path)

    cfg = preprocess_cfg.cfg
    in_cfg = input_cfg.cfg
    out_base = str(cache_base_from_subfolder(local_cache_path, output_cache_subfolder, "preprocess_cache"))

    hash_payload = json.dumps(
        {
            "input": OmegaConf.to_container(in_cfg, resolve=False),
            "preprocess": OmegaConf.to_container(cfg, resolve=False),
            "dataset": OmegaConf.to_container(dataset_cfg.cfg, resolve=False),
        },
        sort_keys=True,
        default=str,
    ).encode()
    cfg_hash = hashlib.sha256(hash_payload).hexdigest()[:8]
    version = str(getattr(cfg, "version", "v1"))
    subdir = f"preprocess_recon_{version}_{getattr(cfg, 'sampling_kind', 'cartesian')}_{cfg_hash}"
    out_dir = os.path.join(out_base, subdir)
    _resolve_configured_trajectory_path(cfg, path_resolver)

    input_paths = _as_list(_resolve(getattr(in_cfg, "data_path_recon", None), path_resolver))
    sens_paths_raw = getattr(in_cfg, "data_path_sensmaps_recon", None)
    sens_paths = _as_list(_resolve(sens_paths_raw, path_resolver)) if sens_paths_raw is not None else [None] * len(input_paths)
    if len(sens_paths) == 0:
        sens_paths = [None] * len(input_paths)
    if len(sens_paths) == 1 and len(input_paths) > 1:
        sens_paths = sens_paths * len(input_paths)
    if len(input_paths) != len(sens_paths):
        raise ValueError("data_path_recon and data_path_sensmaps_recon must have matching lengths.")

    output_paths = []
    for idx, (input_path, sens_path) in enumerate(zip(input_paths, sens_paths)):
        output_path = os.path.join(out_dir, str(idx)) if len(input_paths) > 1 else out_dir
        _prepare_directory(str(input_path), output_path, None if sens_path is None else str(sens_path), cfg, dataset_cfg.cfg, device)
        output_paths.append(output_path)

    result: Dict[str, Any] = {"recon": output_paths if len(output_paths) > 1 else output_paths[0]}
    logging.info("run_recon_observation done. Output paths: %s", result)
    return result
