"""Generic H5 -> preprocessed H5 pipeline for datasets that already ship as H5
files (cc359, stanford_3d, etc.).
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.problem_trafos.dataset_trafo.volume_preprocess_utils import (
    interpolate_sensmaps,
    interpolate_volume,
    scale_by_kspace_norm,
)
from src.problem_trafos.utils.bart_utils import coil_compress_kspace_3d, compute_sens_maps_3d, import_bart
from src.utils.fftn3d import ifft3c


def _squeeze_bart_sensmaps(sens_maps_np: np.ndarray) -> np.ndarray:
    while sens_maps_np.ndim > 4:
        singleton_axes = [axis for axis, size in enumerate(sens_maps_np.shape) if size == 1]
        if not singleton_axes:
            break
        sens_maps_np = np.squeeze(sens_maps_np, axis=singleton_axes[0])
    return sens_maps_np


def _compress_kspace_np_zycx(kspace_np: np.ndarray, device: str, virtual_coils: Optional[int]) -> np.ndarray:
    if virtual_coils is None or int(virtual_coils) <= 0:
        return kspace_np
    kspace_t = torch.view_as_real(
        torch.from_numpy(kspace_np.astype(np.complex64)).movedim(1, 0)
    ).to(device)
    kspace_cc = coil_compress_kspace_3d(kspace_t, virtual_coils)
    return torch.view_as_complex(kspace_cc.movedim(0, 1).contiguous()).cpu().numpy()


def _compute_bart_sensmaps_from_kspace(kspace_np: np.ndarray, device: str) -> np.ndarray:
    """Estimate full-data sensitivity maps from k-space using BART ecalib."""
    kspace_t = torch.view_as_real(
        torch.from_numpy(kspace_np.astype(np.complex64)).movedim(1, 0)
    ).to(device)
    return _squeeze_bart_sensmaps(compute_sens_maps_3d(kspace_t))


def _center_crop_kspace_np_zycx(kspace_np: np.ndarray, scale_factor: float) -> np.ndarray:
    if scale_factor == 1.0:
        return kspace_np
    if scale_factor <= 0.0 or scale_factor > 1.0:
        raise ValueError(f"k-space crop scale_factor must be in (0, 1], got {scale_factor}.")
    z, _coils, y, x = kspace_np.shape
    nz = int(round(z * scale_factor))
    ny = int(round(y * scale_factor))
    nx = int(round(x * scale_factor))
    sz = (z - nz) // 2
    sy = (y - ny) // 2
    sx = (x - nx) // 2
    return kspace_np[sz:sz + nz, :, sy:sy + ny, sx:sx + nx].copy()


def _compute_mvue(kspace_np: np.ndarray, sens_maps_np: np.ndarray, device: str) -> torch.Tensor:
    """Compute MVUE from raw multicoil kspace + sensitivity maps."""
    # kspace: (Z, Coils, Y, X) -> movedim -> (Coils, Z, Y, X) -> view_as_real -> (..., 2)
    kspace_t = torch.view_as_real(
        torch.from_numpy(kspace_np.astype(np.complex64)).movedim(1, 0)
    ).to(device)  # (Coils, Z, Y, X, 2)

    # coil images in image space
    coil_imgs = ifft3c(kspace_t)  # (Coils, Z, Y, X, 2)
    coil_imgs_c = torch.view_as_complex(coil_imgs.contiguous())  # (Coils, Z, Y, X)

    if sens_maps_np.ndim != 4:
        raise ValueError(
            f"Expected 4D sensmaps, got shape {sens_maps_np.shape}."
        )

    # Detect coil axis robustly (supports both coil-last and coil-first files).
    import itertools
    n_coils = kspace_np.shape[1]
    ksp_spatial = kspace_np.shape[0], kspace_np.shape[2], kspace_np.shape[3]  # (Z, Y, X)
    candidate_coil_axes = [axis for axis, size in enumerate(sens_maps_np.shape) if size == n_coils]

    matched = False
    for coil_axis in candidate_coil_axes:
        sens_maps_cfirst = np.moveaxis(sens_maps_np, coil_axis, 0)  # (Coils, s0, s1, s2)
        sm_spatial = sens_maps_cfirst.shape[1:]
        for perm in itertools.permutations(range(3)):
            if tuple(sm_spatial[i] for i in perm) == ksp_spatial:
                sens_maps_np = np.transpose(
                    sens_maps_cfirst, (0, perm[0] + 1, perm[1] + 1, perm[2] + 1)
                )
                matched = True
                break
        if matched:
            break

    if not matched:
        raise ValueError(
            f"Cannot find a sensmap layout/permutation for sensmap shape {sens_maps_np.shape} "
            f"with kspace coils={n_coils} and spatial shape {ksp_spatial}. "
            "Check that the correct sensmaps are being used."
        )

    S = torch.from_numpy(sens_maps_np.astype(np.complex64)).to(device)

    # replace zero-norm voxels with small non-zero to avoid divide-by-zero
    if S.abs().sum() > 0:
        min_abs = S[S != 0].abs().min()
        S[S == 0] = min_abs + 0j

    S_norm = S.abs().square().sum(dim=0).sqrt()  # (Z, Y, X)
    mvue = (coil_imgs_c * S.conj()).sum(dim=0) / S_norm  # (Z, Y, X) complex

    return torch.view_as_real(mvue).float()  # (Z, Y, X, 2)


def _apply_readout_shifts(
    kspace_np: np.ndarray,
    readout_dim: int,
    shifts_enable: tuple,
    keep_spatial: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Apply readout dimension FFT shifts (ifftshift before, fftshift after)
    to kspace, matching the behavior of FastMRIVolumeDataset._fft1c.
    """
    from src.utils.fftn3d import fftshift, ifftshift
    
    kspace_t = torch.view_as_real(
        torch.from_numpy(kspace_np.astype(np.complex64))
    ).to(device)
    
    if shifts_enable[0]:
        kspace_t = ifftshift(kspace_t, dim=[readout_dim])
    
    kspace_t = torch.view_as_real(
        torch.fft.fftn(
            torch.view_as_complex(kspace_t), dim=[readout_dim], norm="ortho"
        )
    )
    
    if shifts_enable[1]:
        kspace_t = fftshift(kspace_t, dim=[readout_dim])
    
    if keep_spatial:
        if shifts_enable[0]:
            kspace_t = ifftshift(kspace_t, dim=[readout_dim])
        kspace_t = torch.view_as_real(
            torch.fft.ifftn(
                torch.view_as_complex(kspace_t), dim=[readout_dim], norm="ortho"
            )
        )
        if shifts_enable[1]:
            kspace_t = fftshift(kspace_t, dim=[readout_dim])

    return torch.view_as_complex(kspace_t.contiguous()).cpu().numpy()


def _apply_perspective_target(target_torch: torch.Tensor, axes: Optional[List[int]]) -> torch.Tensor:
    if axes is None:
        return target_torch
    axes = [int(v) for v in axes]
    if target_torch.ndim == 5 and target_torch.shape[-1] == 2:
        return target_torch.permute(*axes, 3, 4).contiguous()
    if target_torch.ndim == 4 and target_torch.shape[-1] == 2:
        return target_torch.permute(*axes, 3).contiguous()
    if target_torch.ndim == 4:
        return target_torch.permute(*axes, 3).contiguous()
    if target_torch.ndim == 3:
        return target_torch.permute(*axes).contiguous()
    return target_torch


def _apply_perspective_sensmaps(sens_maps_np: Optional[np.ndarray], axes: Optional[List[int]], spatial_shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if sens_maps_np is None or axes is None or sens_maps_np.ndim != 4:
        return sens_maps_np
    axes = [int(v) for v in axes]
    # Support common coil-last (Z,Y,X,C) and coil-first (C,Z,Y,X) layouts.
    if tuple(sens_maps_np.shape[:3]) == tuple(spatial_shape):
        return np.transpose(sens_maps_np, axes + [3]).copy()
    if tuple(sens_maps_np.shape[1:]) == tuple(spatial_shape):
        return np.transpose(sens_maps_np, [0] + [a + 1 for a in axes]).copy()
    logging.warning(
        "Cannot safely apply perspective axes %s to sensmaps shape %s with spatial shape %s; storing unchanged.",
        axes, sens_maps_np.shape, spatial_shape,
    )
    return sens_maps_np


def _load_h5_volume(
    fname: Path,
    recons_key: str,
    sensmaps_key: str,
    device: str,
    compute_mvue_from_kspace: bool = False,
    sens_maps_np_override: Optional[np.ndarray] = None,
    apply_fft1c_on_readout_dim: bool = False,
    apply_fft1c_on_readout_dim_shifts: tuple = (True, True),
    readout_dim: int = 0,
    readout_dim_keep_spatial: bool = False,
    dataset_is_3d: bool = True,
    sensmap_mode: str = "source",
    bart_path: Optional[str] = None,
    kspace_interpolate_by_factor: float = 1.0,
    kspace_interpolation_method: str = "fourier",
    coil_compression_virtual_coils: Optional[int] = None,
):
    """Return (target_torch, kspace_vol_norm, sens_maps_np, attrs)."""
    with h5py.File(fname, "r", locking=False) as hf:
        attrs = dict(hf.attrs)

        # kspace_vol_norm
        if "kspace_vol_norm" in attrs:
            kspace_vol_norm = float(attrs["kspace_vol_norm"])
        elif "kspace" in hf:
            kspace_np = hf["kspace"][:]
            kspace_vol_norm = float(np.linalg.norm(kspace_np))
        else:
            kspace_vol_norm = None

        if compute_mvue_from_kspace:
            if "kspace" not in hf:
                raise KeyError(
                    f"{fname}: compute_mvue_from_kspace=True but 'kspace' not in H5. "
                    "Set preprocess.compute_mvue_from_kspace=False or ensure kspace is present."
                )
            kspace_np = hf["kspace"][:]
            
            # Apply readout dimension FFT shifts if enabled (matching volume dataset behavior)
            if apply_fft1c_on_readout_dim and dataset_is_3d:
                kspace_np = _apply_readout_shifts(
                    kspace_np, readout_dim, apply_fft1c_on_readout_dim_shifts,
                    keep_spatial=readout_dim_keep_spatial, device=device
                )

            if kspace_interpolate_by_factor != 1.0:
                if str(kspace_interpolation_method) != "fourier":
                    raise ValueError(
                        "compute_mvue_from_kspace=True requires Fourier/k-space interpolation; "
                        f"got target_interpolation_method={kspace_interpolation_method!r}."
                    )
                kspace_np = _center_crop_kspace_np_zycx(kspace_np, float(kspace_interpolate_by_factor))

            if coil_compression_virtual_coils is not None and int(coil_compression_virtual_coils) > 0 and bart_path is not None:
                import_bart(str(bart_path))
            kspace_np = _compress_kspace_np_zycx(kspace_np, device, coil_compression_virtual_coils)
            kspace_vol_norm = float(np.linalg.norm(kspace_np))

            # sensitivity maps (must match the cropped k-space). In BART mode,
            # ignore any source maps and re-estimate after k-space cropping.
            if str(sensmap_mode) == "bart_from_kspace":
                sens_maps_np = None
            elif sensmaps_key and sensmaps_key in hf:
                sens_maps_np = hf[sensmaps_key][:]
            else:
                sens_maps_np = None

        else:
            # target
            target_np = hf[recons_key][:]
            target_torch = torch.from_numpy(target_np).to(device)
            if not torch.is_floating_point(target_torch):
                # complex -> real/imag pair
                target_torch = torch.view_as_real(target_torch.to(torch.complex64))

            if kspace_vol_norm is None:
                _norm = float(torch.linalg.norm(target_torch.to(torch.float32)))
                kspace_vol_norm = _norm if _norm > 0.0 else None
                if kspace_vol_norm is None:
                    logging.warning(f"{fname}: target has zero norm; cannot derive kspace_vol_norm.")

            # sensitivity maps
            sens_maps_np = None
            if sensmaps_key and sensmaps_key in hf:
                sens_maps_np = hf[sensmaps_key][:]

    if compute_mvue_from_kspace:
        # prefer externally supplied sensmaps only for source-sensmap mode.
        # In BART mode, sensmaps must be re-estimated after any k-space crop.
        if sens_maps_np_override is not None and str(sensmap_mode) != "bart_from_kspace":
            sens_maps_np = sens_maps_np_override
        if sens_maps_np is None:
            if str(sensmap_mode) == "bart_from_kspace":
                if bart_path is not None:
                    import_bart(str(bart_path))
                sens_maps_np = _compute_bart_sensmaps_from_kspace(kspace_np, device)
            else:
                raise ValueError(
                    f"{fname}: compute_mvue_from_kspace=True but no sensitivity maps found "
                    f"under key '{sensmaps_key}'. Provide external sensmaps via sensmap_dir "
                    "or set preprocess.sensmap_mode=bart_from_kspace."
                )
        target_torch = _compute_mvue(kspace_np, sens_maps_np, device)

    return target_torch, kspace_vol_norm, sens_maps_np, attrs


def _write_preprocessed_h5(
    out_path: Path,
    target_torch: torch.Tensor,
    sens_maps_np: Optional[np.ndarray],
    attrs: dict,
    output_recons_key: str,
    sensmaps_key: str,
):
    """Write preprocessed data to *out_path* using an atomic tmp->rename pattern."""
    os.makedirs(out_path.parent, exist_ok=True)
    tmp_path = out_path.with_name(out_path.stem + f".{os.getpid()}.tmp.h5")
    try:
        with h5py.File(tmp_path, "w") as hf:
            target_np = target_torch.cpu().numpy()
            if target_np.ndim >= 5 and target_np.shape[-2] == 1 and target_np.shape[-1] == 2:
                if np.max(np.abs(target_np[..., 1])) < 1e-8:
                    target_np = target_np[..., 0]
                else:
                    target_np = (target_np[..., 0] + 1j * target_np[..., 1]).astype(np.complex64)
            elif target_np.ndim == 4 and target_np.shape[-1] == 2:
                # (Z, Y, X, 2) real/imag -> complex64
                target_np = (target_np[..., 0] + 1j * target_np[..., 1]).astype(np.complex64)
            hf.create_dataset(output_recons_key, data=target_np)

            if sens_maps_np is not None:
                hf.create_dataset(sensmaps_key, data=sens_maps_np)

            # carry over original attrs; mark scaling already applied
            for k, v in attrs.items():
                try:
                    hf.attrs[k] = v
                except Exception:
                    pass
            # signal to downstream transform that normalisation is already baked in
            hf.attrs["kspace_vol_norm"] = 1.0
    except Exception:
        # Clean up the incomplete tmp file so a retry can start fresh.
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, out_path)


def _load_external_sensmaps(sensmap_dir: str, fname: Path) -> Optional[np.ndarray]:
    """Load sensmaps for ``fname`` from a separate ``sensmap_dir``."""
    candidates = [
        Path(sensmap_dir) / (fname.stem + "_sensmap" + fname.suffix),
        Path(sensmap_dir) / fname.name,
    ]
    for cand in candidates:
        if cand.exists():
            with h5py.File(cand, "r", locking=False) as hf:
                preferred_keys = ("sensitivity_maps", "sens_maps")
                key = next((k for k in preferred_keys if k in hf), None)

                if key is not None:
                    obj = hf[key]
                    if not (isinstance(obj, h5py.Dataset) and getattr(obj, "ndim", 0) >= 3):
                        logging.warning(
                            f"{cand}: preferred key '{key}' is not a volumetric dataset; "
                            f"shape={getattr(obj, 'shape', None)}. Falling back to key scan."
                        )
                        key = None

                # Fallback: choose the first dataset that looks like volumetric sensmaps
                # (typically 4D: Z,Y,X,Coils or Coils,Z,Y,X), not scalar metadata.
                if key is None:
                    for k in hf.keys():
                        obj = hf[k]
                        if isinstance(obj, h5py.Dataset) and getattr(obj, "ndim", 0) >= 3:
                            key = k
                            break

                if key is None:
                    logging.warning(
                        f"{cand}: no suitable sensmap dataset found. Keys={list(hf.keys())}"
                    )
                    continue

                # Use [()] for robust full read (works for both scalar and non-scalar datasets).
                return hf[key][()]
    logging.warning(f"No external sensmap found for {fname.name} in {sensmap_dir}")
    return None


# Module-level picklable helpers for multiprocessing pool

def _factor_list_types():
    try:
        from omegaconf import ListConfig
        return (list, tuple, ListConfig)
    except Exception:
        return (list, tuple)


def _factor_space_transform(space: str):
    """Return ``(to_coord, to_factor)`` for the coordinate the resolution
    distribution is defined on.
    """
    space = str(space).lower()
    if space == "factor":
        return (lambda x: float(x)), (lambda x: float(x))
    if space == "voxel":
        def _inv(x):
            x = float(x)
            if x <= 0.0:
                raise ValueError(
                    "target_interpolate_factor_space='voxel' needs strictly "
                    f"positive interpolation factors, got {x}"
                )
            return 1.0 / x
        return _inv, _inv
    raise ValueError(
        "target_interpolate_factor_space must be 'factor' or 'voxel', "
        f"got {space!r}"
    )


def _factor_choice_grid(
    target_interpolate_by_factor,
    target_interpolate_factor_is_interval: bool,
    target_interpolate_factor_step,
    target_interpolate_factor_space: str = "factor",
) -> Optional[List[float]]:
    """Discrete candidate interpolation factors implied by the spec."""
    value = target_interpolate_by_factor
    if not isinstance(value, _factor_list_types()):
        return [float(value)]

    factors = [float(v) for v in list(value)]
    if len(factors) == 0:
        return [1.0]
    if not target_interpolate_factor_is_interval:
        return factors
    if target_interpolate_factor_step is None:
        return None

    step = float(target_interpolate_factor_step)
    if step <= 0:
        raise ValueError("target_interpolate_factor_step must be > 0")
    to_coord, to_factor = _factor_space_transform(target_interpolate_factor_space)
    low, high = sorted((to_coord(min(factors)), to_coord(max(factors))))
    n_steps = int(np.floor((high - low) / step))
    coords = [low + i * step for i in range(n_steps + 1)]
    if not np.isclose(coords[-1], high):
        coords.append(high)
    return sorted(to_factor(c) for c in coords)


def _piecewise_linear_quantiles(weights, t_lo: float, t_hi: float, q) -> np.ndarray:
    """Inverse CDF of a piecewise-linear density with equidistant knots."""
    w = np.asarray([float(x) for x in weights], dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("target_interpolate_factor_weights must be non-negative")
    q = np.asarray(q, dtype=np.float64)
    if w.size < 2:
        # A single knot carries no shape information - fall back to uniform.
        return t_lo + (t_hi - t_lo) * q
    h = (t_hi - t_lo) / (w.size - 1)
    area = 0.5 * (w[:-1] + w[1:]) * h
    total = float(area.sum())
    if total <= 0:
        raise ValueError("target_interpolate_factor_weights must not sum to zero")
    cdf = np.concatenate([[0.0], np.cumsum(area) / total])
    idx = np.clip(np.searchsorted(cdf, q, side="right") - 1, 0, w.size - 2)
    w0, w1 = w[idx], w[idx + 1]
    # Solve h*(w0*s + (w1-w0)*s^2/2) = (q - cdf[idx])*total for s in [0, 1].
    c = (q - cdf[idx]) * total / h
    a = 0.5 * (w1 - w0)
    flat = np.abs(a) < 1e-12
    lin = np.divide(c, np.where(w0 > 0, w0, 1.0))
    quad = (-w0 + np.sqrt(np.maximum(w0 * w0 + 4.0 * a * c, 0.0))) / (2.0 * np.where(flat, 1.0, a))
    s = np.clip(np.where(flat, lin, quad), 0.0, 1.0)
    return t_lo + (idx + s) * h


def _continuous_factors_from_quantiles(
    factors: List[float],
    target_interpolate_factor_space: str,
    weights,
    q,
) -> List[float]:
    """Map probabilities ``q`` to factors drawn from a continuous interval."""
    to_coord, to_factor = _factor_space_transform(target_interpolate_factor_space)
    t_lo, t_hi = sorted((to_coord(min(factors)), to_coord(max(factors))))
    q = np.asarray(q, dtype=np.float64)
    if weights is None:
        coords = t_lo + (t_hi - t_lo) * q
    else:
        w = [float(x) for x in list(weights)]
        # Ascending factor is descending voxel size, so the coarse-first weight
        # order is already ascending in factor space but reversed in voxel space.
        if str(target_interpolate_factor_space).lower() == "voxel":
            w = w[::-1]
        coords = _piecewise_linear_quantiles(w, t_lo, t_hi, q)
    return [float(to_factor(c)) for c in np.atleast_1d(coords)]


def _normalize_factor_weights(weights, choices: List[float]) -> Optional[np.ndarray]:
    """Turn user-supplied weights into a probability vector over ``choices``."""
    if weights is None:
        return None
    w = np.asarray([float(x) for x in list(weights)], dtype=np.float64)
    if w.size != len(choices):
        raise ValueError(
            f"target_interpolate_factor_weights has {w.size} entries but the "
            f"interpolation spec yields {len(choices)} factors ({choices}). "
            "Provide exactly one weight per candidate factor."
        )
    if np.any(w < 0):
        raise ValueError("target_interpolate_factor_weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("target_interpolate_factor_weights must not sum to zero")
    return w / total


def _resolve_factors_for_file(
    fname_name: str,
    base_seed: int,
    count: int,
    target_interpolate_by_factor,
    target_interpolate_factor_is_interval: bool,
    target_interpolate_factor_step,
    target_interpolate_factor_weights=None,
    target_interpolate_factor_with_replacement=None,
    target_interpolate_factor_space: str = "factor",
) -> List[float]:
    """Derive per-file interpolation factors deterministically from the filename."""
    import zlib
    file_seed = (base_seed ^ (zlib.crc32(fname_name.encode("utf-8")) & 0xFFFFFFFF)) & 0xFFFFFFFF
    _rng = np.random.default_rng(file_seed)

    choices = _factor_choice_grid(
        target_interpolate_by_factor,
        target_interpolate_factor_is_interval,
        target_interpolate_factor_step,
        target_interpolate_factor_space,
    )

    if choices is None:
        factors = [float(v) for v in list(target_interpolate_by_factor)]
        low, high = min(factors), max(factors)
        if (str(target_interpolate_factor_space).lower() == "factor"
                and target_interpolate_factor_weights is None):
            return [float(v) for v in _rng.uniform(low, high, size=count)]
        return _continuous_factors_from_quantiles(
            factors,
            target_interpolate_factor_space,
            target_interpolate_factor_weights,
            _rng.random(count),
        )

    if len(choices) == 1:
        return [float(choices[0])] * count

    probs = _normalize_factor_weights(target_interpolate_factor_weights, choices)
    if target_interpolate_factor_with_replacement is None:
        replace = probs is not None or count > len(choices)
    else:
        replace = bool(target_interpolate_factor_with_replacement)

    choices_arr = np.array(choices, dtype=np.float32)
    picks = _rng.choice(choices_arr, size=count, replace=replace, p=probs)
    return [float(v) for v in picks]


def _assign_factors_exact(
    fname_names: List[str],
    base_seed: int,
    count: int,
    choices: List[float],
    probs: Optional[np.ndarray],
) -> Dict[str, List[float]]:
    """Allocate factors so the realized dataset composition matches ``probs``."""
    names = sorted(fname_names)
    total = len(names) * count
    if probs is None:
        probs = np.full(len(choices), 1.0 / len(choices), dtype=np.float64)

    exact = probs * total
    counts = np.floor(exact).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(exact - counts), kind="stable")
        for i in range(remainder):
            counts[order[i]] += 1

    slots = np.repeat(np.array(choices, dtype=np.float32), counts)
    rng = np.random.default_rng(base_seed)
    rng.shuffle(slots)
    return {
        name: [float(v) for v in slots[i * count:(i + 1) * count]]
        for i, name in enumerate(names)
    }


def _assign_factors_exact_continuous(
    fname_names: List[str],
    base_seed: int,
    count: int,
    factors: List[float],
    target_interpolate_factor_space: str,
    weights,
) -> Dict[str, List[float]]:
    """Quota-based allocation for a *continuous* spec."""
    names = sorted(fname_names)
    total = len(names) * count
    q = (np.arange(total, dtype=np.float64) + 0.5) / total
    slots = np.asarray(
        _continuous_factors_from_quantiles(
            factors, target_interpolate_factor_space, weights, q
        ),
        dtype=np.float64,
    )
    rng = np.random.default_rng(base_seed)
    rng.shuffle(slots)
    return {
        name: [float(v) for v in slots[i * count:(i + 1) * count]]
        for i, name in enumerate(names)
    }


def _factor_histogram(factors_by_file: Dict[str, List[float]]) -> Dict[str, int]:
    """Realized count per (rounded) interpolation factor, for logging/metadata."""
    hist: Dict[str, int] = {}
    for factors in factors_by_file.values():
        for f in factors:
            key = repr(round(float(f), 6))
            hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: float(kv[0])))


def _perspective_to_spec(perspective, output_dir: str) -> dict:
    if isinstance(perspective, str):
        name = perspective
        axes = {
            "ax": [0, 1, 2],
            "axial": [0, 1, 2],
            "cor": [1, 0, 2],
            "coronal": [1, 0, 2],
            "sag": [2, 0, 1],
            "sagittal": [2, 0, 1],
        }[perspective]
    else:
        name = str(perspective.name)
        axes = list(perspective.axes)
    return {
        "name": name,
        "axes": axes,
        "output_dir": os.path.join(output_dir, name),
    }


def _preprocess_single_file(args: dict):
    """Top-level picklable worker: preprocess one H5 file."""
    fname = Path(args["fname"])
    output_dir = args["output_dir"]
    sensmap_dir = args.get("sensmap_dir")
    device = args["device"]
    exists_ok = args["exists_ok"]

    compute_mvue = args["compute_mvue"]
    recons_key = args["recons_key"]
    sensmaps_key = args["sensmaps_key"]
    output_recons_key = args["output_recons_key"]
    output_sensmaps_key = args["output_sensmaps_key"]
    skip_sensmaps = args["skip_sensmaps"]
    scale_by_kspacenorm = args["scale_by_kspacenorm"]
    target_scaling_factor = args["target_scaling_factor"]
    interpolation_method = args["interpolation_method"]
    apply_fft1c_on_readout_dim = args["apply_fft1c_on_readout_dim"]
    apply_fft1c_on_readout_dim_shifts = args["apply_fft1c_on_readout_dim_shifts"]
    readout_dim = args["readout_dim"]
    readout_dim_keep_spatial = args["readout_dim_keep_spatial"]
    dataset_is_3d = args["dataset_is_3d"]
    sensmap_mode = args["sensmap_mode"]
    bart_path = args.get("bart_path")
    coil_compression_virtual_coils = args.get("coil_compression_virtual_coils")
    target_interpolate_samples_per_volume = args["target_interpolate_samples_per_volume"]
    perspective_specs = args.get("perspective_specs")
    if perspective_specs is None:
        perspective_specs = [{
            "name": args.get("perspective_name"),
            "axes": args.get("perspective_axes"),
            "output_dir": output_dir,
        }]

    precomputed_factors = args.get("target_interpolate_factors")
    if precomputed_factors is not None:
        interpolate_factors = [float(v) for v in precomputed_factors]
    else:
        interpolate_factors = _resolve_factors_for_file(
            fname_name=fname.name,
            base_seed=args["target_interpolate_factor_seed"],
            count=target_interpolate_samples_per_volume,
            target_interpolate_by_factor=args["target_interpolate_by_factor"],
            target_interpolate_factor_is_interval=args["target_interpolate_factor_is_interval"],
            target_interpolate_factor_step=args["target_interpolate_factor_step"],
            target_interpolate_factor_weights=args.get("target_interpolate_factor_weights"),
            target_interpolate_factor_with_replacement=args.get(
                "target_interpolate_factor_with_replacement"
            ),
            target_interpolate_factor_space=args.get(
                "target_interpolate_factor_space", "factor"
            ),
        )

    # Exist check - skip expensive I/O if all outputs already valid
    if exists_ok:
        all_exist = True
        for sample_idx in range(len(interpolate_factors)):
            out_name = (
                fname.name
                if target_interpolate_samples_per_volume == 1
                else f"{fname.stem}__interp{sample_idx:02d}{fname.suffix}"
            )
            for perspective_spec in perspective_specs:
                out_path = Path(perspective_spec["output_dir"]) / out_name
                if not out_path.exists():
                    all_exist = False
                    break
                try:
                    with h5py.File(out_path, "r", locking=False):
                        pass
                except OSError:
                    all_exist = False
                    break
            if not all_exist:
                break
        if all_exist:
            for sample_idx in range(len(interpolate_factors)):
                out_name = (
                    fname.name
                    if target_interpolate_samples_per_volume == 1
                    else f"{fname.stem}__interp{sample_idx:02d}{fname.suffix}"
                )
                for perspective_spec in perspective_specs:
                    logging.info(f"Skipping existing (valid) {Path(perspective_spec['output_dir']) / out_name}")
            return None  # data_is_complex undetermined from skipped file

    # Load
    target_base = None
    kspace_vol_norm = None
    sens_maps_base = None
    attrs = {}
    data_is_complex = True

    if not compute_mvue:
        target_base, kspace_vol_norm, sens_maps_base, attrs = _load_h5_volume(
            fname, recons_key, sensmaps_key, device,
            compute_mvue_from_kspace=False,
            apply_fft1c_on_readout_dim=apply_fft1c_on_readout_dim,
            apply_fft1c_on_readout_dim_shifts=apply_fft1c_on_readout_dim_shifts,
            readout_dim=readout_dim,
            readout_dim_keep_spatial=readout_dim_keep_spatial,
            dataset_is_3d=dataset_is_3d,
            sensmap_mode=sensmap_mode,
            bart_path=bart_path,
            coil_compression_virtual_coils=coil_compression_virtual_coils,
        )
        data_is_complex = target_base.ndim == 4 and target_base.shape[-1] == 2

        if sensmap_dir is not None:
            ext_smaps = _load_external_sensmaps(sensmap_dir, fname)
            if ext_smaps is not None:
                sens_maps_base = ext_smaps

    # Per-sample loop
    for sample_idx, interpolate_by_factor in enumerate(interpolate_factors):
        out_name = (
            fname.name
            if target_interpolate_samples_per_volume == 1
            else f"{fname.stem}__interp{sample_idx:02d}{fname.suffix}"
        )
        pending_specs = []
        for perspective_spec in perspective_specs:
            out_path = Path(perspective_spec["output_dir"]) / out_name

            for stale in out_path.parent.glob(out_path.stem + ".*.tmp.h5"):
                logging.warning(f"Removing incomplete tmp file from a previous interrupted run: {stale}")
                stale.unlink(missing_ok=True)

            if out_path.exists() and exists_ok:
                try:
                    with h5py.File(out_path, "r", locking=False):
                        pass
                    logging.info(f"Skipping existing (valid) {out_path}")
                    continue
                except OSError as _e:
                    logging.warning(
                        f"Existing output file appears corrupt ({_e}); "
                        f"removing and reprocessing: {out_path}"
                    )
                    out_path.unlink(missing_ok=True)
            pending_specs.append((perspective_spec, out_path))

        if not pending_specs:
            continue

        if compute_mvue:
            ext_smaps_preloaded = None
            if sensmap_dir is not None and str(sensmap_mode) != "bart_from_kspace":
                ext_smaps_preloaded = _load_external_sensmaps(sensmap_dir, fname)
            target_torch, kspace_vol_norm_current, sens_maps_np, attrs_current = _load_h5_volume(
                fname, recons_key, sensmaps_key, device,
                compute_mvue_from_kspace=True,
                sens_maps_np_override=ext_smaps_preloaded,
                apply_fft1c_on_readout_dim=apply_fft1c_on_readout_dim,
                apply_fft1c_on_readout_dim_shifts=apply_fft1c_on_readout_dim_shifts,
                readout_dim=readout_dim,
                readout_dim_keep_spatial=readout_dim_keep_spatial,
                dataset_is_3d=dataset_is_3d,
                sensmap_mode=sensmap_mode,
                bart_path=bart_path,
                kspace_interpolate_by_factor=float(interpolate_by_factor),
                kspace_interpolation_method=interpolation_method,
                coil_compression_virtual_coils=coil_compression_virtual_coils,
            )
            attrs = attrs_current
            kspace_vol_norm = kspace_vol_norm_current
            data_is_complex = target_torch.ndim == 4 and target_torch.shape[-1] == 2
        else:
            target_torch = target_base.clone()
            sens_maps_np = None if sens_maps_base is None else sens_maps_base.copy()

        # Scaling
        if scale_by_kspacenorm:
            if kspace_vol_norm is None:
                raise ValueError(
                    f"{fname}: preprocess.scale_target_by_kspacenorm=True but no k-space norm "
                    "could be determined (no 'kspace' dataset, no 'kspace_vol_norm' attribute, "
                    "and the target norm is zero). Fix the source data or set "
                    "preprocess.scale_target_by_kspacenorm=False."
                )
            target_torch = scale_by_kspace_norm(target_torch, kspace_vol_norm)
        if target_scaling_factor != 1.0:
            target_torch = target_torch * target_scaling_factor

        # Resolution interpolation
        if not compute_mvue and interpolate_by_factor != 1.0:
            is_real_volume = target_torch.ndim == 3
            if is_real_volume:
                target_torch = torch.stack(
                    [target_torch, torch.zeros_like(target_torch)], dim=-1
                )  # (Z, Y, X, 2)
            target_torch = interpolate_volume(target_torch, interpolate_by_factor, interpolation_method)
            if is_real_volume:
                target_torch = target_torch[..., 0]

        # Sensitivity map interpolation
        if not compute_mvue and interpolate_by_factor != 1.0 and sens_maps_np is not None:
            sens_maps_np = interpolate_sensmaps(
                sens_maps_np, interpolate_by_factor, interpolation_method, device=device
            )

        spatial_shape_before_perspective = tuple(target_torch.shape[:3]) if target_torch.ndim >= 3 else tuple()
        for perspective_spec, out_path in pending_specs:
            perspective_name = perspective_spec.get("name")
            perspective_axes = perspective_spec.get("axes")
            target_to_store = _apply_perspective_target(target_torch, perspective_axes)
            sens_maps_to_store = _apply_perspective_sensmaps(
                sens_maps_np, perspective_axes, spatial_shape_before_perspective
            )
            attrs_to_store = dict(attrs)
            if perspective_name is not None:
                attrs_to_store["perspective"] = str(perspective_name)

            if skip_sensmaps:
                sens_maps_to_store = None
            _write_preprocessed_h5(
                out_path, target_to_store, sens_maps_to_store, attrs_to_store,
                output_recons_key, output_sensmaps_key,
            )

    return data_is_complex



def preprocess_h5_directory(
    input_dir: str,
    output_dir: str,
    preprocess_cfg,
    device: str = "cpu",
    volume_filter: str = ".*\\.h5",
    volume_limit: Optional[int] = None,
    recons_key: str = "reconstruction_mvue",
    sensmaps_key: str = "sensitivity_maps",
    sensmap_dir: Optional[str] = None,
    exists_ok: bool = True,
    num_workers: int = 1,
    perspective_name: Optional[str] = None,
    perspective_axes: Optional[List[int]] = None,
    perspective_specs: Optional[List[dict]] = None,
    bart_path: Optional[str] = None,
):
    """Iterate over all H5 files in ``input_dir``, apply preprocessing, write to
    ``output_dir`` with the same filenames.
    """
    import re
    import json
    scale_by_kspacenorm: bool = getattr(preprocess_cfg, "scale_target_by_kspacenorm", False)
    target_scaling_factor: float = float(getattr(preprocess_cfg, "target_scaling_factor", 1.0))
    target_interpolate_by_factor = getattr(preprocess_cfg, "target_interpolate_by_factor", 1.0)
    target_interpolate_factor_is_interval: bool = bool(
        getattr(preprocess_cfg, "target_interpolate_factor_is_interval", False)
    )
    target_interpolate_factor_step = getattr(preprocess_cfg, "target_interpolate_factor_step", None)
    # Coordinate the resolution distribution lives on: the crop factor itself
    # ("factor", historical) or the voxel size ("voxel").
    target_interpolate_factor_space: str = str(
        getattr(preprocess_cfg, "target_interpolate_factor_space", "factor")
    ).lower()
    _factor_space_transform(target_interpolate_factor_space)  # validate early
    target_interpolate_factor_seed: int = int(getattr(preprocess_cfg, "target_interpolate_factor_seed", 1234))
    # Non-uniform "dataset balance": one weight per candidate factor.
    target_interpolate_factor_weights = getattr(preprocess_cfg, "target_interpolate_factor_weights", None)
    if target_interpolate_factor_weights is not None:
        target_interpolate_factor_weights = [float(w) for w in list(target_interpolate_factor_weights)]
    target_interpolate_factor_with_replacement = getattr(
        preprocess_cfg, "target_interpolate_factor_with_replacement", None
    )
    if target_interpolate_factor_with_replacement is not None:
        target_interpolate_factor_with_replacement = bool(target_interpolate_factor_with_replacement)
    target_interpolate_factor_assignment: str = str(
        getattr(preprocess_cfg, "target_interpolate_factor_assignment", "iid")
    ).lower()
    if target_interpolate_factor_assignment not in ("iid", "exact"):
        raise ValueError(
            "target_interpolate_factor_assignment must be 'iid' or 'exact', "
            f"got {target_interpolate_factor_assignment!r}"
        )
    target_interpolate_samples_per_volume: int = int(
        getattr(preprocess_cfg, "target_interpolate_samples_per_volume", 1)
    )
    if target_interpolate_samples_per_volume < 1:
        raise ValueError("target_interpolate_samples_per_volume must be >= 1")
    interpolation_method: str = getattr(preprocess_cfg, "target_interpolation_method", "fourier")
    # Key to write in the output H5.  Falls back to the source read key when not set.
    output_recons_key: str = str(getattr(preprocess_cfg, "output_recons_key", recons_key))
    output_sensmaps_key: str = sensmaps_key
    compute_mvue: bool = bool(getattr(preprocess_cfg, "compute_mvue_from_kspace", False))
    skip_sensmaps: bool = bool(getattr(preprocess_cfg, "skip_sensmaps", False))
    sensmap_mode: str = str(getattr(preprocess_cfg, "sensmap_mode", "source"))
    coil_compression_virtual_coils = getattr(preprocess_cfg, "coil_compression_virtual_coils", None)
    
    # Dataset-specific readout dimension shifts (from preprocess config; applied to kspace before MVUE)
    apply_fft1c_on_readout_dim: bool = bool(getattr(preprocess_cfg, "apply_fft1c_on_readout_dim", False))
    apply_fft1c_on_readout_dim_shifts: tuple = tuple(getattr(preprocess_cfg, "apply_fft1c_on_readout_dim_shifts", (True, True)))
    readout_dim: int = int(getattr(preprocess_cfg, "readout_dim", 0))
    readout_dim_keep_spatial: bool = bool(getattr(preprocess_cfg, "readout_dim_keep_spatial", False))
    dataset_is_3d: bool = bool(getattr(preprocess_cfg, "dataset_is_3d", True))

    try:
        from omegaconf import ListConfig
        _list_types = (list, tuple, ListConfig)
    except Exception:
        _list_types = (list, tuple)

    def _interpolation_is_baked_in() -> bool:
        value = target_interpolate_by_factor
        if isinstance(value, _list_types):
            factors = [float(v) for v in list(value)]
            if target_interpolate_factor_is_interval:
                return not (len(factors) == 2 and np.isclose(factors[0], 1.0) and np.isclose(factors[1], 1.0))
            return any(not np.isclose(v, 1.0) for v in factors)
        return not np.isclose(float(value), 1.0)

    p = re.compile(volume_filter)
    fnames = sorted(
        f for f in Path(input_dir).iterdir()
        if p.match(f.name)
    )
    if not fnames:
        available = sorted(p.name for p in Path(input_dir).iterdir())
        raise FileNotFoundError(
            f"Train preprocessing found no input H5 files in {input_dir} matching volume_filter={volume_filter!r}. "
            f"Available entries: {available[:10]}{'...' if len(available) > 10 else ''}. "
            "If this is a converted dataset, inspect conversion_meta.json and the fold manifest; "
            "the selected split may be empty."
        )

    if volume_limit is not None and volume_limit > 0:
        if len(fnames) > volume_limit:
            logging.info(f"volume_limit={volume_limit}: truncating {len(fnames)} -> {volume_limit} files in {input_dir}")
        fnames = fnames[:volume_limit]

    if perspective_specs is None:
        perspective_specs = [{
            "name": perspective_name,
            "axes": perspective_axes,
            "output_dir": output_dir,
        }]
    os.makedirs(output_dir, exist_ok=True)

    # Build flat picklable argument dicts - one per file - for worker dispatch.
    _base_args: dict = dict(
        output_dir=output_dir,
        sensmap_dir=sensmap_dir,
        device=device,
        exists_ok=exists_ok,
        compute_mvue=compute_mvue,
        recons_key=recons_key,
        sensmaps_key=sensmaps_key,
        output_recons_key=output_recons_key,
        output_sensmaps_key=output_sensmaps_key,
        skip_sensmaps=skip_sensmaps,
        scale_by_kspacenorm=scale_by_kspacenorm,
        target_scaling_factor=target_scaling_factor,
        interpolation_method=interpolation_method,
        apply_fft1c_on_readout_dim=apply_fft1c_on_readout_dim,
        apply_fft1c_on_readout_dim_shifts=tuple(apply_fft1c_on_readout_dim_shifts),
        readout_dim=readout_dim,
        readout_dim_keep_spatial=readout_dim_keep_spatial,
        dataset_is_3d=dataset_is_3d,
        sensmap_mode=sensmap_mode,
        bart_path=bart_path,
        coil_compression_virtual_coils=coil_compression_virtual_coils,
        target_interpolate_samples_per_volume=target_interpolate_samples_per_volume,
        target_interpolate_by_factor=target_interpolate_by_factor,
        target_interpolate_factor_is_interval=target_interpolate_factor_is_interval,
        target_interpolate_factor_step=target_interpolate_factor_step,
        target_interpolate_factor_seed=target_interpolate_factor_seed,
        target_interpolate_factor_weights=target_interpolate_factor_weights,
        target_interpolate_factor_with_replacement=target_interpolate_factor_with_replacement,
        target_interpolate_factor_space=target_interpolate_factor_space,
        perspective_specs=perspective_specs,
    )

    _factor_choices = _factor_choice_grid(
        target_interpolate_by_factor,
        target_interpolate_factor_is_interval,
        target_interpolate_factor_step,
        target_interpolate_factor_space,
    )
    if target_interpolate_factor_assignment == "exact":
        if _factor_choices is None:
            factors_by_file = _assign_factors_exact_continuous(
                fname_names=[f.name for f in fnames],
                base_seed=target_interpolate_factor_seed,
                count=target_interpolate_samples_per_volume,
                factors=[float(v) for v in list(target_interpolate_by_factor)],
                target_interpolate_factor_space=target_interpolate_factor_space,
                weights=target_interpolate_factor_weights,
            )
        else:
            factors_by_file = _assign_factors_exact(
                fname_names=[f.name for f in fnames],
                base_seed=target_interpolate_factor_seed,
                count=target_interpolate_samples_per_volume,
                choices=_factor_choices,
                probs=_normalize_factor_weights(target_interpolate_factor_weights, _factor_choices),
            )
    else:
        factors_by_file = {
            fname.name: _resolve_factors_for_file(
                fname_name=fname.name,
                base_seed=target_interpolate_factor_seed,
                count=target_interpolate_samples_per_volume,
                target_interpolate_by_factor=target_interpolate_by_factor,
                target_interpolate_factor_is_interval=target_interpolate_factor_is_interval,
                target_interpolate_factor_step=target_interpolate_factor_step,
                target_interpolate_factor_weights=target_interpolate_factor_weights,
                target_interpolate_factor_with_replacement=target_interpolate_factor_with_replacement,
                target_interpolate_factor_space=target_interpolate_factor_space,
            )
            for fname in fnames
        }
    factor_histogram = _factor_histogram(factors_by_file)
    _n_slots = sum(factor_histogram.values()) or 1
    logging.info(
        "preprocess_h5_directory: interpolation factor balance "
        f"(assignment={target_interpolate_factor_assignment}, "
        f"space={target_interpolate_factor_space}, "
        f"weights={target_interpolate_factor_weights}): "
        + ", ".join(f"{k}: {v} ({100.0 * v / _n_slots:.1f}%)" for k, v in factor_histogram.items())
    )

    file_args_list = [
        {
            **_base_args,
            "fname": str(fname),
            "target_interpolate_factors": factors_by_file[fname.name],
        }
        for fname in fnames
    ]

    _data_is_complex: Optional[bool] = None

    if num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logging.info(
            f"preprocess_h5_directory: processing {len(fnames)} files "
            f"with {num_workers} workers -> {output_dir}"
        )
        logging.info("preprocess_h5_directory: starting worker pool")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {
                pool.submit(_preprocess_single_file, args): args["fname"]
                for args in file_args_list
            }
            for fut in tqdm(
                as_completed(futures), total=len(fnames), desc=f"Preprocessing {input_dir}"
            ):
                result = fut.result()  # re-raises any worker exception
                if result is not None and _data_is_complex is None:
                    _data_is_complex = result
        logging.info("preprocess_h5_directory: worker pool finished")
    else:
        for args in tqdm(file_args_list, desc=f"Preprocessing {input_dir}"):
            result = _preprocess_single_file(args)
            if result is not None and _data_is_complex is None:
                _data_is_complex = result

    prior_mode = "complex_2ch" if (compute_mvue or _data_is_complex) else "magnitude_1ch"
    meta = {
        "recons_key": output_recons_key,
        "sensmaps_key": output_sensmaps_key if not skip_sensmaps else None,
        "prior_mode": prior_mode,
        "readout_fft_shifts_baked_in": apply_fft1c_on_readout_dim,
        "scaling_baked_in": scale_by_kspacenorm or target_scaling_factor != 1.0,
        "interpolation_baked_in": _interpolation_is_baked_in(),
        "interpolation_samples_per_volume": target_interpolate_samples_per_volume,
        "effective_dataset_multiplier": target_interpolate_samples_per_volume,
        "interpolation_factor_weights": target_interpolate_factor_weights,
        "interpolation_factor_assignment": target_interpolate_factor_assignment,
        "interpolation_factor_histogram": factor_histogram,
        "skip_sensmaps": skip_sensmaps,
        "interpolation_domain": "kspace" if compute_mvue else "image",
        "sensmap_mode": sensmap_mode,
        "coil_compression_virtual_coils": coil_compression_virtual_coils,
        "perspective": None,
        "perspective_axes": None,
    }
    for perspective_spec in perspective_specs:
        meta_for_perspective = dict(meta)
        meta_for_perspective["perspective"] = perspective_spec.get("name")
        meta_for_perspective["perspective_axes"] = perspective_spec.get("axes")
        meta_path = Path(perspective_spec["output_dir"]) / "_preprocess_meta.json"
        os.makedirs(meta_path.parent, exist_ok=True)
        logging.info("preprocess_h5_directory: writing metadata to %s", meta_path)
        with open(meta_path, "w") as f:
            json.dump(meta_for_perspective, f, indent=2)
    logging.info(f"Done preprocessing {len(fnames)} files -> {output_dir}  (perspectives: {perspective_specs})")


def run(
    input_path,           # str | list[str]
    output_path,          # str | list[str]  (same cardinality as input_path)
    preprocess_cfg,
    path_resolver: Optional[Callable] = None,
    device: str = "cpu",
    volume_filter: str = ".*\\.h5",
    volume_limit: Optional[int] = None,
    recons_key: str = "reconstruction_mvue",
    sensmaps_key: str = "sensitivity_maps",
    sensmap_path: Optional[Any] = None,  # str | list[str] | None
    bart_path: Optional[str] = None,
):
    """Entry point called by ``preprocess_train_dataset_task``."""
    from omegaconf import ListConfig  # local import to avoid hard dep at module level
    if isinstance(input_path, str):
        input_path = [input_path]
    elif isinstance(input_path, ListConfig):
        input_path = list(input_path)
    if isinstance(output_path, str):
        output_path = [output_path]
    elif isinstance(output_path, ListConfig):
        output_path = list(output_path)
    if sensmap_path is None:
        sensmap_path = [None] * len(input_path)
    elif isinstance(sensmap_path, (str,)):
        sensmap_path = [sensmap_path]
    elif isinstance(sensmap_path, ListConfig):
        sensmap_path = list(sensmap_path)

    assert len(input_path) == len(output_path), (
        "input_path and output_path must have the same length"
    )
    assert len(sensmap_path) == len(input_path), (
        "sensmap_path must have the same length as input_path"
    )

    num_workers: int = int(getattr(preprocess_cfg, "num_workers", 1))

    perspectives = getattr(preprocess_cfg, "perspectives", None)
    if perspectives is not None and len(input_path) == 1 and len(output_path) == 1:
        expanded = [(
            input_path[0],
            output_path[0],
            sensmap_path[0],
            [_perspective_to_spec(perspective, output_path[0]) for perspective in list(perspectives)],
        )]
    else:
        expanded = [
            (in_dir, out_dir, sm_dir, None)
            for in_dir, out_dir, sm_dir in zip(input_path, output_path, sensmap_path)
        ]

    for in_dir, out_dir, sm_dir, perspective_specs in expanded:
        preprocess_h5_directory(
            input_dir=in_dir,
            output_dir=out_dir,
            preprocess_cfg=preprocess_cfg,
            device=device,
            volume_filter=volume_filter,
            volume_limit=volume_limit,
            recons_key=recons_key,
            sensmaps_key=sensmaps_key,
            sensmap_dir=sm_dir,
            num_workers=num_workers,
            perspective_specs=perspective_specs,
            bart_path=bart_path,
        )
