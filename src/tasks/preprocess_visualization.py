from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.wandb_utils import WandbParamsTask, wandb_kwargs_for_prefect_task
from src.tasks.dataset_wandb import build_dataset_summary_logs


def _as_dirs(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[Path] = []
        for item in value:
            out.extend(_as_dirs(item))
        return out
    return [Path(str(value))]


def _iter_h5s(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".h5":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.h5")))
    return sorted(files)


def _to_complex_array(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64, copy=False)
    if arr.ndim > 0 and arr.shape[-1] == 2:
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex64, copy=False)
    return arr.astype(np.float32, copy=False)


def _normalize_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    vals = x[finite]
    lo, hi = np.percentile(vals, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _spatialize_complex(arr: np.ndarray) -> np.ndarray:
    arr = _to_complex_array(arr)
    if arr.ndim == 4:
        # Sensmaps/k-space often include a coil axis. Visualize a representative
        # coil so both magnitude and phase remain physically meaningful.
        coil_axis = int(np.argmin(arr.shape))
        return np.moveaxis(arr, coil_axis, 0)[0]
    return arr


def _center_slices(volume: np.ndarray) -> Dict[str, np.ndarray]:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume for visualization, got shape {volume.shape}.")
    z, y, x = volume.shape
    return {
        "axial_z": volume[z // 2, :, :],
        "coronal_y": volume[:, y // 2, :],
        "sagittal_x": volume[:, :, x // 2],
    }


def _add_volume_images(log_dict: Dict[str, Any], prefix: str, arr: np.ndarray) -> None:
    import wandb

    arr = _spatialize_complex(arr)
    if arr.ndim != 3:
        logging.warning("Skipping W&B preprocessing visualization for %s with unsupported shape %s", prefix, arr.shape)
        return

    if np.iscomplexobj(arr):
        mag_vol = np.abs(arr)
        phase_vol = np.angle(arr)
    else:
        mag_vol = arr.astype(np.float32)
        phase_vol = None

    for orient, sl in _center_slices(mag_vol).items():
        log_dict[f"{prefix}/{orient}/magnitude"] = wandb.Image(_normalize_image(sl))
    if phase_vol is not None:
        for orient, sl in _center_slices(phase_vol).items():
            log_dict[f"{prefix}/{orient}/phase"] = wandb.Image(_normalize_image(sl))


def _auto_keys(hf: h5py.File, key_candidates: list[str]) -> list[str]:
    keys: list[str] = []
    for key in key_candidates:
        if key and key in hf and key not in keys:
            keys.append(key)
    if not keys:
        for key in hf.keys():
            obj = hf[key]
            if isinstance(obj, h5py.Dataset) and getattr(obj, "ndim", 0) >= 3:
                keys.append(key)
    return keys


def log_preprocess_outputs_to_wandb(
    *,
    output_paths: Dict[str, Any],
    visualize_cfg: Optional[Any],
    wandb_params_task: Optional[WandbParamsTask],
    key_candidates: list[str],
    name_aux: str,
    summary_stage: Optional[str] = None,
    dataset_name: Optional[str] = None,
    counts: Optional[Dict[str, Any]] = None,
) -> None:
    visualize_enabled = visualize_cfg is not None and bool(getattr(visualize_cfg, "enabled", False))
    summary_enabled = summary_stage is not None
    if not visualize_enabled and not summary_enabled:
        return
    if wandb_params_task is None:
        logging.warning("Preprocess W&B logging requested, but no wandb_params_task was provided.")
        return

    import wandb

    from src.prefect.wandb_lock import locked_wandb_init

    params = wandb_params_task.model_copy(update={"name_aux": name_aux})
    metrics: Dict[str, Any] = {}
    table_rows: list[list[Any]] = []
    if summary_stage is not None:
        metrics, table_rows = build_dataset_summary_logs(
            stage=summary_stage,
            output_paths=output_paths,
            counts=counts,
        )

    with locked_wandb_init(**wandb_kwargs_for_prefect_task(params)):
        if metrics:
            wandb.log(metrics)
            if wandb.run is not None:
                wandb.run.summary.update(metrics)
                if dataset_name is not None:
                    wandb.run.summary[f"{summary_stage}/dataset_name"] = dataset_name
        if table_rows and summary_stage is not None:
            table = wandb.Table(
                columns=["split", "num_files", "num_h5_files", "bytes", "gb", "paths"],
                data=table_rows,
            )
            wandb.log({f"{summary_stage}/summary_table": table})

        if not visualize_enabled:
            return

        num_volumes = int(getattr(visualize_cfg, "num_volumes", 1))
        if num_volumes <= 0:
            return

        files = _iter_h5s(_as_dirs(list(output_paths.values())))
        if not files:
            logging.warning("Preprocess visualization requested, but no H5 outputs found in %s", output_paths)
            return

        if isinstance(visualize_cfg, DictConfig):
            configured_keys = OmegaConf.select(visualize_cfg, "keys", default=None)
        elif isinstance(visualize_cfg, dict):
            configured_keys = visualize_cfg.get("keys")
        else:
            configured_keys = getattr(visualize_cfg, "keys", None)
            if callable(configured_keys):
                configured_keys = None
        configured_keys_list = list(configured_keys) if configured_keys is not None else []

        log_dict: Dict[str, Any] = {}
        for file_idx, file_path in enumerate(files[:num_volumes]):
            with h5py.File(file_path, "r", locking=False) as hf:
                keys = configured_keys_list or _auto_keys(hf, key_candidates)
                for key in keys:
                    if key not in hf:
                        continue
                    arr = hf[key][()]
                    prefix = f"preprocess/{file_idx:02d}_{file_path.stem}/{key}"
                    _add_volume_images(log_dict, prefix, arr)
        if log_dict:
            wandb.log(log_dict)
