"""Train preprocessing implementation"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, Optional


from omegaconf import ListConfig

from src.prefect.caching import CacheableDictConfig
from src.utils.device_utils import get_all_devices
from src.tasks.dataset_pipeline_utils import cache_base_from_subfolder, concrete_path_resolver


# Dispatch table

def _get_preprocess_impl(task_name: str):
    # "generic_h5", "cc359", "stanford_3d", or any alias treated generically.
    # All preprocessing inputs are expected to be converted H5 directories.
    from src.tasks.preprocess_impl import generic_h5_preprocess as impl
    return impl


# Helpers


# Task

def run_train_preprocess(
    task_name: str,
    input_cfg: CacheableDictConfig,
    output_cache_subfolder: str,
    preprocess_cfg: CacheableDictConfig,
    local_cache_path: str,
    bart_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    task_name:
        Dispatch key, currently treated as generic H5 preprocessing.
    input_cfg:
        Hydra config with fields ``data_path_train``, ``data_path_val``
        (plain concrete paths) pointing to H5 directories.
    output_cache_subfolder:
        Subfolder under ``local_cache_path`` where preprocessed files are written.
    preprocess_cfg:
        Fields used: ``scale_target_by_kspacenorm``, ``target_scaling_factor``,
        ``target_interpolate_by_factor``, ``target_interpolation_method``, plus
        H5 read/write keys such as ``recons_key`` and ``sensmaps_key``.
    local_cache_path:
        Concrete base path resolved by the flow.
    bart_path:
        Optional concrete BART installation path resolved by the flow.

    Returns
    -------
    dict with keys ``"train"`` and ``"val"`` -> resolved output paths
    (list[str] or str).
    """
    logging.getLogger().setLevel(logging.INFO)

    devices = get_all_devices()
    device = devices[0] if devices else "cpu"

    path_resolver: Callable = concrete_path_resolver(local_cache_path)

    impl = _get_preprocess_impl(task_name)

    cfg = preprocess_cfg.cfg
    in_cfg = input_cfg.cfg
    # Resolve output base path
    out_base = str(cache_base_from_subfolder(local_cache_path, output_cache_subfolder, "preprocess_cache"))

    interp_factor_cfg = getattr(cfg, "target_interpolate_by_factor", 1.0)
    # Handle both scalar and list/ListConfig (e.g., diverse factors)
    try:
        interp_factor = float(interp_factor_cfg)
    except (TypeError, ValueError):
        # ListConfig or other non-scalar - use a placeholder
        interp_factor = "diverse"
    interp_method = str(getattr(cfg, "target_interpolation_method", "fourier"))

    from omegaconf import OmegaConf
    _hash_payload = json.dumps(
        {
            "input": OmegaConf.to_container(in_cfg, resolve=False),
            "preprocess": OmegaConf.to_container(cfg, resolve=False),
        },
        sort_keys=True,
        default=str,
    ).encode()
    cfg_hash = hashlib.sha256(_hash_payload).hexdigest()[:8]

    version = str(getattr(cfg, "version", "v1"))
    subdir = f"{task_name}_{version}_scale{interp_factor}_{interp_method}_{cfg_hash}"
    out_dir = os.path.join(out_base, subdir)
    logging.info(f"run_train_preprocess: output dir -> {out_dir}")

    # Extra per-impl kwargs from config (fall back gracefully)
    extra_kwargs: Dict[str, Any] = {}
    for key in ("volume_filter", "volume_limit", "recons_key", "sensmaps_key"):
        if hasattr(cfg, key):
            extra_kwargs[key] = getattr(cfg, key)

    output_paths: Dict[str, Any] = {}

    for fold in ("train", "val"):
        in_path_raw = getattr(in_cfg, f"data_path_{fold}", None)
        if in_path_raw is None:
            logging.info(f"run_train_preprocess: no data_path_{fold} configured - skipping.")
            output_paths[fold] = None
            continue

        in_path = path_resolver(in_path_raw)
        if in_path is None:
            logging.info(f"run_train_preprocess: data_path_{fold} resolved to None - skipping.")
            output_paths[fold] = None
            continue

        fold_kwargs = dict(extra_kwargs)
        fold_volume_filter = getattr(cfg, f"volume_filter_{fold}", None)
        if fold_volume_filter is not None:
            fold_kwargs["volume_filter"] = fold_volume_filter

        sensmap_raw = getattr(in_cfg, f"data_path_sensmaps_{fold}", None)
        if sensmap_raw is None and fold == "val":
            sensmap_raw = getattr(in_cfg, "data_path_sensmaps_train", None)
        sensmap_path = path_resolver(sensmap_raw) if sensmap_raw is not None else None

        out_path_fold = os.path.join(out_dir, fold)

        # Handle list of paths (e.g. cc359 multi-perspective)
        # Note: Hydra path resolver may return ListConfig, so check for both.
        if isinstance(in_path, (list, ListConfig)):
            in_path = list(in_path)
            out_path_list = [
                os.path.join(out_path_fold, str(i)) for i in range(len(in_path))
            ]
            # sensmap_path is also a parallel list when in_path is a list.
            sensmap_list = list(sensmap_path) if isinstance(sensmap_path, (list, ListConfig)) else [sensmap_path] * len(in_path)
            for ip, op, sp in zip(in_path, out_path_list, sensmap_list):
                impl.run(
                    input_path=ip,
                    output_path=op,
                    preprocess_cfg=cfg,
                    path_resolver=path_resolver,
                    device=device,
                    sensmap_path=sp,
                    bart_path=bart_path,
                    **fold_kwargs,
                )
            output_paths[fold] = out_path_list
        else:
            impl.run(
                input_path=in_path,
                output_path=out_path_fold,
                preprocess_cfg=cfg,
                path_resolver=path_resolver,
                device=device,
                sensmap_path=sensmap_path,
                bart_path=bart_path,
                **fold_kwargs,
            )
            perspectives = getattr(cfg, "perspectives", None)
            if perspectives is not None:
                output_paths[fold] = [os.path.join(out_path_fold, str(p)) if isinstance(p, str) else os.path.join(out_path_fold, str(p.name)) for p in list(perspectives)]
            else:
                output_paths[fold] = out_path_fold

    logging.info(f"run_train_preprocess done. Output paths: {output_paths}")
    return output_paths
