from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE

from src.prefect.caching import CacheableDictConfig
from src.utils.wandb_utils import WandbParamsTask
from src.tasks.dataset_pipeline_utils import (
    cache_base_from_subfolder,
    ensure_dir,
    stable_config_hash,
    write_json,
)
from src.tasks.dataset_wandb import log_dataset_summary_to_wandb



def _copy_or_link_one(src: Path, output_dir: Path, link: bool, exists_ok: bool) -> int:
    dst = output_dir / src.name
    if dst.exists():
        if exists_ok:
            return 1
        raise FileExistsError(f"{dst} exists and exists_ok=False.")
    if link:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)
    return 1


def _copy_or_link_h5s(input_dir: Path, output_dir: Path, link: bool, exists_ok: bool, num_workers: int = 1) -> int:
    ensure_dir(output_dir)
    files = sorted(input_dir.glob("*.h5"))
    worker_count = max(1, int(num_workers))
    if worker_count > 1 and len(files) > 1:
        count = 0
        with ThreadPoolExecutor(max_workers=min(worker_count, len(files))) as executor:
            futures = [executor.submit(_copy_or_link_one, src, output_dir, link, exists_ok) for src in files]
            for fut in as_completed(futures):
                count += int(fut.result())
        return count
    return sum(_copy_or_link_one(src, output_dir, link, exists_ok) for src in files)


@task(
    cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task",
    name="Convert Dataset Task",
    tags=["dataset-conversion", "dataset-preparation"],
    version="1.0",
    retries=0,
)
def convert_dataset_task(
    dataset_name: str,
    raw_dataset: Dict[str, Any],
    conversion_cfg: CacheableDictConfig,
    local_cache_path: str,
    output_cache_subfolder: str = "converted_datasets",
    wandb_params_task: Optional[WandbParamsTask] = None,
) -> Dict[str, Any]:
    """Convert raw dataset files into the canonical train/test_val H5 layout."""
    logging.getLogger().setLevel(logging.INFO)
    raw_path = Path(raw_dataset["raw"])

    out_base = cache_base_from_subfolder(local_cache_path, output_cache_subfolder, "converted_datasets")
    cfg_hash = stable_config_hash(conversion_cfg.cfg)
    version = str(getattr(conversion_cfg.cfg, "version", "v1"))
    converted_path = out_base / dataset_name / f"{getattr(conversion_cfg.cfg, 'name', 'conversion')}_{version}_{cfg_hash}"
    train_dir = converted_path / "train"
    test_val_dir = converted_path / "test_val"
    ensure_dir(train_dir)
    ensure_dir(test_val_dir)

    mode = str(getattr(conversion_cfg.cfg, "mode", "passthrough_h5"))
    exists_ok = bool(getattr(conversion_cfg.cfg, "exists_ok", True))
    counts: Dict[str, int] = {"train": 0, "test_val": 0}

    for fold in ("train", "test_val"):
        fold_cfg = getattr(conversion_cfg.cfg, fold, None)
        if fold_cfg is None:
            continue
        input_rel = str(getattr(fold_cfg, "input_relpath", fold))
        input_dir = raw_path / input_rel
        output_dir = train_dir if fold == "train" else test_val_dir
        fold_mode = str(getattr(fold_cfg, "mode", mode))

        if exists_ok and any(output_dir.glob("*.h5")):
            counts[fold] = len(list(output_dir.glob("*.h5")))
            continue

        if not input_dir.exists():
            raise FileNotFoundError(
                f"{dataset_name}/{fold}: expected raw conversion input at {input_dir}. "
                "Check conversion.<fold>.input_relpath."
            )

        if fold_mode == "passthrough_h5":
            if not any(input_dir.glob("*.h5")):
                raise FileNotFoundError(
                    f"{dataset_name}/{fold}: no .h5 files found in raw conversion input {input_dir}. "
                    "Check conversion.<fold>.input_relpath or the extracted dataset layout."
                )
            link = bool(getattr(fold_cfg, "link", getattr(conversion_cfg.cfg, "link", False)))
            counts[fold] = _copy_or_link_h5s(input_dir, output_dir, link=link, exists_ok=exists_ok, num_workers=getattr(conversion_cfg.cfg, "num_workers", 1))
        elif fold_mode == "python":
            from src.tasks.conversion_impl import get_conversion_impl

            task_name = str(getattr(fold_cfg, "task_name", getattr(conversion_cfg.cfg, "task_name", dataset_name)))
            impl = get_conversion_impl(task_name)
            counts[fold] = int(impl(input_dir=input_dir, output_dir=output_dir, fold=fold, cfg=conversion_cfg.cfg))
        elif fold_mode in ("manual", "validate"):
            if not input_dir.exists():
                raise FileNotFoundError(f"{dataset_name}/{fold}: expected converted input at {input_dir}.")
            counts[fold] = _copy_or_link_h5s(
                input_dir,
                output_dir,
                link=bool(getattr(fold_cfg, "link", True)),
                exists_ok=exists_ok,
                num_workers=getattr(conversion_cfg.cfg, "num_workers", 1),
            )
        else:
            raise ValueError(f"Unknown conversion mode {fold_mode!r} for {dataset_name}/{fold}.")

    meta = {
        "dataset_name": dataset_name,
        "schema_version": 1,
        "raw_path": str(raw_path),
        "converted_path": str(converted_path),
        "conversion_hash": cfg_hash,
        "conversion_version": version,
        "conversion_mode": mode,
        "conversion_task_name": str(getattr(conversion_cfg.cfg, "task_name", dataset_name)),
        "folds": {"train": str(train_dir), "test_val": str(test_val_dir)},
        "counts": counts,
        "h5_schema": {
            "required_keys": ["kspace"],
            "target_keys": ["reconstruction_mvue", "reconstruction_rss", "reconstruction"],
            "canonical_kspace_layout": "Z,Coil,Y,X for multicoil volume data when applicable",
        },
    }
    meta_path = converted_path / "conversion_meta.json"
    write_json(meta_path, meta)

    result = {
        "dataset_name": dataset_name,
        "raw": str(raw_path),
        "converted": str(converted_path),
        "train": str(train_dir),
        "test_val": str(test_val_dir),
        "meta": str(meta_path),
        "counts": counts,
    }
    log_dataset_summary_to_wandb(
        stage="conversion",
        dataset_name=dataset_name,
        output_paths={"train": str(train_dir), "test_val": str(test_val_dir)},
        wandb_params_task=wandb_params_task,
        counts=counts,
    )
    return result
