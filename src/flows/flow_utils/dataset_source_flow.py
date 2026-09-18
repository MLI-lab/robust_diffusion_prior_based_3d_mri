from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from hydra import compose
from src.prefect.hydra_lock import locked_hydra_initialize
from src.prefect.task_options import task_options as remote_options

from src.prefect.caching import hydra_config_to_cacheable_dict
from src.utils.wandb_utils import flatten_hydra_config
from src.prefect.sweeping import get_hydra_overwrite_list, get_hydra_overwrite_str, get_hydra_sweep_combos, get_prefixed_sweep_combo
from src.tasks.convert_dataset_task import convert_dataset_task
from src.tasks.download_dataset_task import download_dataset_task


def submit_download_dataset(
    dataset_source_enabled: bool,
    download_dataset_base_config_path: str,
    download_dataset_config_name: str,
    download_dataset_base_overrides: List[str],
    download_dataset_cache_refresh: bool,
    download_dataset_num_gpus: int,
    local_cache_path: str,
) -> Any:
    """Submit the optional raw dataset download/validation task."""
    if not dataset_source_enabled:
        return None

    with locked_hydra_initialize(
        config_path=download_dataset_base_config_path,
        version_base="1.2",
    ):
        cfg_download = compose(
            config_name=download_dataset_config_name,
            overrides=download_dataset_base_overrides,
        )
    with remote_options(num_gpus=download_dataset_num_gpus):
        return download_dataset_task.with_options(
            refresh_cache=download_dataset_cache_refresh
        ).submit(
            **hydra_config_to_cacheable_dict(cfg_download),
            local_cache_path=local_cache_path,
        )


def run_conversion_pipeline(
    dataset_source_enabled: bool,
    raw_dataset: Any,
    convert_dataset_base_config_path: str,
    convert_dataset_config_name: str,
    convert_dataset_base_overrides: List[str],
    convert_dataset_task_overrides: Dict[str, List[Any]],
    convert_dataset_cache_refresh: bool,
    convert_dataset_num_gpus: int,
    local_cache_path: str,
    label_prefix: str = "conversion",
    wandb_params: Optional[Any] = None,
    wandb_cfg: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Dict[str, Any] | None, List[Tuple[str, Any]]]]:
    """Run a conversion sweep against an already submitted raw dataset task."""
    if not dataset_source_enabled:
        return [(None, [])]

    conversion_futures = []
    with locked_hydra_initialize(
        config_path=convert_dataset_base_config_path,
        version_base="1.2",
    ):
        for convert_combo in get_hydra_sweep_combos(convert_dataset_task_overrides) or [[]]:
            cfg_convert = compose(
                config_name=convert_dataset_config_name,
                overrides=convert_dataset_base_overrides + get_hydra_overwrite_list(convert_combo),
            )
            labels = get_prefixed_sweep_combo(label_prefix, convert_combo)
            wandb_params_task = None
            if wandb_params is not None:
                wandb_params_task = wandb_params.model_copy(update={
                    "name_aux": label_prefix + ("_" + get_hydra_overwrite_str(convert_combo) if convert_combo else ""),
                    "config": flatten_hydra_config(cfg_convert),
                    **(wandb_cfg or {}),
                })
            with remote_options(num_gpus=convert_dataset_num_gpus):
                fut = convert_dataset_task.with_options(
                    refresh_cache=convert_dataset_cache_refresh
                ).submit(
                    **hydra_config_to_cacheable_dict(cfg_convert),
                    raw_dataset=raw_dataset,
                    local_cache_path=local_cache_path,
                    wandb_params_task=wandb_params_task,
                )
            conversion_futures.append((fut, labels))

    return [(future.result(), labels) for future, labels in conversion_futures]
