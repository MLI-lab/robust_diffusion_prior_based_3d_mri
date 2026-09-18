from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from hydra import compose
from src.prefect.hydra_lock import locked_hydra_initialize
from src.prefect.task_options import task_options as remote_options

from src.flows.flow_utils.dataset_source_flow import run_conversion_pipeline, submit_download_dataset
from src.flows.flow_utils.tuned_recon.types import DatasetOutput
from src.prefect.caching import hydra_config_to_cacheable_dict
from src.prefect.sweeping import get_hydra_overwrite_list, get_hydra_overwrite_str, get_hydra_sweep_combos, get_prefixed_sweep_combo
from src.tasks.preprocess_recon_dataset_task import preprocess_recon_dataset_task
from src.tasks.preprocess_train_dataset_task import preprocess_train_dataset_task
from src.utils.wandb_utils import flatten_hydra_config


def run_dataset_source(cfg: Dict[str, Any], local_cache_path: str, wandb_params: Any, wandb_cfg: Dict[str, Any]) -> Tuple[List[DatasetOutput], List[DatasetOutput]]:
    source = cfg["data"]["source"]
    conversion = source["conversion"]
    download = source["download"]
    shared = conversion["shared"]
    raw_dataset = submit_download_dataset(
        dataset_source_enabled=source["enabled"],
        download_dataset_base_config_path=download["config_path"],
        download_dataset_config_name=download["config_name"],
        download_dataset_base_overrides=download["overrides"],
        download_dataset_cache_refresh=download["cache_refresh"],
        download_dataset_num_gpus=download["num_gpus"],
        local_cache_path=local_cache_path,
    )

    def run_conv(stage: Dict[str, Any], label_prefix: str) -> List[DatasetOutput]:
        return run_conversion_pipeline(
            dataset_source_enabled=source["enabled"],
            raw_dataset=raw_dataset,
            convert_dataset_base_config_path=stage["config_path"],
            convert_dataset_config_name=stage["config_name"],
            convert_dataset_base_overrides=stage["overrides"],
            convert_dataset_task_overrides=stage["sweep"],
            convert_dataset_cache_refresh=stage["cache_refresh"],
            convert_dataset_num_gpus=stage["num_gpus"],
            local_cache_path=local_cache_path,
            label_prefix=label_prefix,
            wandb_params=wandb_params,
            wandb_cfg=wandb_cfg,
        )

    if conversion.get("train") is not None or conversion.get("recon") is not None:
        train_outputs = run_conv(conversion.get("train") or shared, "train_conversion")
        recon_outputs = run_conv(conversion.get("recon") or shared, "recon_conversion")
        return train_outputs, recon_outputs

    outputs = run_conv(shared, "conversion")
    return outputs, outputs


def run_preprocessing(
    cfg: Dict[str, Any],
    train_dataset_outputs: List[DatasetOutput],
    recon_dataset_outputs: List[DatasetOutput],
    wandb_params: Any,
    wandb_cfg: Dict[str, Any],
    local_cache_path: str,
    bart_path: str | None,
    logger: Any,
) -> Tuple[List[DatasetOutput], List[DatasetOutput]]:
    preprocess_cfg = cfg["data"]["preprocess"]
    train_stage = preprocess_cfg["train"]
    recon_stage = preprocess_cfg["recon"]
    train_futures = []
    with locked_hydra_initialize(config_path=train_stage["config_path"], version_base="1.2"):
        for converted_paths, conversion_labels in train_dataset_outputs:
            for combo in get_hydra_sweep_combos(train_stage["sweep"]) or [[]]:
                task_cfg = compose(config_name=train_stage["config_name"], overrides=train_stage["overrides"] + get_hydra_overwrite_list(combo))
                source_kwargs = {}
                if converted_paths is not None:
                    source_kwargs = {"converted_dataset_path": converted_paths["converted"], "fold": "train"}
                labels = list(conversion_labels) + get_prefixed_sweep_combo("preprocess_train", combo)
                task_wandb = wandb_params.model_copy(update={
                    "name_aux": get_hydra_overwrite_str(labels),
                    "config": flatten_hydra_config(task_cfg),
                    **wandb_cfg,
                })
                with remote_options(num_gpus=train_stage["num_gpus"]):
                    fut = preprocess_train_dataset_task.with_options(refresh_cache=train_stage["cache_refresh"]).submit(
                        **hydra_config_to_cacheable_dict(task_cfg),
                        local_cache_path=local_cache_path,
                        bart_path=bart_path,
                        wandb_params_task=task_wandb,
                        **source_kwargs,
                    )
                train_futures.append((fut, labels))

    recon_futures = []
    with locked_hydra_initialize(config_path=recon_stage["config_path"], version_base="1.2"):
        for converted_paths, conversion_labels in recon_dataset_outputs:
            for combo in get_hydra_sweep_combos(recon_stage["sweep"]) or [[]]:
                task_cfg = compose(config_name=recon_stage["config_name"], overrides=recon_stage["overrides"] + get_hydra_overwrite_list(combo))
                source_kwargs = {}
                if converted_paths is not None:
                    source_kwargs = {"converted_dataset_path": converted_paths["converted"], "fold": "test_val"}
                labels = list(conversion_labels) + get_prefixed_sweep_combo("preprocess_recon", combo)
                task_wandb = wandb_params.model_copy(update={
                    "name_aux": get_hydra_overwrite_str(labels),
                    "config": flatten_hydra_config(task_cfg),
                    **wandb_cfg,
                })
                with remote_options(num_gpus=recon_stage["num_gpus"]):
                    fut = preprocess_recon_dataset_task.with_options(refresh_cache=recon_stage["cache_refresh"]).submit(
                        **hydra_config_to_cacheable_dict(task_cfg),
                        local_cache_path=local_cache_path,
                        bart_path=bart_path,
                        wandb_params_task=task_wandb,
                        **source_kwargs,
                    )
                recon_futures.append((fut, labels))

    train_outputs: List[DatasetOutput] = []
    for fut, labels in train_futures:
        paths = fut.result()
        logger.info("Train preprocess combo %s done. Output paths: %s", labels, paths)
        train_outputs.append((paths, labels))

    recon_outputs: List[DatasetOutput] = []
    for fut, labels in recon_futures:
        paths = fut.result()
        logger.info("Recon preprocess combo %s done. Output paths: %s", labels, paths)
        recon_outputs.append((paths, labels))

    return train_outputs, recon_outputs

