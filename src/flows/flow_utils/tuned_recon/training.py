from __future__ import annotations

import os
from typing import Any, Dict, List

from hydra import compose
from src.prefect.hydra_lock import locked_hydra_initialize
from src.prefect.task_options import task_options as remote_options

from src.flows.flow_utils.tuned_recon.types import DatasetOutput, LabeledFuture
from src.prefect.caching import hydra_config_to_cacheable_dict
from src.prefect.sweeping import get_hydra_overwrite_list, get_hydra_overwrite_str, get_hydra_sweep_combos, get_prefixed_sweep_combo
from src.tasks.train_task import train_task
from src.utils.wandb_utils import flatten_hydra_config


def submit_training(
    cfg: Dict[str, Any],
    preprocess_train_outputs: List[DatasetOutput],
    wandb_params: Any,
    wandb_cfg: Dict[str, Any],
    local_cache_path: str,
) -> List[LabeledFuture]:
    train_cfg = cfg["model"]["train"]
    train_futures: List[LabeledFuture] = []

    with locked_hydra_initialize(config_path=train_cfg["config_path"], version_base="1.2"):
        for preprocess_paths, preprocess_labels in preprocess_train_outputs:
            for combo in get_hydra_sweep_combos(train_cfg["sweep"]):
                task_cfg = compose(
                    config_name=train_cfg["config_name"],
                    overrides=train_cfg["overrides"] + get_hydra_overwrite_list(combo) + train_cfg.get("post_overrides", []),
                )
                labels = list(preprocess_labels) + get_prefixed_sweep_combo("train", combo)
                task_wandb = wandb_params.model_copy(update={
                    "name_aux": get_hydra_overwrite_str(labels),
                    "config": flatten_hydra_config(task_cfg),
                    **wandb_cfg,
                })
                path_kwargs = {}
                if preprocess_paths is not None:
                    path_kwargs["data_path_train_override"] = preprocess_paths.get("train")
                    val_path = preprocess_paths.get("val")
                    if val_path is not None:
                        path_kwargs["data_path_val_override"] = val_path
                    else:
                        path_kwargs["skip_validation"] = True

                with remote_options(num_gpus=train_cfg["num_gpus"]):
                    train_futures.append((
                        train_task.with_options(refresh_cache=train_cfg["cache_refresh"]).submit(
                            **hydra_config_to_cacheable_dict(task_cfg),
                            wandb_params_task=task_wandb,
                            local_cache_path=local_cache_path,
                            **path_kwargs,
                        ),
                        labels,
                    ))

    return train_futures

