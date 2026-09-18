from __future__ import annotations

import os
from typing import Any, Dict, List

from hydra import compose
from src.prefect.hydra_lock import locked_hydra_initialize
from omegaconf import OmegaConf
from src.prefect.task_options import task_options as remote_options

from src.flows.flow_utils.tuned_recon.labels import iter_split_groups
from src.flows.flow_utils.tuned_recon.types import LabeledFuture
from src.prefect.caching import CacheableListConfig, hydra_config_to_cacheable_dict
from src.prefect.sweeping import get_pandas_dataframe_from_sweep_combos
from src.tasks.plot_3d_mri_visual_task import plot_3d_mri_visual_task
from src.tasks.plot_task import seaborn_plot_sweep_results_auto
from src.utils.wandb_utils import flatten_hydra_config


def unwrap_list_cfg_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in kwargs.items():
        if key.endswith("_cfg") and isinstance(value, CacheableListConfig):
            result[key[:-4]] = list(OmegaConf.to_container(value.cfg, resolve=True))
        else:
            result[key] = value
    return result


def submit_sweep_plots(
    cfg: Dict[str, Any],
    results: List[LabeledFuture],
    wandb_params: Any,
    wandb_cfg: Dict[str, Any],
    name_aux: str,
    name_prefix: str = "",
) -> None:
    if not results:
        return
    plot_cfg = cfg["reporting"]["sweep_plot"]
    result_futures = [result[0] for result in results]
    df_sweep = get_pandas_dataframe_from_sweep_combos([result[1] for result in results])

    with locked_hydra_initialize(config_path=plot_cfg["config_path"], version_base="1.2"):
        hydra_cfg = compose(config_name=plot_cfg["config_name"], overrides=plot_cfg["overrides"])
        cfg_dict = hydra_config_to_cacheable_dict(hydra_cfg)
        task_wandb = wandb_params.model_copy(update={
            "name_aux": name_aux,
            "config": flatten_hydra_config(hydra_cfg),
            **wandb_cfg,
        })
        with remote_options(num_gpus=plot_cfg["num_gpus"]):
            seaborn_plot_sweep_results_auto.submit(
                results=result_futures,
                df_sweep=df_sweep,
                wandb_params_task=task_wandb,
                **cfg_dict,
            ).wait()

        for split_label, split_results, split_df in iter_split_groups(results, df_sweep, plot_cfg.get("split_params", [])):
            split_wandb = wandb_params.model_copy(update={
                "name_aux": f"{name_prefix}split_{split_label}",
                "config": flatten_hydra_config(hydra_cfg),
                **wandb_cfg,
            })
            with remote_options(num_gpus=plot_cfg["num_gpus"]):
                seaborn_plot_sweep_results_auto.submit(
                    results=[result[0] for result in split_results],
                    df_sweep=split_df,
                    wandb_params_task=split_wandb,
                    **cfg_dict,
                ).wait()


def submit_visual_plots(
    cfg: Dict[str, Any],
    visual_results: List[LabeledFuture],
    wandb_params: Any,
    wandb_cfg: Dict[str, Any],
    name_aux: str,
    name_prefix: str = "",
) -> None:
    visual_cfg = cfg["reporting"]["visual_plot"]
    if not visual_cfg.get("enabled") or not visual_results:
        return

    df_sweep = get_pandas_dataframe_from_sweep_combos([result[1] for result in visual_results])
    with locked_hydra_initialize(config_path=visual_cfg["config_path"], version_base="1.2"):
        hydra_cfg = compose(config_name=visual_cfg["config_name"], overrides=visual_cfg["overrides"])
        cfg_dict = unwrap_list_cfg_kwargs(hydra_config_to_cacheable_dict(hydra_cfg))
        task_wandb = wandb_params.model_copy(update={
            "name_aux": name_aux,
            "config": flatten_hydra_config(hydra_cfg),
            **wandb_cfg,
        })
        with remote_options(num_gpus=visual_cfg["num_gpus"]):
            plot_3d_mri_visual_task.submit(
                results=[result[0] for result in visual_results],
                df_sweep=df_sweep,
                wandb_params_task=task_wandb,
                **cfg_dict,
            ).wait()

        for split_label, split_results, split_df in iter_split_groups(visual_results, df_sweep, visual_cfg.get("split_params", [])):
            split_wandb = wandb_params.model_copy(update={
                "name_aux": f"{name_prefix}visual_split_{split_label}",
                "config": flatten_hydra_config(hydra_cfg),
                **wandb_cfg,
            })
            with remote_options(num_gpus=visual_cfg["num_gpus"]):
                plot_3d_mri_visual_task.submit(
                    results=[result[0] for result in split_results],
                    df_sweep=split_df,
                    wandb_params_task=split_wandb,
                    **cfg_dict,
                ).wait()


def submit_aggregate_plot(cfg: Dict[str, Any], results: List[LabeledFuture], wandb_params: Any, wandb_cfg: Dict[str, Any]) -> None:
    agg_cfg = cfg["reporting"]["aggregate_plot"]
    if not agg_cfg.get("enabled") or not results:
        return
    plot_cfg = cfg["reporting"]["sweep_plot"]
    num_gpus = plot_cfg["num_gpus"] if agg_cfg.get("num_gpus") is None else agg_cfg["num_gpus"]
    df_sweep = get_pandas_dataframe_from_sweep_combos([result[1] for result in results])

    with locked_hydra_initialize(config_path=plot_cfg["config_path"], version_base="1.2"):
        hydra_cfg = compose(config_name=plot_cfg["config_name"], overrides=plot_cfg["overrides"] + agg_cfg.get("overrides", []))
        cfg_dict = hydra_config_to_cacheable_dict(hydra_cfg)
        task_wandb = wandb_params.model_copy(update={
            "name_aux": agg_cfg.get("name_aux", "aggregate_cross_holdout"),
            "config": flatten_hydra_config(hydra_cfg),
            **wandb_cfg,
        })
        with remote_options(num_gpus=num_gpus):
            seaborn_plot_sweep_results_auto.submit(
                results=[result[0] for result in results],
                df_sweep=df_sweep,
                wandb_params_task=task_wandb,
                **cfg_dict,
            ).wait()

