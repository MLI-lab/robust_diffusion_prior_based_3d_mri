from __future__ import annotations

from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig

from src.flows.flow_utils.tuned_recon.data import run_dataset_source, run_preprocessing
from src.flows.flow_utils.tuned_recon.labels import group_train_futures, override_sweep_key
from src.flows.flow_utils.tuned_recon.plotting import submit_aggregate_plot, submit_sweep_plots, submit_visual_plots
from src.flows.flow_utils.tuned_recon.reconstruction import run_tuned_recon_for_groups
from src.flows.flow_utils.tuned_recon.training import submit_training
from src.flows.flow_utils.tuned_recon.types import DatasetOutput, LabeledFuture


def _prepare(
    cfg: Dict[str, Any],
    wandb_cfg: DictConfig,
    local_cache_path: str,
    bart_path: str | None,
    logger: Any,
    wandb_params: Any,
) -> Tuple[Dict[tuple, List[LabeledFuture]], List[DatasetOutput], List[str]]:
    train_source_outputs, recon_source_outputs = run_dataset_source(cfg, local_cache_path, wandb_params, wandb_cfg)
    preprocess_train_outputs, preprocess_recon_outputs = run_preprocessing(
        cfg,
        train_source_outputs,
        recon_source_outputs,
        wandb_params,
        wandb_cfg,
        local_cache_path,
        bart_path,
        logger,
    )
    train_futures = submit_training(cfg, preprocess_train_outputs, wandb_params, wandb_cfg, local_cache_path)
    hp_sharing = cfg["reconstruction"].get("hp_sharing", {}) or {}
    stable_keys = hp_sharing.get("assume_same", []) or []
    representative = hp_sharing.get("representative", {}) or {}
    train_groups = group_train_futures(train_futures, stable_keys, representative)
    logger.info(
        "HP-sharing: stable keys=%s representative=%s | %d train tuning group(s) for %d trained model(s)",
        stable_keys, representative, len(train_groups), len(train_futures),
    )
    return train_groups, preprocess_recon_outputs, stable_keys


def run_standard(
    cfg: Dict[str, Any],
    wandb_cfg: DictConfig,
    local_cache_path: str,
    bart_path: str | None,
    logger: Any,
    wandb_params: Any,
) -> None:
    train_groups, preprocess_recon_outputs, recon_stable_keys = _prepare(
        cfg, wandb_cfg, local_cache_path, bart_path, logger, wandb_params
    )
    recon_results, visual_results = run_tuned_recon_for_groups(
        cfg,
        train_groups,
        preprocess_recon_outputs,
        recon_stable_keys,
        wandb_cfg,
        local_cache_path,
        bart_path,
        logger,
        wandb_params,
    )
    submit_sweep_plots(cfg, recon_results, wandb_params, wandb_cfg, name_aux="")
    submit_visual_plots(cfg, visual_results, wandb_params, wandb_cfg, name_aux="visual")


def run_index_holdout(
    cfg: Dict[str, Any],
    wandb_cfg: DictConfig,
    local_cache_path: str,
    bart_path: str | None,
    logger: Any,
    wandb_params: Any,
) -> None:
    train_groups, preprocess_recon_outputs, recon_stable_keys = _prepare(
        cfg, wandb_cfg, local_cache_path, bart_path, logger, wandb_params
    )
    holdout = cfg["holdout"]["index"]
    holdout_values = list(dict.fromkeys([int(value) for value in holdout["values"]]))
    if len(holdout_values) < 2:
        raise ValueError("holdout.index.values must contain at least two unique indices.")

    combined_recon_results: List[LabeledFuture] = []
    combined_visual_results: List[LabeledFuture] = []
    base_hp_sweep = cfg["reconstruction"]["tuning"]["aggregate_sweep"]
    base_run_sweep = cfg["reconstruction"]["evaluation"]["sweep"]
    base_visual_sweep = cfg["reconstruction"]["visual_evaluation"]["sweep"]

    for holdout_idx in holdout_values:
        split_cfg = cfg.copy()
        split_cfg["reconstruction"] = {**cfg["reconstruction"]}
        split_cfg["reconstruction"]["tuning"] = {**cfg["reconstruction"]["tuning"]}
        split_cfg["reconstruction"]["evaluation"] = {**cfg["reconstruction"]["evaluation"]}
        split_cfg["reconstruction"]["visual_evaluation"] = {**cfg["reconstruction"]["visual_evaluation"]}

        hp_indices = [idx for idx in holdout_values if idx != holdout_idx]
        run_indices = [holdout_idx]
        split_key = holdout["split_key"]
        split_cfg["reconstruction"]["tuning"]["aggregate_sweep"] = override_sweep_key(base_hp_sweep, split_key, hp_indices)
        split_cfg["reconstruction"]["evaluation"]["sweep"] = override_sweep_key(base_run_sweep, split_key, run_indices)
        split_cfg["reconstruction"]["visual_evaluation"]["sweep"] = override_sweep_key(base_visual_sweep, split_key, run_indices)

        holdout_label = [("holdout_idx", holdout_idx)]
        logger.info("Running index holdout: holdout=%s, tuning indices=%s", holdout_idx, hp_indices)
        recon_results, visual_results = run_tuned_recon_for_groups(
            split_cfg,
            train_groups,
            preprocess_recon_outputs,
            recon_stable_keys,
            wandb_cfg,
            local_cache_path,
            bart_path,
            logger,
            wandb_params,
            extra_prefix_labels=holdout_label,
        )
        combined_recon_results.extend(recon_results)
        combined_visual_results.extend(visual_results)
        submit_sweep_plots(
            cfg,
            recon_results,
            wandb_params,
            wandb_cfg,
            name_aux=f"holdout_{holdout_idx}",
            name_prefix=f"holdout_{holdout_idx}_",
        )
        submit_visual_plots(
            cfg,
            visual_results,
            wandb_params,
            wandb_cfg,
            name_aux=f"holdout_{holdout_idx}_visual",
            name_prefix=f"holdout_{holdout_idx}_",
        )

    submit_aggregate_plot(cfg, combined_recon_results, wandb_params, wandb_cfg)
