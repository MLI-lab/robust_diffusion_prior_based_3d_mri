from __future__ import annotations

from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
from src.prefect.task_options import task_options as remote_options

from src.flows.flow_utils.tuned_recon.labels import group_recon_outputs
from src.flows.flow_utils.tuned_recon.prior_mesh import prior_mesh_scale_overrides
from src.flows.flow_utils.tuned_recon.types import DatasetOutput, LabeledFuture
from src.flows.flow_utils.tuning_flow import recon_run_only, tune_recon_only


def tuning_task_params(cfg: Dict[str, Any], recon_stable_keys: List[str], wandb_cfg: DictConfig, local_cache_path: str, bart_path: str | None, logger: Any, wandb_params: Any) -> Dict[str, Any]:
    recon = cfg["reconstruction"]
    tuning = recon["tuning"]
    eval_cfg = recon["evaluation"]
    visual = recon["visual_evaluation"]
    return dict(
        recon_recon_config_path=recon["config_path"],
        recon_recon_config_name=recon["config_name"],
        recon_tuning_config_path=tuning["selection_config_path"],
        recon_tuning_config_name=tuning["selection_config_name"],
        recon_tuning_overrides=tuning["selection_overrides"],
        recon_tuning_cache_refresh=tuning["cache_refresh"],
        recon_tuning_num_gpus=tuning["num_gpus"],
        common_hp_sweep_aggregate=tuning["aggregate_sweep"],
        common_run_sweep=eval_cfg["sweep"],
        common_recon_overrides=recon["overrides"] + recon.get("common_overrides", []),
        common_post_overrides=recon["post_overrides"] + recon["common_post_overrides"],
        recon_recon_sweep=recon["method_sweep"],
        recon_recon_hp_overrides=tuning["hp_overrides"],
        recon_recon_hp_sweep_tune_basic_key=tuning["hp_sweep_key"],
        recon_recon_hp_sweep_tune_extra=tuning["hp_sweep_extra"],
        recon_recon_hp_log_wandb=tuning["hp_log_wandb"],
        recon_recon_run_overrides=eval_cfg["run_overrides"],
        recon_recon_run_post_overrides=eval_cfg.get("post_overrides", []) or [],
        recon_recon_cache_refresh=eval_cfg["cache_refresh"],
        recon_recon_num_gpus=eval_cfg["num_gpus"],
        recon_recon_hp_budget_trials_per_param=tuning["coarse_budget_trials_per_param"],
        recon_recon_hp_refine_budget_trials_per_param=tuning["refine_budget_trials_per_param"],
        common_visual_run_sweep=visual["sweep"],
        common_visual_recon_overrides_extra=visual["overrides_extra"],
        common_visual_recon_post_overrides=visual.get("post_overrides", []) or [],
        common_visual_recon_cache_refresh=visual["cache_refresh"],
        common_visual_recon_num_gpus=visual["num_gpus"],
        wandb_cfg=wandb_cfg,
        local_cache_path=local_cache_path,
        bart_path=bart_path,
        logger=logger,
        wandb_params=wandb_params,
        recon_stable_keys=recon_stable_keys,
    )


def transfer_task_params(cfg: Dict[str, Any], recon_stable_keys: List[str], wandb_cfg: DictConfig, local_cache_path: str, bart_path: str | None, logger: Any, wandb_params: Any) -> Dict[str, Any]:
    recon = cfg["reconstruction"]
    eval_cfg = recon["evaluation"]
    visual = recon["visual_evaluation"]
    return dict(
        recon_recon_config_path=recon["config_path"],
        recon_recon_config_name=recon["config_name"],
        common_recon_overrides=recon["overrides"] + recon.get("common_overrides", []),
        common_post_overrides=recon["post_overrides"] + recon["common_post_overrides"],
        recon_recon_sweep=recon["method_sweep"],
        recon_recon_run_overrides=eval_cfg["run_overrides"],
        recon_recon_run_post_overrides=eval_cfg.get("post_overrides", []) or [],
        common_run_sweep=eval_cfg["sweep"],
        common_visual_run_sweep=visual["sweep"],
        common_visual_recon_overrides_extra=visual["overrides_extra"],
        common_visual_recon_post_overrides=visual.get("post_overrides", []) or [],
        common_visual_recon_cache_refresh=visual["cache_refresh"],
        common_visual_recon_num_gpus=visual["num_gpus"],
        recon_recon_cache_refresh=eval_cfg["cache_refresh"],
        recon_recon_num_gpus=eval_cfg["num_gpus"],
        recon_stable_keys=recon_stable_keys,
        wandb_cfg=wandb_cfg,
        local_cache_path=local_cache_path,
        bart_path=bart_path,
        logger=logger,
        wandb_params=wandb_params,
    )


def run_tuned_recon_for_groups(
    cfg: Dict[str, Any],
    train_groups: Dict[tuple, List[LabeledFuture]],
    preprocess_recon_outputs: List[DatasetOutput],
    recon_stable_keys: List[str],
    wandb_cfg: DictConfig,
    local_cache_path: str,
    bart_path: str | None,
    logger: Any,
    wandb_params: Any,
    extra_prefix_labels: List[Tuple[str, Any]] | None = None,
) -> Tuple[List[LabeledFuture], List[LabeledFuture]]:
    extra_prefix_labels = extra_prefix_labels or []
    all_recon_results: List[LabeledFuture] = []
    all_visual_results: List[LabeledFuture] = []
    rep_futures = []
    params = tuning_task_params(cfg, recon_stable_keys, wandb_cfg, local_cache_path, bart_path, logger, wandb_params)

    hp_sharing = cfg["reconstruction"].get("hp_sharing", {}) or {}
    recon_groups = group_recon_outputs(
        preprocess_recon_outputs, recon_stable_keys, hp_sharing.get("representative", {}) or {}
    )
    if len(recon_groups) < len(preprocess_recon_outputs):
        logger.info(
            "HP-sharing across recon preprocessing: %d tuning group(s) for %d recon dataset(s), representatives=%s",
            len(recon_groups), len(preprocess_recon_outputs),
            [members[0][1] for members in recon_groups.values()],
        )

    for group_members in train_groups.values():
        rep_future, rep_combo = group_members[0]
        for recon_members in recon_groups.values():
            rep_paths, rep_labels = recon_members[0]
            logger.info("Submitting tuned recon for train combo representative: %s | recon preprocess: %s", rep_combo, rep_labels)
            recon_input = {"pretrained_model_path": rep_future}
            if rep_paths is not None:
                recon_input["data_path_recon_override"] = rep_paths.get("recon")
            scale_overrides = prior_mesh_scale_overrides(cfg, rep_combo, rep_labels, logger)
            group_params = dict(params)
            group_params["common_recon_overrides"] = list(params["common_recon_overrides"]) + scale_overrides
            fut = tune_recon_only.submit(
                task_to_tune_recon_input=recon_input,
                extra_combo_labels=extra_prefix_labels + rep_combo + list(rep_labels),
                **group_params,
            )
            rep_futures.append((fut, group_members, recon_members))

    transfer_futures = []
    transfer_params = transfer_task_params(cfg, recon_stable_keys, wandb_cfg, local_cache_path, bart_path, logger, wandb_params)
    for tune_future, group_members, recon_members in rep_futures:
        recon_results, visual_results, hp_registry = tune_future.result()
        all_recon_results.extend(recon_results)
        all_visual_results.extend(visual_results)
        for train_idx, (nr_future, nr_combo) in enumerate(group_members):
            for recon_idx, (preprocess_paths, preprocess_labels) in enumerate(recon_members):
                if train_idx == 0 and recon_idx == 0:
                    continue  # already covered by the tuned representative run above
                logger.info(
                    "Submitting HP-transfer recon for train combo: %s | recon preprocess: %s",
                    nr_combo, preprocess_labels,
                )
                recon_input = {"pretrained_model_path": nr_future}
                if preprocess_paths is not None:
                    recon_input["data_path_recon_override"] = preprocess_paths.get("recon")
                nr_scale_overrides = prior_mesh_scale_overrides(cfg, nr_combo, preprocess_labels, logger)
                nr_params = dict(transfer_params)
                nr_params["common_recon_overrides"] = list(transfer_params["common_recon_overrides"]) + nr_scale_overrides
                transfer_future = recon_run_only.submit(
                    hp_registry=hp_registry,
                    task_to_tune_recon_input=recon_input,
                    extra_combo_labels=extra_prefix_labels + nr_combo + list(preprocess_labels),
                    **nr_params,
                )
                transfer_futures.append(transfer_future)

    for transfer_future in transfer_futures:
        recon_results, visual_results = transfer_future.result()
        all_recon_results.extend(recon_results)
        all_visual_results.extend(visual_results)

    return all_recon_results, all_visual_results
