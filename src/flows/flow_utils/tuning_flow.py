"""Tuning flow utilities: ported from cine_gs_mri, adapted for the main project."""

from dataclasses import dataclass, asdict, field
from prefect import flow
from omegaconf import DictConfig, OmegaConf
from hydra import compose
from src.prefect.hydra_lock import locked_hydra_initialize
from typing import List, Dict, Any, Tuple, Optional
from itertools import product
import os
import math
import numpy as np
import pandas as pd
from src.prefect.caching import CacheableDict, CacheableDictConfig
from src.prefect.sweeping import get_hydra_sweep_combos
from src.prefect.task_options import task_options as remote_options
import logging
import sys
from typing import cast

from src.prefect.sweeping import (
    get_hydra_overwrite_list,
    get_hydra_sweep_combos,
    get_hydra_overwrite_str,
    get_hydra_overwrite_str_short,
    get_pandas_dataframe_from_sweep_combos,
)

from src.tasks.tuning_selection_task import tuning_selection_task, resolve_metric_matches

from prefect.logging import get_run_logger
from prefect import task
from prefect.futures import PrefectFuture

from src.utils.wandb_utils import (
    wandb_kwargs_for_prefect_task,
    gather_flow_infos_for_wandb,
    flatten_hydra_config,
)
from src.prefect.caching import hydra_config_to_cacheable_dict

from prefect import Task


@dataclass
class TuningConfig:
    task_config_path: str
    task_config_name: str
    tuning_config_path: str
    tuning_config_name: str
    tuning_overrides: List[str]
    tuning_cache_refresh: bool
    tuning_num_gpus: int
    task_overrides: List[str]
    task_sweep: Dict[str, List[Any]]
    task_hp_overrides: List[str]
    task_hp_sweep_tune_basic_key: str
    task_hp_sweep_tune_extra: Dict[str, List[Any]]
    task_hp_sweep_aggregate: Dict[str, List[Any]]
    task_hp_log_wandb: bool
    task_run_overrides: List[str]
    task_run_sweep: Dict[str, List[Any]]
    task_visual_run_overrides_extra: List[str]
    task_visual_run_sweep: Dict[str, List[Any]]
    task_visual_cache_refresh: bool
    task_visual_num_gpus: int
    task_cache_refresh: bool
    task_common_post_overrides: List[str]
    task_num_gpus: int
    task_hp_budget_trials_per_param: Optional[int] = None
    task_hp_refine_budget_trials_per_param: Optional[int] = None
    task_run_post_overrides: List[str] = field(default_factory=list)
    task_visual_run_post_overrides: List[str] = field(default_factory=list)
    task_sweep_stable_keys: List[str] = field(default_factory=list)


# helper utilities

def _is_bound_spec(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return "low" in value and "high" in value


def _is_numeric_list_sweep(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False
    for element in value:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            return False
    return True


def _discretize_numeric_list(
    values: List[Any],
    n_points: int,
    discretization_mode: str = "auto",
) -> List[Any]:
    numeric_values = [float(v) for v in values]
    low = min(numeric_values)
    high = max(numeric_values)
    if math.isclose(low, high):
        return [values[0]]
    if n_points <= 1:
        interpolated: List[float] = [low]
    else:
        positive = low > 0.0 and high > 0.0
        mode = str(discretization_mode).strip().lower()
        if mode == "log":
            use_log = positive
        elif mode == "linear":
            use_log = False
        else:
            use_log = positive and (high / low) > 100.0
        if use_log:
            log_low = math.log10(low)
            log_high = math.log10(high)
            interpolated = [
                math.pow(10.0, log_low + i * (log_high - log_low) / (n_points - 1))
                for i in range(n_points)
            ]
        else:
            interpolated = [
                low + i * (high - low) / (n_points - 1)
                for i in range(n_points)
            ]
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    if all_int:
        rounded = sorted(set(int(round(v)) for v in interpolated))
        return rounded if rounded else [int(round(low))]
    return interpolated


def _product_of_counts(counts: Dict[str, int]) -> int:
    result = 1
    for value in counts.values():
        result *= max(1, int(value))
    return result


def _allocate_counts_with_budget(
    counts: Dict[str, int],
    expandable_keys: List[str],
    budget: int,
) -> Dict[str, int]:
    if budget <= 0 or len(expandable_keys) == 0:
        return counts
    current_counts = dict(counts)
    current_product = _product_of_counts(current_counts)
    if current_product >= budget:
        return current_counts
    while True:
        best_key = None
        best_product = current_product
        for key in expandable_keys:
            next_product = (current_product // current_counts[key]) * (current_counts[key] + 1)
            if next_product <= budget and next_product > best_product:
                best_product = next_product
                best_key = key
        if best_key is None:
            break
        current_counts[best_key] += 1
        current_product = best_product
    return current_counts


def _discretize_bound_spec(
    spec: Dict[str, Any],
    n_points: int,
    discretization_mode: str = "auto",
) -> List[Any]:
    low = float(spec["low"])
    high = float(spec["high"])
    if high < low:
        low, high = high, low
    if math.isclose(low, high):
        n_points = 1
    if n_points <= 1:
        values: List[float] = [low]
    else:
        mode = str(discretization_mode).strip().lower()
        if mode == "log":
            scale = "log"
        elif mode == "linear":
            scale = "linear"
        else:
            scale = str(spec.get("scale", "linear")).strip().lower()
        if scale == "log":
            if low <= 0.0:
                values = [low + i * (high - low) / (n_points - 1) for i in range(n_points)]
            else:
                log_low = math.log10(low)
                log_high = math.log10(high)
                values = [
                    math.pow(10.0, log_low + i * (log_high - log_low) / (n_points - 1))
                    for i in range(n_points)
                ]
        else:
            values = [low + i * (high - low) / (n_points - 1) for i in range(n_points)]
    dtype = str(spec.get("dtype", "float")).strip().lower()
    if dtype in ["int", "integer"]:
        rounded = sorted(set(int(round(v)) for v in values))
        return rounded if rounded else [int(round(low))]
    return values


def _build_budgeted_tuning_hp_sweep(
    hp_sweep: Dict[str, Any],
    task_hp_budget_trials_per_param: Optional[int],
    discretization_mode: str = "auto",
) -> Dict[str, Any]:
    if task_hp_budget_trials_per_param is None or task_hp_budget_trials_per_param <= 0:
        return hp_sweep

    tunable_numeric_keys: List[str] = []
    fixed_combo_count = 1
    count_per_key: Dict[str, int] = {}
    expandable_keys: List[str] = []

    for key, value in hp_sweep.items():
        if _is_bound_spec(value):
            tunable_numeric_keys.append(key)
            spec = cast(Dict[str, Any], value)
            if "num" in spec:
                count_per_key[key] = max(2, int(spec["num"]))
            else:
                count_per_key[key] = 2
                expandable_keys.append(key)
            continue
        if _is_numeric_list_sweep(value):
            tunable_numeric_keys.append(key)
            unique_values = sorted(set(cast(List[Any], value)))
            count_per_key[key] = max(2, len(unique_values))
            expandable_keys.append(key)
            continue
        if isinstance(value, list):
            fixed_combo_count *= max(1, len(value))

    if len(tunable_numeric_keys) == 0:
        return hp_sweep

    budget_per_param = max(1, int(task_hp_budget_trials_per_param))
    n_tunable = max(1, len(tunable_numeric_keys))
    budget_for_numeric = max(1, budget_per_param * n_tunable // max(1, fixed_combo_count))
    count_per_key = _allocate_counts_with_budget(count_per_key, expandable_keys, budget_for_numeric)

    expanded_hp_sweep = dict(hp_sweep)
    for key in tunable_numeric_keys:
        value = hp_sweep[key]
        n_points = max(2, count_per_key[key])
        if _is_bound_spec(value):
            expanded_hp_sweep[key] = _discretize_bound_spec(
                cast(Dict[str, Any], value), n_points, discretization_mode=discretization_mode,
            )
        else:
            expanded_hp_sweep[key] = _discretize_numeric_list(
                cast(List[Any], value), n_points, discretization_mode=discretization_mode,
            )
    return expanded_hp_sweep


def _split_stable_keys(
    sweep: Dict[str, List[Any]],
    stable_keys: List[str],
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], List[str]]:
    """Partition a sweep dict into (tuning_sweep, stable_sweep, representative_overrides)."""
    cols = list(sweep.keys())
    matched_stable: List[str] = []
    for sk in stable_keys:
        if sk in cols:
            matched_stable.append(sk)
            continue
        sk_bare = sk.lstrip("+~").lstrip("=")
        sk_leaf = sk_bare.rsplit(".", 1)[-1]
        for c in cols:
            c_bare = c.lstrip("+~").lstrip("=")
            if c_bare == sk_bare or c_bare.rsplit(".", 1)[-1] == sk_leaf:
                matched_stable.append(c)
                break

    matched_set = set(matched_stable)
    tuning_sweep = {k: v for k, v in sweep.items() if k not in matched_set}
    stable_sweep = {k: v for k, v in sweep.items() if k in matched_set}

    representative_overrides: List[str] = []
    for k, vals in stable_sweep.items():
        first = (vals[0] if isinstance(vals, list) and vals else vals)
        representative_overrides.append(get_hydra_overwrite_list([(k, first)])[0])

    return tuning_sweep, stable_sweep, representative_overrides


def _parse_best_overrides(overrides: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if str(value).strip().lower() in ["true", "false"]:
            parsed[key] = str(value).strip().lower() == "true"
            continue
        try:
            number = float(value)
            if number.is_integer() and str(value).strip().isdigit():
                parsed[key] = int(number)
            else:
                parsed[key] = number
        except ValueError:
            parsed[key] = value
    return parsed


def _build_refined_tuning_hp_sweep(
    hp_sweep_base: Dict[str, Any],
    phase1_hp_sweep: Dict[str, Any],
    phase1_best_overrides: List[str],
    task_hp_refine_budget_trials_per_param: Optional[int],
    phase1_anchor_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if task_hp_refine_budget_trials_per_param is None or task_hp_refine_budget_trials_per_param <= 0:
        return phase1_hp_sweep

    tunable_keys = [
        key for key, value in phase1_hp_sweep.items()
        if _is_numeric_list_sweep(value)
    ]
    if len(tunable_keys) == 0:
        logging.info("Skipping phase-2 refinement: no tunable numeric parameters found.")
        return phase1_hp_sweep

    best_values = _parse_best_overrides(phase1_best_overrides)
    if phase1_anchor_values is not None:
        best_values = dict(best_values) | dict(phase1_anchor_values)

    refined_sweep: Dict[str, Any] = dict(hp_sweep_base)

    for key, value in phase1_hp_sweep.items():
        if not _is_numeric_list_sweep(value):
            continue
        sorted_vals = sorted(set(cast(List[Any], value)))
        if len(sorted_vals) < 2:
            continue
        best_value = best_values.get(key, sorted_vals[len(sorted_vals) // 2])
        try:
            best_value_float = float(best_value)
        except (TypeError, ValueError):
            continue
        distances = [abs(float(v) - best_value_float) for v in sorted_vals]
        nearest_idx = int(np.argmin(distances))
        if len(sorted_vals) >= 3:
            if nearest_idx == 0:
                neighborhood_indices = [0, 1, 2]
            elif nearest_idx == len(sorted_vals) - 1:
                neighborhood_indices = [len(sorted_vals) - 3, len(sorted_vals) - 2, len(sorted_vals) - 1]
            else:
                neighborhood_indices = [nearest_idx - 1, nearest_idx, nearest_idx + 1]
            left_idx = min(neighborhood_indices)
            right_idx = max(neighborhood_indices)
        else:
            left_idx = max(0, nearest_idx - 1)
            right_idx = min(len(sorted_vals) - 1, nearest_idx + 1)
        if left_idx == right_idx:
            continue
        low = float(sorted_vals[left_idx])
        high = float(sorted_vals[right_idx])
        if math.isclose(low, high):
            continue
        refinement_mode = "log" if (low > 0.0 and high > 0.0 and high / low > 10.0) else "linear"
        all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in sorted_vals)
        refined_sweep[key] = {
            "low": low, "high": high,
            "dtype": "int" if all_int else "float",
            "scale": refinement_mode,
        }
        logging.info("Phase-2 refinement for '%s': range=[%s, %s], scale=%s", key, low, high, refinement_mode)

    return _build_budgeted_tuning_hp_sweep(refined_sweep, task_hp_refine_budget_trials_per_param)


def _result_to_dict_for_refinement(result: Any) -> Dict[str, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        return _result_to_dict_for_refinement(result[0])
    if isinstance(result, CacheableDict):
        return dict(result.cfg)
    if isinstance(result, dict):
        return dict(result)
    cfg_value = getattr(result, "cfg", None)
    if cfg_value is not None:
        if isinstance(cfg_value, dict):
            return dict(cfg_value)
        container = OmegaConf.to_container(cfg_value, resolve=True)
        if isinstance(container, dict):
            return cast(Dict[str, Any], dict(container))
    return {}


def _get_phase1_anchor_values_from_results(
    tuning_result: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    tuning_cfg: DictConfig,
) -> Dict[str, Any]:
    if not tuning_result:
        return {}
    resolved_results = [_result_to_dict_for_refinement(task_future.result()) for task_future, _ in tuning_result]
    if not resolved_results:
        return {}
    df_results = pd.DataFrame(resolved_results)
    if df_results.empty:
        return {}
    numeric_metrics = df_results.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_metrics:
        return {}
    tune_metric_pattern = str(getattr(tuning_cfg, "tune_metric", "")).strip()
    if not tune_metric_pattern:
        return {}
    matching_metrics = resolve_metric_matches(tune_metric_pattern, numeric_metrics)
    if len(matching_metrics) != 1:
        return {}
    tuning_metric = matching_metrics[0]
    tune_metric_take_max = bool(getattr(tuning_cfg, "tune_metric_take_max", True))
    df_sweep = get_pandas_dataframe_from_sweep_combos([combo for _, combo in tuning_result])
    sweep_tune_params = [col for col in df_sweep.columns.tolist() if col.startswith("tun:")]
    if not sweep_tune_params:
        return {}
    df_plot = pd.concat([df_results[[tuning_metric]], df_sweep], axis=1)
    agg = df_plot.groupby(sweep_tune_params, dropna=False)[tuning_metric].mean().reset_index()
    if agg.empty or tuning_metric not in agg.columns:
        return {}
    best_idx = agg[tuning_metric].idxmax() if tune_metric_take_max else agg[tuning_metric].idxmin()
    best_row = agg.loc[best_idx]
    return {param.replace("tun:", ""): best_row[param] for param in sweep_tune_params if param in best_row}


# main tuning logic

def tuning_flow_before_plot(
    logger,
    wandb_params,
    task_config_path: str,
    task_config_name: str,
    tuning_config_path: str,
    tuning_config_name: str,
    tuning_overrides: List[str],
    tuning_cache_refresh: bool,
    tuning_num_gpus: int,
    task_overrides: List[str],
    task_sweep: Dict[str, List[Any]],
    task_hp_overrides: List[str],
    task_hp_sweep_tune_basic_key: str,
    task_hp_sweep_tune_extra: Dict[str, List[Any]],
    task_hp_sweep_aggregate: Dict[str, List[Any]],
    task_hp_log_wandb: bool,
    task_run_overrides: List[str],
    task_run_sweep: Dict[str, List[Any]],
    task_visual_run_overrides_extra: List[str],
    task_visual_run_sweep: Dict[str, List[Any]],
    task_visual_cache_refresh: bool,
    task_visual_num_gpus: int,
    task_cache_refresh: bool,
    task_common_post_overrides: List[str],
    task_num_gpus: int,
    task_hp_budget_trials_per_param: Optional[int],
    task_hp_refine_budget_trials_per_param: Optional[int],
    task_to_tune: Task,
    task_to_tunes_inputs: Dict[str, Any],
    wandb_cfg: DictConfig,
    extra_combo_labels: List[Tuple[str, Any]] = [],
    task_sweep_stable_keys: Optional[List[str]] = None,
    task_run_post_overrides: Optional[List[str]] = None,
    task_visual_run_post_overrides: Optional[List[str]] = None,
):
    task_run_post_overrides = list(task_run_post_overrides or [])
    task_visual_run_post_overrides = list(task_visual_run_post_overrides or [])
    logger.info(
        "Starting tuning flow with config_path=%s config_name=%s",
        task_config_path, task_config_name,
    )

    # Partition sweep into tuning and HP-shared (stable) axes
    _tuning_sweep, _stable_sweep, _stable_rep_overrides = _split_stable_keys(
        task_sweep, task_sweep_stable_keys or []
    )
    if _stable_sweep:
        logging.info(
            "HP-sharing enabled: tuning over %s | stable keys with representative values: %s",
            list(_tuning_sweep.keys()), _stable_rep_overrides,
        )

    # Phase 1: HP tuning sweep
    tuning_task_result_list = []
    with locked_hydra_initialize(config_path=task_config_path, version_base="1.2"):
        for sweep_combo in get_hydra_sweep_combos(_tuning_sweep):
            tuning_task_result = []
            initial_cfg = compose(
                config_name=task_config_name,
                overrides=task_overrides + task_hp_overrides + get_hydra_overwrite_list(sweep_combo) + _stable_rep_overrides + task_common_post_overrides,
            )
            if task_hp_sweep_tune_basic_key in initial_cfg:
                task_hp_sweep_tune_basic = cast(Dict[str, List[Any]], OmegaConf.to_container(initial_cfg[task_hp_sweep_tune_basic_key], resolve=True))
            else:
                task_hp_sweep_tune_basic: Dict[str, List[Any]] = {}

            if isinstance(task_hp_sweep_tune_extra, (DictConfig,)):
                _extra = cast(Dict[str, Any], OmegaConf.to_container(task_hp_sweep_tune_extra, resolve=True))
            else:
                _extra = cast(Dict[str, Any], dict(task_hp_sweep_tune_extra))
            hp_sweep = task_hp_sweep_tune_basic | _extra
            hp_sweep_phase1 = _build_budgeted_tuning_hp_sweep(hp_sweep, task_hp_budget_trials_per_param, discretization_mode="log")

            logging.info(
                "Phase-1 HP sweep for combo %s: %d params, sweep=%s",
                sweep_combo, len(hp_sweep_phase1),
                {k: (len(v) if isinstance(v, list) else v) for k, v in hp_sweep_phase1.items()},
            )

            for sweep_combo_tuning in get_hydra_sweep_combos(hp_sweep_phase1):
                for sweep_combo_agg in get_hydra_sweep_combos(task_hp_sweep_aggregate):
                    cfg = compose(
                        config_name=task_config_name,
                        overrides=task_overrides + task_hp_overrides
                            + get_hydra_overwrite_list(sweep_combo)
                            + get_hydra_overwrite_list(sweep_combo_tuning)
                            + get_hydra_overwrite_list(sweep_combo_agg)
                            + _stable_rep_overrides
                            + task_common_post_overrides,
                    )
                    wandb_params_task = wandb_params.model_copy(update={
                        "name_aux": get_hydra_overwrite_str_short(sweep_combo) + "_" + get_hydra_overwrite_str_short(sweep_combo_tuning) + "_" + get_hydra_overwrite_str_short(sweep_combo_agg),
                        "config": flatten_hydra_config(cfg),
                        **cast(Dict[str, Any], wandb_cfg),
                    }).model_copy(update={"log": bool(task_hp_log_wandb)})

                    with remote_options(num_gpus=task_num_gpus):
                        tuning_task_result.append((
                            task_to_tune.with_options(refresh_cache=task_cache_refresh).submit(
                                **hydra_config_to_cacheable_dict(cfg),
                                wandb_params_task=wandb_params_task,
                                **task_to_tunes_inputs,
                            ),
                            [(f"tun:{x}", y) for x, y in sweep_combo_tuning] + [(f"agg:{x}", y) for x, y in sweep_combo_agg],
                        ))

            tuning_task_result_list.append({
                "tuning_result": tuning_task_result,
                "sweep_combo": sweep_combo,
                "hp_sweep_base": hp_sweep,
                "hp_sweep_phase1": hp_sweep_phase1,
            })

    # Phase 1: tuning selection
    tuning_hp_choice_overrides_stage1 = []
    tuning_cfg_for_refinement = None
    with locked_hydra_initialize(config_path=tuning_config_path, version_base="1.2"):
        tuning_cfg = compose(config_name=tuning_config_name, overrides=tuning_overrides)
        tuning_cfg_for_refinement = tuning_cfg

        for tuning_stage in tuning_task_result_list:
            tuning_result = cast(List[Tuple[PrefectFuture, List[Tuple[str, Any]]]], tuning_stage["tuning_result"])
            sweep_combo = cast(List[Tuple[str, Any]], tuning_stage["sweep_combo"])
            hp_sweep_base = cast(Dict[str, Any], tuning_stage["hp_sweep_base"])
            hp_sweep_phase1 = cast(Dict[str, Any], tuning_stage["hp_sweep_phase1"])

            tuning_result_futures = [r[0] for r in tuning_result]
            tuning_df_sweep = get_pandas_dataframe_from_sweep_combos([r[1] for r in tuning_result])

            _extra_prefix = (get_hydra_overwrite_str_short(extra_combo_labels) + "_") if extra_combo_labels else ""
            wandb_params_task = wandb_params.model_copy(update={
                "name_aux": _extra_prefix + get_hydra_overwrite_str_short(sweep_combo) + "_phase1",
                "config": flatten_hydra_config(tuning_cfg),
                **cast(Dict[str, Any], wandb_cfg),
            })

            with remote_options(num_gpus=tuning_num_gpus):
                tuning_hp_choice_overrides_stage1.append((
                    tuning_selection_task.with_options(refresh_cache=tuning_cache_refresh).submit(
                        results=tuning_result_futures,
                        df_sweep=tuning_df_sweep,
                        wandb_params_task=wandb_params_task,
                        tuning_phase_label="phase1",
                        **hydra_config_to_cacheable_dict(tuning_cfg),
                    ),
                    sweep_combo, hp_sweep_base, hp_sweep_phase1, tuning_result,
                ))

    # Phase 2 (optional): neighbourhood refinement
    tuning_task_result_list_phase2 = []
    if task_hp_refine_budget_trials_per_param is not None and task_hp_refine_budget_trials_per_param > 0:
        with locked_hydra_initialize(config_path=task_config_path, version_base="1.2"):
            for (ph1_overrides_future, sweep_combo, hp_sweep_base, hp_sweep_phase1, phase1_tuning_result) in tuning_hp_choice_overrides_stage1:
                phase1_overrides = ph1_overrides_future.result().cfg
                phase1_anchor_values = {}
                if tuning_cfg_for_refinement is not None:
                    phase1_anchor_values = _get_phase1_anchor_values_from_results(
                        tuning_result=cast(List[Tuple[PrefectFuture, List[Tuple[str, Any]]]], phase1_tuning_result),
                        tuning_cfg=tuning_cfg_for_refinement,
                    )
                hp_sweep_phase2 = _build_refined_tuning_hp_sweep(
                    hp_sweep_base=cast(Dict[str, Any], hp_sweep_base),
                    phase1_hp_sweep=cast(Dict[str, Any], hp_sweep_phase1),
                    phase1_best_overrides=cast(List[str], phase1_overrides),
                    task_hp_refine_budget_trials_per_param=task_hp_refine_budget_trials_per_param,
                    phase1_anchor_values=phase1_anchor_values,
                )
                logging.info("Phase-2 refinement sweep: %s", hp_sweep_phase2)

                tuning_task_result_phase2 = []
                for sweep_combo_tuning in get_hydra_sweep_combos(hp_sweep_phase2):
                    for sweep_combo_agg in get_hydra_sweep_combos(task_hp_sweep_aggregate):
                        cfg_phase2 = compose(
                            config_name=task_config_name,
                            overrides=task_overrides + task_hp_overrides
                                + get_hydra_overwrite_list(cast(List[Tuple[str, Any]], sweep_combo))
                                + get_hydra_overwrite_list(sweep_combo_tuning)
                                + get_hydra_overwrite_list(sweep_combo_agg)
                                + _stable_rep_overrides
                                + task_common_post_overrides,
                        )
                        wandb_params_task_p2 = wandb_params.model_copy(update={
                            "name_aux": get_hydra_overwrite_str_short(cast(List[Tuple[str, Any]], sweep_combo)) + "_phase2_" + get_hydra_overwrite_str_short(sweep_combo_tuning) + "_" + get_hydra_overwrite_str_short(sweep_combo_agg),
                            "config": flatten_hydra_config(cfg_phase2),
                            **cast(Dict[str, Any], wandb_cfg),
                        }).model_copy(update={"log": bool(task_hp_log_wandb)})

                        with remote_options(num_gpus=task_num_gpus):
                            tuning_task_result_phase2.append((
                                task_to_tune.with_options(refresh_cache=task_cache_refresh).submit(
                                    **hydra_config_to_cacheable_dict(cfg_phase2),
                                    wandb_params_task=wandb_params_task_p2,
                                    **task_to_tunes_inputs,
                                ),
                                [(f"tun:{x}", y) for x, y in sweep_combo_tuning] + [(f"agg:{x}", y) for x, y in sweep_combo_agg],
                            ))

                tuning_task_result_list_phase2.append((tuning_task_result_phase2, sweep_combo, phase1_tuning_result, phase1_overrides))

    # Final HP selection (phase-2 if available, else phase-1)
    if task_hp_refine_budget_trials_per_param is not None and task_hp_refine_budget_trials_per_param > 0:
        tuning_hp_choice_overrides_list = []
        with locked_hydra_initialize(config_path=tuning_config_path, version_base="1.2"):
            tuning_cfg = compose(config_name=tuning_config_name, overrides=tuning_overrides)
            for (p2_tuning_result, sweep_combo, ph1_tuning_result_ref, ph1_overrides_ref) in tuning_task_result_list_phase2:
                p2_futures = [r[0] for r in p2_tuning_result]
                p2_df_sweep = get_pandas_dataframe_from_sweep_combos([r[1] for r in p2_tuning_result])
                ph1_futures_ref = [r[0] for r in ph1_tuning_result_ref]
                ph1_df_ref = get_pandas_dataframe_from_sweep_combos([r[1] for r in ph1_tuning_result_ref])
                _extra_prefix = (get_hydra_overwrite_str_short(extra_combo_labels) + "_") if extra_combo_labels else ""
                wandb_params_task = wandb_params.model_copy(update={
                    "name_aux": _extra_prefix + get_hydra_overwrite_str_short(cast(List[Tuple[str, Any]], sweep_combo)) + "_phase2",
                    "config": flatten_hydra_config(tuning_cfg),
                    **cast(Dict[str, Any], wandb_cfg),
                })
                with remote_options(num_gpus=tuning_num_gpus):
                    tuning_hp_choice_overrides_list.append((
                        tuning_selection_task.with_options(refresh_cache=tuning_cache_refresh).submit(
                            results=p2_futures,
                            df_sweep=p2_df_sweep,
                            wandb_params_task=wandb_params_task,
                            tuning_phase_label="phase2",
                            phase1_reference_results=ph1_futures_ref,
                            phase1_reference_df_sweep=ph1_df_ref,
                            phase1_reference_best_overrides=ph1_overrides_ref,
                            **hydra_config_to_cacheable_dict(tuning_cfg),
                        ),
                        sweep_combo,
                    ))
    else:
        tuning_hp_choice_overrides_list = [
            (ph1_override, sweep_combo)
            for ph1_override, sweep_combo, _, _, _ in tuning_hp_choice_overrides_stage1
        ]

    # Final recon runs with best HPs
    hp_registry: Dict[Any, List[str]] = {}
    for (hp_choice_future, tuning_combo) in tuning_hp_choice_overrides_list:
        best_hp = hp_choice_future.result().cfg
        hp_key = tuple(sorted([(k, str(v)) for k, v in tuning_combo]))
        hp_registry[hp_key] = best_hp

    task_results = []
    visual_results = []
    with locked_hydra_initialize(config_path=task_config_path, version_base="1.2"):
        # Iterate over the full sweep (tuning x stable) so every combo gets a run.
        for sweep_combo in get_hydra_sweep_combos(task_sweep):
            # Identify the tuning-dimension portion and look up best HPs.
            tuning_portion = [(k, v) for k, v in sweep_combo if k in _tuning_sweep]
            hp_key = tuple(sorted([(k, str(v)) for k, v in tuning_portion]))
            best_hp_overrides = hp_registry.get(hp_key, [])
            if not best_hp_overrides and hp_registry:
                # Fallback: single-entry registry (all dims were stable)
                best_hp_overrides = next(iter(hp_registry.values()))
            logging.info("Final runs for sweep=%s with best HPs: %s", sweep_combo, best_hp_overrides)

            for sweep_combo_run in get_hydra_sweep_combos(task_run_sweep):
                cfg = compose(
                    config_name=task_config_name,
                    overrides=task_overrides + task_run_overrides
                        + get_hydra_overwrite_list(sweep_combo)
                        + get_hydra_overwrite_list(sweep_combo_run)
                        + task_common_post_overrides
                        + best_hp_overrides
                        + task_run_post_overrides,
                )
                wandb_params_task = wandb_params.model_copy(update={
                    "name_aux": get_hydra_overwrite_str_short(sweep_combo) + "_" + get_hydra_overwrite_str_short(sweep_combo_run),
                    "config": flatten_hydra_config(cfg),
                    **cast(Dict[str, Any], wandb_cfg),
                })
                with remote_options(num_gpus=task_num_gpus):
                    task_results.append((
                        task_to_tune.with_options(refresh_cache=task_cache_refresh).submit(
                            **hydra_config_to_cacheable_dict(cfg),
                            wandb_params_task=wandb_params_task,
                            **task_to_tunes_inputs,
                        ),
                        extra_combo_labels + sweep_combo + sweep_combo_run,
                    ))

            for sweep_combo_visual in get_hydra_sweep_combos(task_visual_run_sweep):
                visual_cfg = compose(
                    config_name=task_config_name,
                    overrides=task_overrides + task_run_overrides + task_visual_run_overrides_extra
                        + get_hydra_overwrite_list(sweep_combo)
                        + get_hydra_overwrite_list(sweep_combo_visual)
                        + task_common_post_overrides
                        + best_hp_overrides
                        + task_visual_run_post_overrides,
                )
                visual_wandb_params_task = wandb_params.model_copy(update={
                    "name_aux": get_hydra_overwrite_str_short(sweep_combo) + "_visual_" + get_hydra_overwrite_str_short(sweep_combo_visual),
                    "config": flatten_hydra_config(visual_cfg),
                    **cast(Dict[str, Any], wandb_cfg),
                })
                with remote_options(num_gpus=task_visual_num_gpus):
                    visual_results.append((
                        task_to_tune.with_options(refresh_cache=task_visual_cache_refresh).submit(
                            **hydra_config_to_cacheable_dict(visual_cfg),
                            wandb_params_task=visual_wandb_params_task,
                            **task_to_tunes_inputs,
                        ),
                        extra_combo_labels + sweep_combo + sweep_combo_visual,
                    ))

    return task_results, visual_results, hp_registry


def tuning_flow(
    tuning_configs: List[TuningConfig],
    wandb_cfg: DictConfig,
    task_to_tune: List[Task],
    wandb_task_params,
    task_to_tunes_inputs: List[Dict[str, Any]],
    logger,
    extra_combo_labels: List[Tuple[str, Any]] = [],
) -> Tuple[
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
]:
    assert len(task_to_tune) == len(tuning_configs)

    all_task_results = []
    all_visual_results = []
    all_hp_registry: Dict[Any, List[str]] = {}

    for tuning_config, task, task_inputs in zip(tuning_configs, task_to_tune, task_to_tunes_inputs):
        task_results, visual_results, hp_registry = tuning_flow_before_plot(
            **asdict(tuning_config),
            task_to_tune=task,
            wandb_cfg=wandb_cfg,
            logger=logger,
            wandb_params=wandb_task_params,
            task_to_tunes_inputs=task_inputs,
            extra_combo_labels=extra_combo_labels,
        )
        all_task_results.extend(task_results)
        all_visual_results.extend(visual_results)
        all_hp_registry.update(hp_registry)

    return all_task_results, all_visual_results, all_hp_registry


# recon_run_only (HP transfer)

@task
def recon_run_only(
    recon_recon_config_path: str,
    recon_recon_config_name: str,
    common_recon_overrides: List[str],
    common_post_overrides: List[str],
    recon_recon_sweep: Dict[str, List[Any]],
    recon_recon_run_overrides: List[str],
    common_run_sweep: Dict[str, List[Any]],
    common_visual_run_sweep: Optional[Dict[str, List[Any]]],
    common_visual_recon_overrides_extra: Optional[List[str]],
    common_visual_recon_cache_refresh: bool,
    common_visual_recon_num_gpus: int,
    recon_recon_cache_refresh: bool,
    recon_recon_num_gpus: int,
    hp_registry: Dict[Any, List[str]],
    recon_stable_keys: List[str],
    wandb_cfg: DictConfig,
    logger,
    wandb_params,
    local_cache_path: Optional[str] = None,
    bart_path: Optional[str] = None,
    task_to_tune_recon_input: Dict[str, Any] = {},
    extra_combo_labels: List[Tuple[str, Any]] = [],
    recon_recon_run_post_overrides: Optional[List[str]] = None,
    common_visual_recon_post_overrides: Optional[List[str]] = None,
) -> Tuple[
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
]:
    """Run final recon directly (no tuning) by transferring HPs from *hp_registry*."""
    from src.tasks.recon_task import recon_task

    task_to_tune_recon_input = dict(task_to_tune_recon_input)
    if local_cache_path is not None:
        task_to_tune_recon_input.setdefault("local_cache_path", local_cache_path)

    visual_run_overrides_extra = list(common_visual_recon_overrides_extra or [])
    visual_run_sweep = common_visual_run_sweep or {}
    run_post_overrides = list(recon_recon_run_post_overrides or [])
    visual_post_overrides = list(common_visual_recon_post_overrides or [])

    # Determine which keys were tuned (non-stable) during the original run
    _tuning_sweep, _, _ = _split_stable_keys(recon_recon_sweep, recon_stable_keys)

    task_results: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] = []
    visual_results: List[Tuple[PrefectFuture, List[Tuple[str, Any]]]] = []
    with locked_hydra_initialize(config_path=recon_recon_config_path, version_base="1.2"):
        for sweep_combo in get_hydra_sweep_combos(recon_recon_sweep):
            tuning_portion = [(k, v) for k, v in sweep_combo if k in _tuning_sweep]
            hp_key = tuple(sorted([(k, str(v)) for k, v in tuning_portion]))
            best_hp_overrides: List[str] = hp_registry.get(hp_key, [])
            if not best_hp_overrides and hp_registry:
                best_hp_overrides = next(iter(hp_registry.values()))
            logging.info("(HP-transfer) runs for sweep=%s best_hp=%s", sweep_combo, best_hp_overrides)

            _extra_prefix = (get_hydra_overwrite_str_short(extra_combo_labels) + "_") if extra_combo_labels else ""

            for sweep_combo_run in get_hydra_sweep_combos(common_run_sweep):
                cfg = compose(
                    config_name=recon_recon_config_name,
                    overrides=common_recon_overrides + recon_recon_run_overrides
                        + get_hydra_overwrite_list(sweep_combo)
                        + get_hydra_overwrite_list(sweep_combo_run)
                        + common_post_overrides
                        + best_hp_overrides
                        + run_post_overrides,
                )
                wandb_params_task = wandb_params.model_copy(update={
                    "name_aux": _extra_prefix + get_hydra_overwrite_str_short(sweep_combo) + "_" + get_hydra_overwrite_str_short(sweep_combo_run) + "_hptransfer",
                    "config": flatten_hydra_config(cfg),
                    **cast(Dict[str, Any], wandb_cfg),
                })
                with remote_options(num_gpus=recon_recon_num_gpus):
                    task_results.append((
                        recon_task.with_options(refresh_cache=recon_recon_cache_refresh).submit(
                            **hydra_config_to_cacheable_dict(cfg),
                            wandb_params_task=wandb_params_task,
                            **task_to_tune_recon_input,
                        ),
                        extra_combo_labels + sweep_combo + sweep_combo_run,
                    ))

            for sweep_combo_visual in get_hydra_sweep_combos(visual_run_sweep):
                visual_cfg = compose(
                    config_name=recon_recon_config_name,
                    overrides=common_recon_overrides + recon_recon_run_overrides + visual_run_overrides_extra
                        + get_hydra_overwrite_list(sweep_combo)
                        + get_hydra_overwrite_list(sweep_combo_visual)
                        + common_post_overrides
                        + best_hp_overrides
                        + visual_post_overrides,
                )
                visual_wandb_params_task = wandb_params.model_copy(update={
                    "name_aux": _extra_prefix + get_hydra_overwrite_str_short(sweep_combo) + "_visual_" + get_hydra_overwrite_str_short(sweep_combo_visual) + "_hptransfer",
                    "config": flatten_hydra_config(visual_cfg),
                    **cast(Dict[str, Any], wandb_cfg),
                })
                with remote_options(num_gpus=common_visual_recon_num_gpus):
                    visual_results.append((
                        recon_task.with_options(refresh_cache=common_visual_recon_cache_refresh).submit(
                            **hydra_config_to_cacheable_dict(visual_cfg),
                            wandb_params_task=visual_wandb_params_task,
                            **task_to_tune_recon_input,
                        ),
                        extra_combo_labels + sweep_combo + sweep_combo_visual,
                    ))

    return task_results, visual_results


# convenience wrapper (no baselines)

@task
def tune_recon_only(
    recon_recon_config_path: str,
    recon_recon_config_name: str,
    recon_tuning_config_path: str,
    recon_tuning_config_name: str,
    recon_tuning_overrides: List[str],
    recon_tuning_cache_refresh: bool,
    recon_tuning_num_gpus: int,
    common_hp_sweep_aggregate: Dict[str, List[Any]],
    common_run_sweep: Dict[str, List[Any]],
    common_recon_overrides: List[str],
    common_post_overrides: List[str],
    recon_recon_sweep: Dict[str, List[Any]],
    recon_recon_hp_overrides: List[str],
    recon_recon_hp_sweep_tune_basic_key: str,
    recon_recon_hp_sweep_tune_extra: Dict[str, List[Any]],
    recon_recon_hp_log_wandb: bool,
    recon_recon_run_overrides: List[str],
    recon_recon_cache_refresh: bool,
    recon_recon_num_gpus: int,
    recon_recon_hp_budget_trials_per_param: Optional[int],
    recon_recon_hp_refine_budget_trials_per_param: Optional[int],
    common_visual_run_sweep: Optional[Dict[str, List[Any]]],
    common_visual_recon_overrides_extra: Optional[List[str]],
    common_visual_recon_cache_refresh: bool,
    common_visual_recon_num_gpus: int,
    wandb_cfg: DictConfig,
    logger,
    wandb_params,
    local_cache_path: Optional[str] = None,
    bart_path: Optional[str] = None,
    task_to_tune_recon_input: Dict[str, Any] = {},
    extra_combo_labels: List[Tuple[str, Any]] = [],
    recon_stable_keys: Optional[List[str]] = None,
    recon_recon_run_post_overrides: Optional[List[str]] = None,
    common_visual_recon_post_overrides: Optional[List[str]] = None,
) -> Tuple[
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    List[Tuple[PrefectFuture, List[Tuple[str, Any]]]],
    Dict[Any, List[str]],
]:
    from src.tasks.recon_task import recon_task

    task_to_tune_recon_input = dict(task_to_tune_recon_input)
    if local_cache_path is not None:
        task_to_tune_recon_input.setdefault("local_cache_path", local_cache_path)

    visual_run_overrides_extra = list(common_visual_recon_overrides_extra or [])
    visual_run_sweep = common_visual_run_sweep or {}

    recon_tuning_config = TuningConfig(
        task_config_path=recon_recon_config_path,
        task_config_name=recon_recon_config_name,
        tuning_config_path=recon_tuning_config_path,
        tuning_config_name=recon_tuning_config_name,
        tuning_overrides=recon_tuning_overrides,
        tuning_cache_refresh=recon_tuning_cache_refresh,
        tuning_num_gpus=recon_tuning_num_gpus,
        task_overrides=common_recon_overrides,
        task_sweep=recon_recon_sweep,
        task_hp_overrides=recon_recon_hp_overrides,
        task_hp_sweep_tune_basic_key=recon_recon_hp_sweep_tune_basic_key,
        task_hp_sweep_tune_extra=recon_recon_hp_sweep_tune_extra,
        task_hp_sweep_aggregate=common_hp_sweep_aggregate,
        task_hp_log_wandb=recon_recon_hp_log_wandb,
        task_run_overrides=recon_recon_run_overrides,
        task_run_sweep=common_run_sweep,
        task_visual_run_overrides_extra=visual_run_overrides_extra,
        task_visual_run_sweep=visual_run_sweep,
        task_visual_cache_refresh=common_visual_recon_cache_refresh,
        task_visual_num_gpus=common_visual_recon_num_gpus,
        task_cache_refresh=recon_recon_cache_refresh,
        task_common_post_overrides=common_post_overrides,
        task_num_gpus=recon_recon_num_gpus,
        task_hp_budget_trials_per_param=recon_recon_hp_budget_trials_per_param,
        task_hp_refine_budget_trials_per_param=recon_recon_hp_refine_budget_trials_per_param,
        task_sweep_stable_keys=recon_stable_keys or [],
        task_run_post_overrides=list(recon_recon_run_post_overrides or []),
        task_visual_run_post_overrides=list(common_visual_recon_post_overrides or []),
    )

    return tuning_flow(
        tuning_configs=[recon_tuning_config],
        wandb_cfg=wandb_cfg,
        task_to_tune=[recon_task],
        wandb_task_params=wandb_params,
        task_to_tunes_inputs=[task_to_tune_recon_input],
        logger=logger,
        extra_combo_labels=extra_combo_labels,
    )
