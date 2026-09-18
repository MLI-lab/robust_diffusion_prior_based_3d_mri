from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def _extend(target: Dict[str, Any], path: list[str], values: Any) -> None:
    if values is None:
        return
    node = target
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = list(node.get(path[-1], [])) + list(values)


def _replace(target: Dict[str, Any], path: list[str], value: Any) -> None:
    if value is None:
        return
    node = target
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = deepcopy(value)


def apply_smoke_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(cfg)
    if not (result.get("smoke", {}).get("enabled") or result.get("protocol", {}).get("smoke_test")):
        return result

    overrides = result.get("smoke", {}).get("overrides", {})
    data = overrides.get("data", {})
    recon = overrides.get("reconstruction", {})
    reporting = overrides.get("reporting", {})
    holdout = overrides.get("holdout", {})

    _extend(result, ["data", "source", "download", "overrides"], data.get("download_overrides"))
    _extend(result, ["data", "source", "conversion", "shared", "overrides"], data.get("conversion_overrides"))
    _replace(result, ["data", "source", "conversion", "shared", "sweep"], data.get("conversion_sweep"))

    for split in ("train", "recon"):
        conv = result["data"]["source"]["conversion"].get(split)
        if conv is not None:
            _extend(result, ["data", "source", "conversion", split, "overrides"], data.get(f"{split}_conversion_overrides") or data.get("conversion_overrides"))
            _replace(result, ["data", "source", "conversion", split, "sweep"], data.get(f"{split}_conversion_sweep") or data.get("conversion_sweep"))

    _extend(result, ["data", "preprocess", "train", "overrides"], data.get("preprocess_train_overrides"))
    _extend(result, ["data", "preprocess", "recon", "overrides"], data.get("preprocess_recon_overrides"))
    _replace(result, ["data", "preprocess", "recon", "sweep"], data.get("preprocess_recon_sweep"))

    _extend(result, ["model", "train", "post_overrides"], overrides.get("model", {}).get("train_post_overrides"))
    _extend(result, ["reconstruction", "post_overrides"], recon.get("post_overrides"))
    _replace(result, ["reconstruction", "method_sweep"], recon.get("method_sweep"))
    _replace(result, ["reconstruction", "tuning", "aggregate_sweep"], recon.get("aggregate_sweep"))
    _replace(result, ["reconstruction", "evaluation", "sweep"], recon.get("evaluation_sweep"))
    _replace(result, ["reconstruction", "visual_evaluation", "sweep"], recon.get("visual_evaluation_sweep"))
    _replace(result, ["reconstruction", "tuning", "coarse_budget_trials_per_param"], recon.get("coarse_budget_trials_per_param"))
    _replace(result, ["reconstruction", "tuning", "refine_budget_trials_per_param"], recon.get("refine_budget_trials_per_param"))

    _replace(result, ["reporting", "visual_plot", "enabled"], reporting.get("visual_plot_enabled"))
    _replace(result, ["reporting", "aggregate_plot", "enabled"], reporting.get("aggregate_plot_enabled"))
    _replace(result, ["holdout", "index", "values"], holdout.get("index_values"))
    _replace(result, ["holdout", "label", "values"], holdout.get("label_values"))
    return result

