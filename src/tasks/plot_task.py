from prefect import flow, task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE
import logging
import sys
import wandb
from src.prefect.wandb_lock import locked_wandb_init
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
from typing import Any, List, Optional, Dict, Sequence
from src.prefect.caching import CacheableDict, CacheableDictConfig, CacheableListConfig
import pandas as pd
import seaborn as sns
from typing import Tuple
import io
import math
import numpy as np
from itertools import permutations as _permutations
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from scipy import stats as scipy_stats

import matplotlib.pyplot as plt


# helper utilities

def _to_jsonable_scalar(value):
    if isinstance(value, pd.Series):
        return ", ".join(str(v) for v in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple, set, dict)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _sanitize_table_df(df: pd.DataFrame) -> pd.DataFrame:
    sanitized = df.copy()
    for col in sanitized.columns:
        sanitized[col] = sanitized[col].map(_to_jsonable_scalar)
    return sanitized


def _normalize_param_name(param_name: str) -> str:
    return str(param_name).strip().strip('"').strip("'").replace("+", "")


def _resolve_column_name(requested_name: str, available_columns: List[str]) -> Optional[str]:
    if requested_name in available_columns:
        return requested_name
    normalized_requested = _normalize_param_name(requested_name)
    normalized_to_actual = {_normalize_param_name(col): col for col in available_columns}
    exact_match = normalized_to_actual.get(normalized_requested)
    if exact_match is not None:
        return exact_match

    suffix_matches = [
        col for col in available_columns
        if _normalize_param_name(col).endswith(f".{normalized_requested}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def _filter_metrics_by_cfg(
    metrics: List[str],
    metrics_filter_cfg: Optional[CacheableListConfig],
    warning_context: str,
) -> List[str]:
    if metrics_filter_cfg is None:
        return metrics
    metric_filters = [str(v).strip().lower() for v in metrics_filter_cfg.cfg if str(v).strip()]
    if not metric_filters:
        return metrics
    filtered = [m for m in metrics if any(f in m.lower() for f in metric_filters)]
    if not filtered:
        logging.warning(
            "No metrics matched filters %s for %s. Falling back to all numeric metrics.",
            metric_filters, warning_context,
        )
        return metrics
    return filtered


def _render_table_figure(
    table_df: pd.DataFrame,
    title: str,
    figsize: Sequence[float],
    dpi: int,
) -> "plt.Figure":
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    table = ax.table(
        cellText=table_df.astype(str).values,
        colLabels=[str(col) for col in table_df.columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def _build_summary_table(
    df_plot: pd.DataFrame,
    group_cols: List[str],
    numeric_metrics: List[str],
    significance_lookup: Optional[Dict[Tuple[Any, str], bool]] = None,
) -> pd.DataFrame:
    if group_cols:
        mean_df = df_plot.groupby(group_cols, dropna=False)[numeric_metrics].mean().reset_index()
        std_df = df_plot.groupby(group_cols, dropna=False)[numeric_metrics].std().reset_index()
        std_df = std_df.rename(columns={m: f"{m}__std" for m in numeric_metrics})
        agg = mean_df.merge(std_df, on=group_cols, how="left")
    else:
        agg = pd.DataFrame({"setting": ["all"]})
        for m in numeric_metrics:
            agg[m] = [df_plot[m].mean()]
            agg[f"{m}__std"] = [df_plot[m].std()]
        group_cols = ["setting"]

    rows = []
    for _, row in agg.iterrows():
        # Raw (pre-sanitized) group values double as the significance-lookup key, so they must
        # match the dtypes produced by _compute_stat_comparisons (which also groups on raw df_plot values).
        significance_key = tuple(row[col] for col in group_cols)
        row_dict = {col: _to_jsonable_scalar(row[col]) for col in group_cols}
        for m in numeric_metrics:
            mean_val = row[m]
            std_val = row[f"{m}__std"]
            cell = f"{mean_val:.4g} ± {0.0 if pd.isna(std_val) else std_val:.3g}"
            if significance_lookup is not None and significance_lookup.get((significance_key, m)):
                cell += " *"
            row_dict[m] = cell
        rows.append(row_dict)

    return _sanitize_table_df(pd.DataFrame(rows, columns=group_cols + numeric_metrics))


def _compute_stat_comparisons(
    df_plot: pd.DataFrame,
    summary_group_cols: List[str],
    stochastic_params: List[str],
    baseline_col: str,
    baseline_value: Any,
    numeric_metrics: List[str],
    lower_is_better_metrics: List[str],
    alpha: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[Any, str], bool]]:
    """Compares every non-baseline value of `baseline_col` against `baseline_value`, pairing rows on (summary_group_cols minus baseline_col) + stochastic_params (e.g. per ++sample_idx), and computes per-metric win-rate, paired sample count, two-sided Wilcoxon signed-rank p, and mean paired diff (variant - baseline).
    """
    group_cols = [c for c in summary_group_cols if c != baseline_col]
    merge_keys = group_cols + stochastic_params
    lower_is_better_metrics = [m.lower() for m in lower_is_better_metrics]

    base_rows = df_plot[df_plot[baseline_col] == baseline_value]
    other_values = [v for v in df_plot[baseline_col].dropna().unique().tolist() if v != baseline_value]

    records: List[Dict[str, Any]] = []
    significance_lookup: Dict[Tuple[Any, str], bool] = {}

    for other_value in other_values:
        other_rows = df_plot[df_plot[baseline_col] == other_value]
        if merge_keys:
            paired_all = base_rows.merge(
                other_rows, on=merge_keys, how="inner", suffixes=("__baseline", "__variant")
            )
        else:
            n_pairs = min(len(base_rows), len(other_rows))
            paired_all = pd.concat(
                [
                    base_rows.iloc[:n_pairs].reset_index(drop=True).add_suffix("__baseline"),
                    other_rows.iloc[:n_pairs].reset_index(drop=True).add_suffix("__variant"),
                ],
                axis=1,
            )
        if paired_all.empty:
            continue

        grouped_iter = paired_all.groupby(group_cols, dropna=False) if group_cols else [((), paired_all)]
        for group_key, sub in grouped_iter:
            group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
            value_map = dict(zip(group_cols, group_key_tuple))
            value_map[baseline_col] = other_value
            summary_key = tuple(value_map[c] for c in summary_group_cols)

            for metric in numeric_metrics:
                base_col, var_col = f"{metric}__baseline", f"{metric}__variant"
                if base_col not in sub.columns or var_col not in sub.columns:
                    continue
                base_vals = pd.to_numeric(sub[base_col], errors="coerce").to_numpy(dtype=float)
                var_vals = pd.to_numeric(sub[var_col], errors="coerce").to_numpy(dtype=float)
                valid = ~np.isnan(base_vals) & ~np.isnan(var_vals)
                base_vals, var_vals = base_vals[valid], var_vals[valid]
                n = int(len(base_vals))
                diffs = var_vals - base_vals

                if n == 0:
                    win_rate, mean_diff, p_value = float("nan"), float("nan"), float("nan")
                else:
                    lower_better = any(f in metric.lower() for f in lower_is_better_metrics)
                    wins = diffs < 0 if lower_better else diffs > 0
                    win_rate = float(np.mean(wins))
                    mean_diff = float(np.mean(diffs))
                    if n >= 2 and np.any(diffs != 0):
                        try:
                            p_value = float(scipy_stats.wilcoxon(diffs, alternative="two-sided").pvalue)
                        except ValueError:
                            p_value = float("nan")
                    else:
                        p_value = float("nan")

                significant = bool(n > 0 and not math.isnan(p_value) and p_value < alpha)
                significance_lookup[(summary_key, metric)] = significant

                row = dict(value_map)
                row["baseline"] = baseline_value
                row["metric"] = metric
                row["n"] = n
                row["win_rate"] = win_rate
                row["wilcoxon_p"] = p_value
                row["mean_diff"] = mean_diff
                records.append(row)

    columns = group_cols + [baseline_col, "baseline", "metric", "n", "win_rate", "wilcoxon_p", "mean_diff"]
    stat_df = pd.DataFrame(records, columns=columns)
    return stat_df, significance_lookup


def _format_stat_table_for_display(stat_df: pd.DataFrame) -> pd.DataFrame:
    display_df = stat_df.copy()
    display_df["win_rate"] = display_df["win_rate"].map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%")
    display_df["wilcoxon_p"] = display_df["wilcoxon_p"].map(lambda v: "" if pd.isna(v) else f"{v:.3g}")
    display_df["mean_diff"] = display_df["mean_diff"].map(lambda v: "" if pd.isna(v) else f"{v:+.4g}")
    return _sanitize_table_df(display_df)



def _cfg_to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, CacheableListConfig):
        return OmegaConf.to_container(value.cfg, resolve=True)
    if isinstance(value, CacheableDictConfig):
        return OmegaConf.to_container(value.cfg, resolve=True)
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _cfg_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _split_cfg_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _parse_pivot_spec(spec: Any) -> Dict[str, Any]:
    spec = _cfg_to_plain(spec)
    if isinstance(spec, str):
        parsed: Dict[str, Any] = {}
        for part in spec.split(";"):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"Invalid pivot table segment '{part}'. Expected key=value.")
            key, value = part.split("=", 1)
            parsed[key.strip()] = value.strip()
        spec = parsed
    if not isinstance(spec, dict):
        raise ValueError(f"Pivot table spec must be a dict or key=value string, got {type(spec)}.")

    metrics = _split_cfg_list(spec.get("metrics", spec.get("metric")))
    rows = _split_cfg_list(spec.get("rows", spec.get("index")))
    cols = _split_cfg_list(spec.get("cols", spec.get("columns")))
    agg = _split_cfg_list(spec.get("agg", spec.get("aggs", "mean")))
    include_std = _cfg_bool(spec.get("include_std", spec.get("std", False)))
    if include_std and "std" not in agg:
        agg.append("std")
    title = str(spec.get("title", "Pivot Table"))
    name = str(spec.get("name", title.lower().replace(" ", "_").replace("/", "_")))
    decimals = int(spec.get("decimals", 3))

    if not metrics:
        raise ValueError("Pivot table spec needs at least one metric.")
    if not rows:
        raise ValueError("Pivot table spec needs at least one row/index column.")
    if not cols:
        raise ValueError("Pivot table spec needs at least one column axis.")
    return {
        "metrics": metrics,
        "rows": rows,
        "cols": cols,
        "agg": agg,
        "title": title,
        "name": name,
        "decimals": decimals,
    }


def _resolve_required_columns(requested: List[str], available_columns: List[str], context: str) -> List[str]:
    resolved = []
    missing = []
    for item in requested:
        column = _resolve_column_name(item, available_columns)
        if column is None:
            missing.append(item)
        else:
            resolved.append(column)
    if missing:
        raise ValueError(f"Pivot table {context} columns not found: {missing}. Available columns: {available_columns}")
    return resolved


def _format_pivot_value(value: Any, decimals: int) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def _label_from_key(key: Any) -> str:
    if isinstance(key, tuple):
        return " / ".join(str(v) for v in key)
    return str(key)


def _build_pivot_table(
    df_plot: pd.DataFrame,
    spec: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    available = df_plot.columns.tolist()
    row_cols = _resolve_required_columns(spec["rows"], available, "row")
    col_cols = _resolve_required_columns(spec["cols"], available, "column")
    metrics = _resolve_required_columns(spec["metrics"], available, "metric")

    pivot = pd.pivot_table(
        df_plot,
        values=metrics,
        index=row_cols,
        columns=col_cols,
        aggfunc=spec["agg"],
        dropna=False,
        sort=True,
    )
    if isinstance(pivot, pd.Series):
        pivot = pivot.to_frame()

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    if isinstance(pivot.columns, pd.MultiIndex):
        normalized_columns = []
        for col in pivot.columns.tolist():
            parts = tuple(col)
            if parts and str(parts[0]) in spec["agg"]:
                stat = str(parts[0])
                metric = str(parts[1]) if len(parts) > 1 else metrics[0]
                group_parts = parts[2:]
            else:
                stat = spec["agg"][0]
                metric = str(parts[0]) if parts else metrics[0]
                group_parts = parts[1:]
            group = group_parts[0] if len(group_parts) == 1 else tuple(group_parts)
            normalized_columns.append((group, metric, stat))
        pivot.columns = pd.MultiIndex.from_tuples(
            normalized_columns,
            names=["column", "metric", "stat"],
        )
        pivot = pivot.sort_index(axis=1, level=[0, 1, 2])
    else:
        pivot.columns = pd.MultiIndex.from_tuples(
            [(col, metrics[0], spec["agg"][0]) for col in pivot.columns],
            names=["column", "metric", "stat"],
        )

    meta = {
        "row_cols": row_cols,
        "col_cols": col_cols,
        "metrics": metrics,
        "agg": spec["agg"],
    }
    return pivot, meta


def _flatten_pivot_for_wandb(pivot: pd.DataFrame) -> pd.DataFrame:
    flat = pivot.reset_index()
    flat.columns = [
        " / ".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in flat.columns
    ]
    return _sanitize_table_df(flat)


def _render_pivot_table_figure(
    pivot: pd.DataFrame,
    meta: Dict[str, Any],
    title: str,
    figsize: Sequence[float],
    dpi: int,
    decimals: int,
) -> "plt.Figure":
    row_labels = [_label_from_key(idx) for idx in pivot.index.tolist()]

    columns = []
    for col in pivot.columns.tolist():
        parts = col if isinstance(col, tuple) else (col,)
        parts = tuple(parts)
        if len(parts) == 3:
            col_value, metric, stat = parts
        elif len(parts) == 2:
            col_value, metric = parts
            stat = meta["agg"][0]
        else:
            col_value = parts[0]
            metric = meta["metrics"][0]
            stat = meta["agg"][0]
        columns.append({
            "group": _label_from_key(col_value),
            "metric": str(metric),
            "stat": str(stat),
        })

    stat_enabled = len(set(c["stat"] for c in columns)) > 1
    header_rows = 3 if stat_enabled else 2
    n_rows = len(row_labels) + header_rows
    n_cols = len(columns) + 1
    fig_width = max(float(figsize[0]), 1.2 + 1.25 * n_cols)
    fig_height = max(float(figsize[1]), 1.2 + 0.34 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.axis("off")

    data = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    data[0][0] = " / ".join(meta["row_cols"])
    for col_idx, info in enumerate(columns, start=1):
        data[1][col_idx] = info["metric"]
        if stat_enabled:
            data[2][col_idx] = info["stat"]
    body_start = header_rows
    for row_idx, row_label in enumerate(row_labels, start=body_start):
        data[row_idx][0] = row_label
        for col_idx, value in enumerate(pivot.iloc[row_idx - body_start].tolist(), start=1):
            data[row_idx][col_idx] = _format_pivot_value(value, decimals)

    table = ax.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.28)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#8a8a8a")
        cell.set_linewidth(0.35)
        if row < header_rows:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f4f4f4")
        if col == 0:
            cell.set_text_props(ha="left", weight="bold" if row < header_rows else "normal")

    group_start = 1
    while group_start <= len(columns):
        group = columns[group_start - 1]["group"]
        group_end = group_start
        while group_end <= len(columns) and columns[group_end - 1]["group"] == group:
            group_end += 1
        center = (group_start + group_end - 1) // 2
        for col_idx in range(group_start, group_end):
            cell = table[(0, col_idx)]
            cell.get_text().set_text(group if col_idx == center else "")
            cell.set_linewidth(0.8 if col_idx in (group_start, group_end - 1) else 0.2)
        for row_idx in range(n_rows):
            table[(row_idx, group_start)].set_linewidth(0.8)
            table[(row_idx, group_end - 1)].set_linewidth(0.8)
        group_start = group_end

    ax.set_title(title, pad=12)
    plt.tight_layout()
    return fig

"""
This task plots the results of a multistage sweep, where each stage has its own set of sweep combinations.
So the structure is that we have L stages, where each stage has n_i sweep combinations.

    Parameters:
        - metric: The metric to plot, e.g. "loss", "accuracy", should must be an accessible key in the CacheableDict.
        - run_results: A list of CacheableDict objects,

    We assume that the run_results is a list of CacheableDict, from which we can extract a certain parameter.

    Examples:
        - L=1, n_1=5 (single metric) -> plot ser over number of iterations
        - L=2, n_1=5, n_2=3: -> recon


    Basically, throughout the multiple stages we can consider the final results as a list of results, each having one sweep combo, combined throughout the stages. For L=2 this leads to a sweep combo depth of 2, where we can pick one as x axis and the other via color grading.
    For all other sweep combo depths it is not clear yet how we can represent it (maybe taking averages or something like that). 
"""

@task(cache_policy=None, name="Seaborn plot", tags=["plot"], version="1.0", retries=0, description="Plot the results of a multistage sweep.")
def seaborn_plot_sweep_results_auto(
        # df_results : pd.DataFrame,
        results : List[CacheableDict], # or rather PrefectFuture
        df_sweep : pd.DataFrame,
        wandb_params_task: WandbParamsTask,
        palette: str,
        figsize_cfg: CacheableListConfig,
        type : str,
        stochastic_params_cfg: Optional[CacheableListConfig] = None,
        paper_summary_table_enabled: bool = False,
        paper_summary_metrics_filter_cfg: Optional[CacheableListConfig] = None,
        paper_stat_table_enabled: bool = False,
        paper_stat_baseline_param: Optional[str] = None,
        paper_stat_baseline_value: Optional[str] = None,
        paper_stat_alpha: float = 0.05,
        paper_stat_lower_is_better_metrics_cfg: Optional[CacheableListConfig] = None,
        pivot_tables_cfg: Optional[CacheableListConfig] = None,
        tradeoff_plots_enabled: bool = False,
        tradeoff_same_color_lines_enabled: bool = False,
        tradeoff_errorbars_enabled: bool = False,
        log_lin_scale_auto_enabled : bool = True,
        log_lin_scale_threshold : float = 2.0,
        dpi : int = 200
    ) -> None:

    """
        Parameters:
            - df_results: A pandas DataFrame containing the results of the sweep.
            - df_sweep: A pandas DataFrame containing the sweep combinations.
            - wandb_params_task: The WandbParamsTask object containing the parameters for the wandb run.
            - stochastic_params_cfg: Parameters to treat as stochastic (aggregated over when building the summary table).
            - paper_summary_table_enabled: Whether to upload a summary table (mean ± std) image to W&B.
            - paper_summary_metrics_filter_cfg: If given, only these metric substrings are included in the summary table.
            - paper_stat_table_enabled: Whether to compute/upload a statistical comparison table (win-rate, n,
              two-sided paired Wilcoxon p, mean per-example diff) of every setting vs. a baseline setting, and
              annotate significant cells (p < paper_stat_alpha) in the paper_summary_table with a "*".
            - paper_stat_baseline_param: Sweep parameter identifying the baseline axis, e.g. "+rec_method".
            - paper_stat_baseline_value: Value of paper_stat_baseline_param identifying the baseline setting.
            - paper_stat_alpha: Two-sided significance threshold for the Wilcoxon test.
            - paper_stat_lower_is_better_metrics_cfg: Metric name substrings for which a lower value is a "win"
              (default: higher is better).
            - pivot_tables_cfg: Optional list of pivot specs. Each spec may be a dict or key=value string.

        Returns:
            None, but logs the plot to wandb.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    df_results = pd.DataFrame([result.cfg for result in results])

    with locked_wandb_init(**wandb_kwargs_for_prefect_task(wandb_params_task)):

        # First we log the dataframes to wandb

        # Identify numeric metrics in results
        numeric_metrics = df_results.select_dtypes(include=['number']).columns.tolist()
        print(f"Ignoring the following result params: {df_results.columns.difference(numeric_metrics).tolist()}")

        # Identify sweep parameters (numeric or categorical)
        sweep_params = df_sweep.columns.tolist()

        # aggregate the results and sweep combinations
        df_plot = pd.concat([df_results, df_sweep], axis=1)
        wandb.log({"dataframe": wandb.Table(dataframe=df_plot)})

        # Resolve stochastic params
        stochastic_params: List[str] = []
        if stochastic_params_cfg is not None:
            for param in stochastic_params_cfg.cfg:
                resolved = _resolve_column_name(str(param), df_plot.columns.tolist())
                if resolved is not None:
                    stochastic_params.append(resolved)
                else:
                    logging.warning("Stochastic parameter '%s' not found in dataframe; skipping.", param)

        # Apply metrics filter (applies to both summary table AND line/scatter plots)
        plot_metrics = _filter_metrics_by_cfg(
            numeric_metrics, paper_summary_metrics_filter_cfg, "plots"
        )

        # Statistical comparison table (win-rate / n / paired Wilcoxon p / mean diff)
        summary_group_cols = [p for p in sweep_params if p not in stochastic_params]
        significance_lookup: Optional[Dict[Tuple[Any, str], bool]] = None
        stat_baseline_col: Optional[str] = None
        stat_baseline_value: Any = None
        if paper_stat_table_enabled:
            if not paper_stat_baseline_param or paper_stat_baseline_value is None:
                logging.warning(
                    "paper_stat_table_enabled=True but paper_stat_baseline_param/paper_stat_baseline_value "
                    "are not both set; skipping statistical comparison table."
                )
            else:
                stat_baseline_col = _resolve_column_name(str(paper_stat_baseline_param), df_plot.columns.tolist())
                if stat_baseline_col is None:
                    logging.warning(
                        "paper_stat_baseline_param '%s' not found in dataframe; skipping statistical comparison table.",
                        paper_stat_baseline_param,
                    )
                else:
                    matches = [v for v in df_plot[stat_baseline_col].tolist() if str(v) == str(paper_stat_baseline_value)]
                    if not matches:
                        logging.warning(
                            "paper_stat_baseline_value '%s' not found in column '%s'; skipping statistical comparison table.",
                            paper_stat_baseline_value, stat_baseline_col,
                        )
                        stat_baseline_col = None
                    else:
                        stat_baseline_value = matches[0]

            if stat_baseline_col is not None:
                lower_is_better_metrics: List[str] = []
                if paper_stat_lower_is_better_metrics_cfg is not None:
                    lower_is_better_metrics = [
                        str(v).strip() for v in paper_stat_lower_is_better_metrics_cfg.cfg if str(v).strip()
                    ]

                stat_df, significance_lookup = _compute_stat_comparisons(
                    df_plot=df_plot,
                    summary_group_cols=summary_group_cols,
                    stochastic_params=stochastic_params,
                    baseline_col=stat_baseline_col,
                    baseline_value=stat_baseline_value,
                    numeric_metrics=plot_metrics,
                    lower_is_better_metrics=lower_is_better_metrics,
                    alpha=paper_stat_alpha,
                )
                if stat_df.empty:
                    logging.info(
                        "Statistical comparison table empty (no non-baseline settings or no paired samples found)."
                    )
                else:
                    # One table + figure per metric (a single combined table gets cluttered once
                    # more than one metric / setting is being compared).
                    for metric in plot_metrics:
                        metric_df = stat_df[stat_df["metric"] == metric].drop(columns=["metric"])
                        if metric_df.empty:
                            continue
                        wandb.log({f"paper_stat_table/{metric}/data": wandb.Table(dataframe=_sanitize_table_df(metric_df))})
                        stat_fig = _render_table_figure(
                            table_df=_format_stat_table_for_display(metric_df),
                            title=(
                                f"Statistical Comparison ({metric}) vs. baseline {stat_baseline_col}={stat_baseline_value} "
                                f"(* = two-sided Wilcoxon p < {paper_stat_alpha})"
                            ),
                            figsize=figsize_cfg.cfg,
                            dpi=dpi,
                        )
                        wandb.log({f"paper_stat_table/{metric}/figure": wandb.Image(stat_fig)})
                        plt.close(stat_fig)

        # Paper summary table (mean ± std, "*" marks cells significant vs. baseline)
        if paper_summary_table_enabled:
            summary_df = _build_summary_table(
                df_plot=df_plot,
                group_cols=list(summary_group_cols),
                numeric_metrics=plot_metrics,
                significance_lookup=significance_lookup,
            )
            wandb.log({"paper_summary_table/data": wandb.Table(dataframe=summary_df)})
            summary_title = "Sweep Summary (mean ± std)"
            if significance_lookup is not None:
                summary_title += (
                    f" - * p < {paper_stat_alpha} vs. baseline {stat_baseline_col}={stat_baseline_value} "
                    "(two-sided Wilcoxon)"
                )
            summary_fig = _render_table_figure(
                table_df=summary_df,
                title=summary_title,
                figsize=figsize_cfg.cfg,
                dpi=dpi,
            )
            wandb.log({"paper_summary_table/figure": wandb.Image(summary_fig)})
            plt.close(summary_fig)

        # -- Optional paper-style pivot tables --------------------------------
        if pivot_tables_cfg is not None:
            for pivot_spec_raw in pivot_tables_cfg.cfg:
                try:
                    pivot_spec = _parse_pivot_spec(pivot_spec_raw)
                    pivot_df, pivot_meta = _build_pivot_table(df_plot=df_plot, spec=pivot_spec)
                except ValueError as exc:
                    logging.warning("Skipping pivot table %s: %s", pivot_spec_raw, exc)
                    continue

                pivot_name = pivot_spec["name"]
                pivot_flat = _flatten_pivot_for_wandb(pivot_df)
                wandb.log({f"pivot_table/{pivot_name}/data": wandb.Table(dataframe=pivot_flat)})
                pivot_fig = _render_pivot_table_figure(
                    pivot=pivot_df,
                    meta=pivot_meta,
                    title=pivot_spec["title"],
                    figsize=figsize_cfg.cfg,
                    dpi=dpi,
                    decimals=pivot_spec["decimals"],
                )
                wandb.log({f"pivot_table/{pivot_name}/figure": wandb.Image(pivot_fig)})
                plt.close(pivot_fig)

        plot_fn = sns.lineplot if type == "line" else sns.scatterplot

        # For each metric, plot against each sweep parameter (x), and optionally color by a different sweep param
        for metric in plot_metrics:
            for i, x_param in enumerate(sweep_params):
            # Optionally, try coloring by a different sweep param (if available)
                for j, hue_param in enumerate(sweep_params):
                    if hue_param == x_param:
                        continue  # Don't use same param for x and hue

                    # continue if x_param is only one unique value
                    if df_plot[x_param].nunique() <= 1:
                        continue

                    plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)  # Set desired dpi here
                    # Merge results and sweep for plotting
                    plot_fn(
                        data=df_plot,
                        x=x_param,
                        y=metric,
                        hue=hue_param if len(sweep_params) > 1 else None,
                        palette=palette
                    )
                    plot_title = f"{metric} vs {x_param}" + (f" cb {hue_param}" if len(sweep_params) > 1 else "")
                    plt.title(plot_title)
                    plt.xlabel(x_param)
                    plt.ylabel(metric)
                    if log_lin_scale_auto_enabled:
                        # first check if the df_plot[x_param] is a numeric type
                        if pd.api.types.is_numeric_dtype(df_plot[x_param]):
                            x_ratio = df_plot[x_param].max() / df_plot[x_param].min()
                            plt.xscale('log' if x_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                        y_ratio = df_plot[metric].max() / df_plot[metric].min()
                        plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                    plt.tight_layout()
                    plt.show()
                    wandb.log({f"{metric}__vs__{x_param}": wandb.Image(plt)})
                    plt.close()
                    
                # If only one sweep param, plot without hue
                if len(sweep_params) == 1:
                    if df_plot[x_param].nunique() <= 1:
                        continue
                    
                    plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)
                    df_plot = pd.concat([df_results, df_sweep], axis=1)
                    plot_fn(
                        data=df_plot,
                        x=x_param,
                        y=metric,
                        palette=palette
                    )
                    plot_title = f"{metric} vs {x_param}"
                    plt.title(plot_title)
                    plt.xlabel(x_param)
                    plt.ylabel(metric)
                    if log_lin_scale_auto_enabled:
                        x_ratio = df_plot[x_param].max() / df_plot[x_param].min()
                        plt.xscale('log' if x_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                        y_ratio = df_plot[metric].max() / df_plot[metric].min()
                        plt.yscale('log' if y_ratio > math.pow(10, log_lin_scale_threshold) else 'linear')
                    plt.tight_layout()
                    plt.show()
                    wandb.log({f"{metric}__vs__{x_param}": wandb.Image(plt)})
                    plt.close()

        # Tradeoff plots (cine-style automatic pairwise metric tradeoff)
        if tradeoff_plots_enabled:
            tradeoff_metrics = _filter_metrics_by_cfg(
                numeric_metrics, paper_summary_metrics_filter_cfg, "tradeoff"
            )
            if len(tradeoff_metrics) < 2:
                logging.info("Tradeoff plots skipped: fewer than 2 filtered metrics available.")
            else:
                # Build aggregated dataframe (same aggregation as used above)
                stochastic_params_resolved: List[str] = []
                if stochastic_params_cfg is not None:
                    for _sp in stochastic_params_cfg.cfg:
                        _r = _resolve_column_name(str(_sp), df_plot.columns.tolist())
                        if _r is not None:
                            stochastic_params_resolved.append(_r)

                tradeoff_group_params = [p for p in sweep_params if p not in stochastic_params_resolved]
                if stochastic_params_resolved and tradeoff_group_params:
                    _tagg = df_plot.groupby(tradeoff_group_params, dropna=False)[tradeoff_metrics].agg(["mean", "std"])
                    _tagg.columns = [f"{m}__{s}" for m, s in _tagg.columns]
                    tradeoff_df = _tagg.reset_index()
                    _mean = {m: f"{m}__mean" for m in tradeoff_metrics}
                    _std  = {m: f"{m}__std"  for m in tradeoff_metrics}
                elif stochastic_params_resolved:
                    _global = {f"{m}__mean": [df_plot[m].mean()] for m in tradeoff_metrics}
                    _global.update({f"{m}__std": [df_plot[m].std()] for m in tradeoff_metrics})
                    tradeoff_df = pd.DataFrame(_global)
                    tradeoff_df["_all"] = "all"
                    tradeoff_group_params = ["_all"]
                    _mean = {m: f"{m}__mean" for m in tradeoff_metrics}
                    _std  = {m: f"{m}__std"  for m in tradeoff_metrics}
                else:
                    tradeoff_df = df_plot.copy()
                    tradeoff_group_params = list(sweep_params)
                    _mean = {m: m for m in tradeoff_metrics}
                    _std  = {m: None for m in tradeoff_metrics}

                color_param = tradeoff_group_params[0] if len(tradeoff_group_params) >= 1 else None
                style_param = tradeoff_group_params[1] if len(tradeoff_group_params) >= 2 else None

                if color_param is not None:
                    color_values = list(dict.fromkeys(tradeoff_df[color_param].tolist()))
                else:
                    color_values = ["all"]
                    tradeoff_df = tradeoff_df.copy()
                    tradeoff_df["_color_val"] = "all"
                if style_param is not None:
                    style_values = list(dict.fromkeys(tradeoff_df[style_param].tolist()))
                else:
                    style_values = ["all"]
                    tradeoff_df["_style_val"] = "all"

                if color_param is not None:
                    tradeoff_df["_color_val"] = tradeoff_df[color_param]
                if style_param is not None:
                    tradeoff_df["_style_val"] = tradeoff_df[style_param]

                _pal = sns.color_palette(palette, n_colors=max(1, len(color_values)))
                color_to_rgb = {v: _pal[i] for i, v in enumerate(color_values)}
                line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
                style_to_ls = {v: line_styles[i % len(line_styles)] for i, v in enumerate(style_values)}

                def _sort_isoline(df_iso: pd.DataFrame, axis_param: Optional[str], fallback_col: str) -> pd.DataFrame:
                    if axis_param is None or axis_param not in df_iso.columns:
                        return df_iso.sort_values(fallback_col)

                    axis_vals = df_iso[axis_param]
                    numeric_vals = pd.to_numeric(axis_vals, errors="coerce")
                    if numeric_vals.notna().all():
                        return (
                            df_iso.assign(_sort_numeric=numeric_vals)
                            .sort_values("_sort_numeric", kind="mergesort")
                            .drop(columns=["_sort_numeric"])
                        )

                    extracted_numeric = pd.to_numeric(
                        axis_vals.astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0],
                        errors="coerce",
                    )
                    if extracted_numeric.notna().any():
                        return (
                            df_iso.assign(
                                _sort_numeric=extracted_numeric,
                                _sort_text=axis_vals.astype(str),
                            )
                            .sort_values(["_sort_numeric", "_sort_text"], na_position="last", kind="mergesort")
                            .drop(columns=["_sort_numeric", "_sort_text"])
                        )

                    return (
                        df_iso.assign(_sort_text=axis_vals.astype(str))
                        .sort_values("_sort_text", kind="mergesort")
                        .drop(columns=["_sort_text"])
                    )

                for x_metric, y_metric in _permutations(tradeoff_metrics, 2):
                    x_col = _mean[x_metric]
                    y_col = _mean[y_metric]
                    x_std_col = _std[x_metric]
                    y_std_col = _std[y_metric]

                    plt.figure(figsize=figsize_cfg.cfg, dpi=dpi)

                    # Isolines: group by style_val, draw connecting line in gray
                    if color_param is not None and style_param is not None:
                        for _sv, _iso in tradeoff_df.groupby("_style_val", dropna=False):
                            _iso = _sort_isoline(_iso, color_param, x_col)
                            if len(_iso) > 1:
                                plt.plot(_iso[x_col], _iso[y_col],
                                         linestyle=style_to_ls[_sv], color="gray",
                                         alpha=0.28, linewidth=0.9, zorder=1)
                        if tradeoff_same_color_lines_enabled:
                            for _cv, _iso in tradeoff_df.groupby("_color_val", dropna=False):
                                _iso = _sort_isoline(_iso, style_param, x_col)
                                if len(_iso) > 1:
                                    plt.plot(_iso[x_col], _iso[y_col],
                                             linestyle="-", color=color_to_rgb[_cv],
                                             alpha=0.20, linewidth=0.8, zorder=1)

                    for _, _row in tradeoff_df.iterrows():
                        _c = color_to_rgb[_row["_color_val"]]
                        _xerr = (_row[x_std_col] if x_std_col and x_std_col in _row.index
                                 and not pd.isna(_row[x_std_col]) else None)
                        _yerr = (_row[y_std_col] if y_std_col and y_std_col in _row.index
                                 and not pd.isna(_row[y_std_col]) else None)
                        if stochastic_params_resolved and tradeoff_errorbars_enabled:
                            plt.errorbar(x=_row[x_col], y=_row[y_col],
                                         xerr=_xerr, yerr=_yerr,
                                         fmt="o", color=_c, alpha=0.85, capsize=3)
                        else:
                            plt.scatter(_row[x_col], _row[y_col],
                                        color=_c, marker="o", alpha=0.85, zorder=3)

                    _color_handles = [
                        Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color_to_rgb[v], markeredgecolor=color_to_rgb[v],
                               markersize=6, linestyle="", label=str(v))
                        for v in color_values
                    ]
                    _style_handles = [
                        Line2D([0], [0], color="black",
                               linestyle=style_to_ls[v], linewidth=1.5, label=str(v))
                        for v in style_values
                    ]
                    _legend_y = -0.12
                    if len(_color_handles) > 1 or color_param is not None:
                        _cl = plt.legend(handles=_color_handles,
                                         title=color_param if color_param else "color",
                                         loc="upper center",
                                         bbox_to_anchor=(0.5, _legend_y),
                                         borderaxespad=0.0,
                                         ncol=min(6, max(1, len(_color_handles))),
                                         frameon=False)
                        plt.gca().add_artist(_cl)
                        _legend_y -= 0.14
                    if len(_style_handles) > 1 or style_param is not None:
                        plt.legend(handles=_style_handles,
                                   title=style_param if style_param else "style",
                                   loc="upper center",
                                   bbox_to_anchor=(0.5, _legend_y),
                                   borderaxespad=0.0,
                                   ncol=min(6, max(1, len(_style_handles))),
                                   frameon=False)

                    plt.title(f"Tradeoff: {y_metric} vs {x_metric}")
                    plt.xlabel(x_metric)
                    plt.ylabel(y_metric)
                    if log_lin_scale_auto_enabled:
                        _xv = tradeoff_df[x_col]
                        _yv = tradeoff_df[y_col]
                        if pd.api.types.is_numeric_dtype(_xv):
                            _xmin, _xmax = _xv.min(), _xv.max()
                            if _xmin > 0 and _xmax > 0:
                                plt.xscale('log' if _xmax / _xmin > math.pow(10, log_lin_scale_threshold) else 'linear')
                        if pd.api.types.is_numeric_dtype(_yv):
                            _ymin, _ymax = _yv.min(), _yv.max()
                            if _ymin > 0 and _ymax > 0:
                                plt.yscale('log' if _ymax / _ymin > math.pow(10, log_lin_scale_threshold) else 'linear')
                    plt.tight_layout()
                    if style_param is not None:
                        plt.subplots_adjust(bottom=0.30)
                    elif color_param is not None:
                        plt.subplots_adjust(bottom=0.20)
                    plt.show()
                    wandb.log({f"tradeoff__{y_metric}__vs__{x_metric}": wandb.Image(plt)})
                    plt.close()