from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.utils.wandb_utils import WandbParamsTask, wandb_kwargs_for_prefect_task


def _as_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[Path] = []
        for item in value.values():
            out.extend(_as_paths(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[Path] = []
        for item in value:
            out.extend(_as_paths(item))
        return out
    return [Path(str(value))]


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (item for item in path.rglob("*") if item.is_file())


def summarize_path(value: Any) -> Dict[str, Any]:
    paths = _as_paths(value)
    files: list[Path] = []
    for path in paths:
        files.extend(_iter_files(path))

    total_bytes = 0
    for file_path in files:
        try:
            total_bytes += file_path.stat().st_size
        except OSError:
            logging.warning("Could not stat dataset file %s", file_path)

    h5_files = [path for path in files if path.suffix == ".h5"]
    return {
        "paths": [str(path) for path in paths],
        "num_files": len(files),
        "num_h5_files": len(h5_files),
        "bytes": total_bytes,
        "gb": total_bytes / (1024 ** 3),
    }


def summarize_output_paths(output_paths: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for key, value in output_paths.items():
        if key in {"dataset_name", "status", "meta", "counts"}:
            continue
        paths = _as_paths(value)
        if not paths:
            continue
        if any(path.exists() for path in paths):
            summary[str(key)] = summarize_path(value)
    return summary


def build_dataset_summary_logs(
    *,
    stage: str,
    output_paths: Dict[str, Any],
    counts: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], list[list[Any]]]:
    summary = summarize_output_paths(output_paths)
    counts = dict(counts or {})
    metrics: Dict[str, Any] = {
        f"{stage}/num_splits": len(summary),
    }
    total_files = 0
    total_h5_files = 0
    total_bytes = 0
    table_rows: list[list[Any]] = []
    for split, values in summary.items():
        num_files = int(values["num_files"])
        num_h5_files = int(values["num_h5_files"])
        num_bytes = int(values["bytes"])
        total_files += num_files
        total_h5_files += num_h5_files
        total_bytes += num_bytes
        metrics[f"{stage}/{split}/num_files"] = num_files
        metrics[f"{stage}/{split}/num_h5_files"] = num_h5_files
        metrics[f"{stage}/{split}/gb"] = float(values["gb"])
        if split in counts:
            metrics[f"{stage}/{split}/count"] = counts[split]
        table_rows.append([split, num_files, num_h5_files, num_bytes, float(values["gb"]), "\n".join(values["paths"])])

    metrics[f"{stage}/total_num_files"] = total_files
    metrics[f"{stage}/total_num_h5_files"] = total_h5_files
    metrics[f"{stage}/total_bytes"] = total_bytes
    metrics[f"{stage}/total_gb"] = total_bytes / (1024 ** 3)
    for key, value in counts.items():
        if isinstance(value, (int, float)):
            metrics[f"{stage}/counts/{key}"] = value
    return metrics, table_rows


def log_dataset_summary_to_wandb(
    *,
    stage: str,
    dataset_name: str,
    output_paths: Dict[str, Any],
    wandb_params_task: Optional[WandbParamsTask],
    counts: Optional[Dict[str, Any]] = None,
    name_aux: Optional[str] = None,
) -> None:
    if wandb_params_task is None:
        return

    import wandb

    from src.prefect.wandb_lock import locked_wandb_init

    params = wandb_params_task.model_copy(update={"name_aux": name_aux or stage})
    metrics, table_rows = build_dataset_summary_logs(
        stage=stage,
        output_paths=output_paths,
        counts=counts,
    )

    with locked_wandb_init(**wandb_kwargs_for_prefect_task(params)):
        wandb.log(metrics)
        if table_rows:
            table = wandb.Table(
                columns=["split", "num_files", "num_h5_files", "bytes", "gb", "paths"],
                data=table_rows,
            )
            wandb.log({f"{stage}/summary_table": table})
        if wandb.run is not None:
            wandb.run.summary.update(metrics)
            wandb.run.summary[f"{stage}/dataset_name"] = dataset_name
