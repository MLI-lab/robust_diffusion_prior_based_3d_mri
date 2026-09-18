from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm.autonotebook import tqdm


DEFAULT_EXCLUDED_SESSION_PATTERN = None


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_list(value: Any) -> list[Any]:
    value = _to_plain(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _read_manifest_from_jsons(raw_path: Path, cfg: Any) -> pd.DataFrame:
    import glob

    wildcard = str(getattr(cfg, "json_subfolder_wildcard", "sub-yv98/*/anat/*T1w.json"))
    json_files = sorted(glob.glob(str(raw_path / wildcard)))
    exclude_pattern = getattr(cfg, "exclude_session_regex", DEFAULT_EXCLUDED_SESSION_PATTERN)
    rows = []
    for json_file in json_files:
        if exclude_pattern and re.search(str(exclude_pattern), json_file):
            continue
        with open(json_file, "r") as f:
            row = json.load(f)
        row["json_file_path"] = json_file
        nifti_file = Path(json_file).with_suffix("")
        if nifti_file.name.endswith(".nii"):
            nifti_path = nifti_file
        else:
            nifti_path = Path(json_file).with_name(Path(json_file).name.replace(".json", ".nii.gz"))
        row["nifti_file_path"] = str(nifti_path)
        if not nifti_path.exists():
            raise FileNotFoundError(f"Expected NIfTI file for {json_file}: {nifti_path}")
        rows.append(row)
    return pd.DataFrame(rows)


def _read_manifest(raw_path: Path, cfg: Any) -> pd.DataFrame:
    manifest_path = getattr(cfg, "manifest_path", None)
    if manifest_path:
        path = Path(str(manifest_path))
        if not path.is_absolute():
            path = raw_path / path
        if path.suffix.lower() in (".pkl", ".pickle"):
            df = pd.read_pickle(path)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path)
        if "nifti_file_path" not in df.columns:
            raise ValueError(f"Lüsebrink manifest {path} must contain a nifti_file_path column.")
        if "json_file_path" not in df.columns:
            df["json_file_path"] = None
        return df
    return _read_manifest_from_jsons(raw_path, cfg)


def _base_frames(df: pd.DataFrame, cfg: Any) -> dict[str, pd.DataFrame]:
    base_train_filter = _cfg_get(cfg, "base_train_filter", {})
    excluded_resolutions = _as_list(_cfg_get(base_train_filter, "exclude_resolutions", []))
    exclude_path_regex = _cfg_get(base_train_filter, "exclude_path_regex", None)
    test_025 = df[df["SliceThickness"] == 0.25]
    test_07 = df[df["SliceThickness"] == 0.7]
    test_065 = df[df["SliceThickness"] == 0.65]
    train = df[~df["SliceThickness"].isin([float(v) for v in excluded_resolutions])]
    train = _apply_path_regex(train, exclude_path_regex, include=False)
    return {"df": df, "train_base": train, "test_025mm": test_025, "test_07mm": test_07, "test_065mm": test_065}


def _global_split_pools(df: pd.DataFrame, cfg: Any, seed: int) -> dict[str, pd.DataFrame]:
    split_cfg = _cfg_get(cfg, "global_split", {})
    if not bool(_cfg_get(split_cfg, "enabled", True)):
        return {"train_pool": _base_frames(df, cfg)["train_base"], "test_val_pool": df.iloc[0:0]}

    source_cfg = _cfg_get(split_cfg, "source", {"source": "all"})
    if isinstance(source_cfg, str):
        source_cfg = {"source": source_cfg}
    candidate = _source_frame(df, source_cfg, {"__root_cfg": cfg})
    candidate = _filter_subset(candidate, split_cfg)

    max_per_resolution = _cfg_get(
        split_cfg,
        "max_test_val_per_resolution",
        _cfg_get(split_cfg, "test_val_per_resolution", 1),
    )
    max_per_resolution = int(max_per_resolution)
    if max_per_resolution <= 0 or "SliceThickness" not in candidate:
        test_val = candidate.iloc[0:0]
    else:
        parts = []
        for idx, resolution in enumerate(sorted(candidate["SliceThickness"].dropna().unique())):
            sub = candidate[candidate["SliceThickness"] == resolution]
            n = min(max_per_resolution, len(sub))
            if n > 0:
                parts.append(sub.sample(n=n, random_state=np.random.default_rng(seed=seed + idx)))
        test_val = pd.concat(parts) if parts else candidate.iloc[0:0]

    train_pool = candidate.drop(test_val.index)
    return {"train_pool": train_pool, "test_val_pool": test_val}


def _sample_exact(df: pd.DataFrame, n: Optional[int], random_state: Any) -> pd.DataFrame:
    if n is None:
        return df
    n = int(n)
    if n < 0:
        return df
    if len(df) < n:
        raise ValueError(f"Cannot sample {n} rows from only {len(df)} rows.")
    return df.sample(n=n, random_state=random_state)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolution_label(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _apply_path_regex(df: pd.DataFrame, pattern: Any, include: bool) -> pd.DataFrame:
    if not pattern:
        return df
    mask = pd.Series(False, index=df.index)
    for column in ("json_file_path", "nifti_file_path"):
        if column in df:
            mask |= df[column].astype(str).str.contains(str(pattern), regex=True, na=False)
    return df[mask] if include else df[~mask]


def _apply_query(df: pd.DataFrame, query: Optional[str]) -> pd.DataFrame:
    if not query:
        return df
    return df.query(str(query), engine="python")


def _filter_subset(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    out = df
    resolutions = _as_list(_cfg_get(cfg, "resolutions", _cfg_get(cfg, "slice_thickness", None)))
    if resolutions:
        out = out[out["SliceThickness"].isin([float(v) for v in resolutions])]
    resolution_regex = _cfg_get(cfg, "resolution_regex", None)
    if resolution_regex:
        pattern = re.compile(str(resolution_regex))
        out = out[out["SliceThickness"].map(lambda v: bool(pattern.search(_resolution_label(v))))]
    out = _apply_query(out, _cfg_get(cfg, "query", None))
    exclude_query = _cfg_get(cfg, "exclude_query", None)
    if exclude_query:
        out = out.drop(_apply_query(out, exclude_query).index)
    out = _apply_path_regex(out, _cfg_get(cfg, "path_regex", None), include=True)
    out = _apply_path_regex(out, _cfg_get(cfg, "exclude_path_regex", None), include=False)
    out = _apply_path_regex(out, _cfg_get(cfg, "exclude_session_regex", None), include=False)
    dedup_by = _as_list(_cfg_get(cfg, "deduplicate_by", None))
    if dedup_by:
        present = [c for c in (str(c) for c in dedup_by) if c in out.columns]
        if present:
            out = out.drop_duplicates(subset=present, keep="first")
    return out


def _sample_distributed_by_resolution(df: pd.DataFrame, limit: int, random_state: Any, allow_short_groups: bool) -> pd.DataFrame:
    resolutions = sorted(df["SliceThickness"].dropna().unique())
    if not resolutions or limit < 0:
        return df
    if allow_short_groups:
        return _sample_balanced_by_resolution(df, limit, random_state)
    base_n = limit // len(resolutions)
    remainder = limit % len(resolutions)
    parts = []
    for idx, resolution in enumerate(resolutions):
        quota = base_n + (1 if idx < remainder else 0)
        if quota <= 0:
            continue
        sub = df[df["SliceThickness"] == resolution]
        if len(sub) < quota:
            raise ValueError(f"Cannot sample {quota} rows at SliceThickness={resolution}; only {len(sub)} available.")
        parts.append(sub.sample(n=quota, random_state=random_state))
    return pd.concat(parts) if parts else df.iloc[0:0]


def _sample_balanced_by_resolution(df: pd.DataFrame, limit: int, random_state: Any) -> pd.DataFrame:
    resolutions = sorted(df["SliceThickness"].dropna().unique())
    if not resolutions or limit < 0:
        return df

    groups = {resolution: df[df["SliceThickness"] == resolution] for resolution in resolutions}
    quotas = {resolution: 0 for resolution in resolutions}
    remaining = int(limit)

    while remaining > 0:
        available = [resolution for resolution in resolutions if quotas[resolution] < len(groups[resolution])]
        if not available:
            break
        step = max(1, remaining // len(available))
        progressed = False
        for resolution in available:
            if remaining <= 0:
                break
            add = min(step, len(groups[resolution]) - quotas[resolution], remaining)
            if add > 0:
                quotas[resolution] += add
                remaining -= add
                progressed = True
        if not progressed:
            break

    parts = [
        groups[resolution].sample(n=quota, random_state=random_state)
        for resolution, quota in quotas.items()
        if quota > 0
    ]
    return pd.concat(parts) if parts else df.iloc[0:0]


def _apply_limit(df: pd.DataFrame, cfg: Any, seed: int) -> pd.DataFrame:
    volume_limit = _cfg_get(cfg, "volume_limit", _cfg_get(cfg, "sample_n", None))
    if volume_limit is None:
        limit = _cfg_get(cfg, "limit", None)
        if limit is not None and int(limit) >= 0:
            return df.head(int(limit))
        return df
    volume_limit = int(volume_limit)
    if volume_limit < 0:
        return df
    if bool(_cfg_get(cfg, "distribute_volume_limit_by_resolution", False)):
        return _sample_distributed_by_resolution(
            df,
            volume_limit,
            np.random.default_rng(seed=seed),
            bool(_cfg_get(cfg, "allow_short_resolution_groups", False)),
        )
    return _sample_exact(df, volume_limit, np.random.default_rng(seed=seed))


def _select_grouped_subset(source: pd.DataFrame, cfg: Any, seed: int) -> pd.DataFrame:
    groups = _as_list(_cfg_get(cfg, "resolution_groups", None))
    if not groups:
        return _apply_limit(_filter_subset(source, cfg), cfg, seed)
    parts = []
    for idx, group_cfg in enumerate(groups):
        group = _filter_subset(source, group_cfg)
        parts.append(_apply_limit(group, group_cfg, seed + idx))
    return pd.concat(parts) if parts else source.iloc[0:0]


def _source_frame(df: pd.DataFrame, cfg: Any, sources: dict[str, Any]) -> pd.DataFrame:
    source = str(_cfg_get(cfg, "source", "base_train" if bool(_cfg_get(cfg, "base_train_only", True)) else "all"))
    if source == "all":
        return df
    if source == "base_train":
        return _base_frames(df, _cfg_get(sources, "__root_cfg", None))["train_base"]
    if source in ("train_pool", "test_val_pool"):
        if source not in sources:
            raise ValueError(f"Lüsebrink split source={source} requires global_split to be prepared first.")
        return sources[source]
    if source == "train_selection":
        if "train_selection" not in sources:
            raise ValueError("Lüsebrink split source=train_selection requires the train split to be selected first.")
        return sources["train_selection"]
    if source == "empty":
        return df.iloc[0:0]
    raise ValueError(f"Unknown Lüsebrink split source={source!r}.")


def _model_subset(df: pd.DataFrame, split_cfg: Any, seed: int, sources: dict[str, Any]) -> pd.DataFrame:
    source = _source_frame(df, split_cfg, sources)
    return _select_grouped_subset(source, split_cfg, seed)


def _resolve_subset_model(cfg: Any, subset_name: str) -> Any:
    subset_model = _cfg_get(cfg, "subset_model", None)
    if subset_model is not None:
        return subset_model
    subset_name = str(_cfg_get(_cfg_get(cfg, "subset_aliases", {}), subset_name, subset_name))
    if subset_name == "custom":
        return {
            "train": getattr(cfg, "custom_train", {}),
            "test_val": getattr(cfg, "custom_test_val", {}),
        }
    return _cfg_get(_cfg_get(cfg, "subset_models", {}), subset_name, None)


def _select_splits(df: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(getattr(cfg, "seed", 42))
    subset_name = str(getattr(cfg, "subset_name", "diverse_v14"))
    subset_model = _resolve_subset_model(cfg, subset_name)
    if subset_model is None:
        raise ValueError(f"Unknown Lüsebrink subset_name={subset_name!r} and no subset_model was configured.")

    sources: dict[str, Any] = {"__root_cfg": cfg}
    sources.update(_global_split_pools(df, cfg, seed))
    train_cfg = _cfg_get(subset_model, "train", {"source": "train_pool"})
    test_val_cfg = _cfg_get(subset_model, "test_val", {"source": "empty"})
    train = _model_subset(df, train_cfg, seed, sources)
    sources["train_selection"] = train
    test_val = _model_subset(df, test_val_cfg, seed + 1, sources)
    if str(_cfg_get(test_val_cfg, "source", "")) == "train_selection":
        train = train.drop(test_val.index)

    max_train = getattr(cfg, "max_train", None)
    max_test_val = getattr(cfg, "max_test_val", None)
    if max_train is not None and int(max_train) >= 0:
        train = train.head(int(max_train))
    if max_test_val is not None and int(max_test_val) >= 0:
        test_val = test_val.head(int(max_test_val))
    return train.copy(), test_val.copy()


def _safe_attr(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(value, default=str)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def _write_one_orientation(output_file: Path, data: np.ndarray, attrs: dict[str, Any], orientation: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_name(output_file.stem + f".{os.getpid()}.tmp{output_file.suffix}")
    try:
        with h5py.File(tmp, "w") as hf:
            hf.create_dataset("reconstruction_rss", data=data.astype(np.float32, copy=False))
            hf.attrs["orientation"] = orientation
            hf.attrs["original_shape"] = np.asarray(attrs.pop("original_shape"), dtype=np.int64)
            hf.attrs["magnitude_only"] = True
            hf.attrs["source_modality"] = "nifti"
            for key, value in attrs.items():
                if key in ("nifti_file_path", "json_file_path"):
                    continue
                try:
                    hf.attrs[str(key)] = _safe_attr(value)
                except Exception:
                    hf.attrs[str(key)] = str(value)
        os.replace(tmp, output_file)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _convert_rows(df: pd.DataFrame, output_dir: Path, cfg: Any) -> int:
    import nibabel

    orientations = [str(v) for v in _as_list(getattr(cfg, "orientations", ["axial", "coronal", "sagittal"]))]
    exists_ok = bool(getattr(cfg, "exists_ok", True))
    count = 0
    for _, row in tqdm(list(df.iterrows()), total=len(df), desc=f"Converting Lüsebrink {output_dir.name}"):
        nifti_path = Path(str(row["nifti_file_path"]))
        img = nibabel.load(str(nifti_path))
        volume = np.asarray(img.get_fdata(), dtype=np.float32)
        base_name = nifti_path.name.replace(".nii.gz", "").replace(".nii", "")
        attrs = row.to_dict()
        attrs["original_shape"] = volume.shape
        data_by_orientation = {
            "axial": np.transpose(volume, (2, 1, 0)),
            "coronal": np.transpose(volume, (1, 0, 2)),
            "sagittal": volume,
        }
        for orientation in orientations:
            if orientation not in data_by_orientation:
                raise ValueError(f"Unknown Lüsebrink orientation {orientation!r}.")
            suffix = {"axial": "axial", "coronal": "coronal", "sagittal": "sagittal"}[orientation]
            output_file = output_dir / f"{base_name}_{suffix}.h5"
            if exists_ok and output_file.exists():
                count += 1
                continue
            _write_one_orientation(output_file, data_by_orientation[orientation], dict(attrs), orientation)
            count += 1
    return count


def _jsonable_resolution_values(values: Any) -> list[float]:
    return [float(v) for v in _as_list(values)]


def _selected_resolution_summary(selected: pd.DataFrame) -> dict[str, Any]:
    if "SliceThickness" not in selected:
        return {"values_mm": [], "counts": {}}
    counts = selected["SliceThickness"].value_counts().sort_index()
    return {
        "values_mm": [float(v) for v in counts.index.tolist()],
        "counts": {str(k): int(v) for k, v in counts.items()},
    }


def _preview_selected_rows(selected: pd.DataFrame, max_rows: int = 8) -> list[dict[str, Any]]:
    preferred = [
        "participant_id",
        "subject_id",
        "sub",
        "session_id",
        "ses",
        "SliceThickness",
        "nifti_file_path",
        "json_file_path",
    ]
    columns = [c for c in preferred if c in selected.columns]
    if not columns:
        return []
    preview = []
    for row in selected[columns].head(max_rows).to_dict(orient="records"):
        item = {}
        for key, value in row.items():
            if key == "SliceThickness" and value == value:
                item["resolution_mm"] = float(value)
            elif key.endswith("_file_path") and value:
                item[key] = Path(str(value)).name
            else:
                item[key] = _safe_attr(value)
        preview.append(item)
    return preview


def run(input_dir: str | Path, output_dir: str | Path, fold: str, cfg: Any, **_: object) -> int:
    raw_path = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _read_manifest(raw_path, cfg)
    train_df, test_val_df = _select_splits(df, cfg)
    selected = train_df if fold == "train" else test_val_df
    selected.to_csv(output_dir / f"_luesebrink_manifest_{fold}.csv", index=False)
    count = _convert_rows(selected, output_dir, cfg)

    subset_name = str(getattr(cfg, "subset_name", "diverse_v14"))
    subset_model = _resolve_subset_model(cfg, subset_name) or {}
    split_cfg = _cfg_get(subset_model, fold, {})
    requested_resolutions = _cfg_get(split_cfg, "resolutions", _cfg_get(split_cfg, "slice_thickness", None))
    selected_resolution_summary = _selected_resolution_summary(selected)

    meta = {
        "dataset_name": "luesebrink",
        "subset_name": subset_name,
        "fold": fold,
        "split_source": str(_cfg_get(split_cfg, "source", "train_pool" if fold == "train" else "empty")),
        "requested_resolutions_mm": None if requested_resolutions is None else _jsonable_resolution_values(requested_resolutions),
        "selected_resolutions_mm": selected_resolution_summary["values_mm"],
        "selected_resolution_counts": selected_resolution_summary["counts"],
        "num_source_rows": int(len(selected)),
        "num_h5_files": int(count),
        "orientations": [str(v) for v in _as_list(getattr(cfg, "orientations", ["axial", "coronal", "sagittal"]))],
        "selected_rows_preview": _preview_selected_rows(selected),
        "slice_thickness_counts": selected_resolution_summary["counts"],
    }
    with open(output_dir / f"_luesebrink_conversion_{fold}.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return count
