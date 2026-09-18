from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from src.flows.flow_utils.tuned_recon.types import DatasetOutput, LabeledFuture, Labels


def iter_split_groups(
    result_list: List[LabeledFuture],
    df_sweep: pd.DataFrame,
    split_keys: List[str],
) -> Iterator[Tuple[str, List[LabeledFuture], pd.DataFrame]]:
    if not split_keys or df_sweep.empty:
        return

    cols = df_sweep.columns.tolist()
    matched: List[str] = []
    for split_key in split_keys:
        if split_key in cols:
            matched.append(split_key)
            continue
        bare = split_key.lstrip("+").lstrip("=")
        candidates = [col for col in cols if col == bare or col.endswith(bare)]
        if candidates:
            matched.append(candidates[0])

    if not matched:
        return

    for combo in df_sweep[matched].drop_duplicates().itertuples(index=False):
        values: Dict[str, Any] = dict(zip(matched, combo))
        mask = pd.Series([True] * len(df_sweep), dtype=bool)
        for col, value in values.items():
            mask = mask & (df_sweep[col] == value)
        indices = mask[mask].index.tolist()
        filtered_results = [result_list[i] for i in indices]
        filtered_df = df_sweep.iloc[indices].reset_index(drop=True)
        label = "_".join(f"{col.rstrip().rsplit('.', 1)[-1]}={value}" for col, value in values.items())
        yield label, filtered_results, filtered_df


def override_sweep_key(base_sweep: Dict[str, List[Any]], key: str, values: List[Any]) -> Dict[str, List[Any]]:
    sweep = dict(base_sweep)
    sweep[key] = list(values)
    return sweep


def label_value(labels: Labels, key: str) -> Optional[str]:
    for label_key, value in labels:
        if label_key == key:
            return str(value)
    return None


_LABEL_STAGE_PREFIXES = (
    "conversion",
    "train_conversion",
    "recon_conversion",
    "preprocess_train",
    "preprocess_recon",
    "train",
)


def _bare_key(key: str) -> str:
    return key.lstrip("+~").lstrip("=")


def _split_stage_prefix(key: str) -> Tuple[Optional[str], str]:
    """Split ``preprocess_recon.++preprocess.mask_accelerations`` into
    ``("preprocess_recon", "++preprocess.mask_accelerations")``.  Keys that do
    not start with a known stage prefix are returned unsplit."""
    stage, _, rest = key.partition(".")
    if rest and stage in _LABEL_STAGE_PREFIXES:
        return stage, rest
    return None, key


def label_key_is_stable(label_key: str, stable_keys: List[str]) -> bool:
    """Whether *label_key* refers to one of the HP-sharing *stable_keys*."""
    if not stable_keys:
        return False
    label_stage, label_rest = _split_stage_prefix(label_key)
    label_bare = _bare_key(label_rest)
    label_forms = {label_rest, label_bare, label_bare.rsplit(".", 1)[-1]}
    for stable_key in stable_keys:
        stable_stage, stable_rest = _split_stage_prefix(stable_key)
        if stable_stage is not None and stable_stage != label_stage:
            continue
        stable_bare = _bare_key(stable_rest)
        if label_forms & {stable_rest, stable_bare, stable_bare.rsplit(".", 1)[-1]}:
            return True
    return False


def _representative_score(labels: Labels, representative: Dict[str, Any]) -> int:
    """How many *representative* choices this member's labels satisfy."""
    return sum(
        1
        for key, value in labels
        for rep_key, rep_value in representative.items()
        if label_key_is_stable(key, [rep_key]) and str(value) == str(rep_value)
    )


def _warn_unmatched_representatives(items: List[Any], representative: Dict[str, Any]) -> None:
    for rep_key, rep_value in representative.items():
        swept = {
            str(value)
            for _, labels in items
            for key, value in labels
            if label_key_is_stable(key, [rep_key])
        }
        if swept and str(rep_value) not in swept:
            logging.warning(
                "hp_sharing.representative %s=%s matches none of the swept values %s - "
                "tuning falls back to the first one.",
                rep_key, rep_value, sorted(swept),
            )


def _group_by_stable_keys(
    items: List[Any],
    stable_keys: List[str],
    representative: Optional[Dict[str, Any]] = None,
) -> Dict[tuple, List[Any]]:
    """Group ``(payload, labels)`` items by their non-stable label dimensions."""
    groups: Dict[tuple, List[Any]] = {}
    for item in items:
        labels = item[1]
        tuning_dims = tuple(sorted([
            (key, str(value)) for key, value in labels
            if not label_key_is_stable(key, stable_keys)
        ]))
        groups.setdefault(tuning_dims, []).append(item)
    if not representative:
        return groups
    _warn_unmatched_representatives(items, representative)
    # sorted() is stable, so ties keep the original sweep order.
    return {
        key: sorted(members, key=lambda member: -_representative_score(member[1], representative))
        for key, members in groups.items()
    }


def group_train_futures(
    train_futures: List[LabeledFuture],
    stable_keys: List[str],
    representative: Optional[Dict[str, Any]] = None,
) -> Dict[tuple, List[LabeledFuture]]:
    """Group trained models that differ only in HP-sharing keys (e.g. ``+exps``)."""
    return _group_by_stable_keys(train_futures, stable_keys, representative)


def group_recon_outputs(
    recon_outputs: List[DatasetOutput],
    stable_keys: List[str],
    representative: Optional[Dict[str, Any]] = None,
) -> Dict[tuple, List[DatasetOutput]]:
    """Group recon preprocessing outputs that differ only in HP-sharing keys
    (e.g. ``++preprocess.mask_accelerations``)."""
    return _group_by_stable_keys(recon_outputs, stable_keys, representative)

