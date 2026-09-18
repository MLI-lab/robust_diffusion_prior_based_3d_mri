"""Per-trained-model prior-mesh scaling for resolution-shift flows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hydra import compose
from src.prefect.hydra_lock import locked_hydra_initialize
from omegaconf import OmegaConf

from src.flows.flow_utils.tuned_recon.types import Labels
from src.prefect.sweeping import get_hydra_overwrite_list

_INTERP_KEY = "preprocess.target_interpolate_by_factor"


def _stage_overrides_from_labels(labels: Labels, prefix: str) -> List[str]:
    """Turn ``[('preprocess_train.+exps', 'stanford_3d/1mm')]`` back into overrides."""
    combo = [
        (key.split(".", 1)[1], value)
        for key, value in labels
        if key.startswith(f"{prefix}.")
    ]
    return get_hydra_overwrite_list(combo) if combo else []


def _interp_factor(stage: Dict[str, Any], labels: Labels, prefix: str) -> Optional[float]:
    """Compose one preprocessing stage and read its interpolation factor."""
    with locked_hydra_initialize(config_path=stage["config_path"], version_base="1.2"):
        cfg = compose(
            config_name=stage["config_name"],
            overrides=list(stage["overrides"]) + _stage_overrides_from_labels(labels, prefix),
        )
    value = OmegaConf.select(cfg, _INTERP_KEY, default=None)
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prior_mesh_scale_overrides(
    cfg: Dict[str, Any],
    train_labels: Labels,
    recon_labels: Labels,
    logger: Any = None,
) -> List[str]:
    """``++representation.mesh_prior_scale_factor=<scale>`` for one model/data pair."""
    scale_cfg = cfg["reconstruction"].get("prior_mesh_scale") or {}
    if not scale_cfg.get("enabled", False):
        return []

    train_stage = cfg["data"]["preprocess"].get("train")
    recon_stage = cfg["data"]["preprocess"]["recon"]
    if train_stage is None:
        return []
    train_factor = _interp_factor(train_stage, train_labels, "preprocess_train")
    recon_factor = _interp_factor(recon_stage, recon_labels, "preprocess_recon")

    if train_factor is None or recon_factor is None or recon_factor <= 0.0:
        if logger is not None:
            logger.warning(
                "prior_mesh_scale: no single interpolation factor for this pair "
                "(train=%s, recon=%s); leaving representation.mesh_prior_scale_factor "
                "as the recon config declares it.",
                train_factor,
                recon_factor,
            )
        return []

    scale = train_factor / recon_factor
    if logger is not None:
        logger.info(
            "prior_mesh_scale: train factor %s / recon factor %s -> mesh_prior_scale_factor=%s",
            train_factor,
            recon_factor,
            scale,
        )
    return [f"++representation.mesh_prior_scale_factor={scale}"]
