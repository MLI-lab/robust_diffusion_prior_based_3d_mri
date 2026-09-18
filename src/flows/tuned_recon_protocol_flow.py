from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf
from prefect import flow
from prefect.logging import get_run_logger

from src.flows.flow_utils.tuned_recon.config import normalize_protocol_cfg
from src.flows.flow_utils.tuned_recon.protocols import run_index_holdout, run_standard
from src.flows.flow_utils.tuned_recon.smoke import apply_smoke_overrides
from src.utils.wandb_utils import gather_flow_infos_for_wandb


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


@flow(name="tuned_recon_protocol_flow")
def tuned_recon_protocol_flow(
    cfg: Dict[str, Any] | DictConfig,
    wandb_cfg: DictConfig,
    local_cache_path: str,
    bart_path: str | None = None,
) -> None:
    logger = get_run_logger()
    wandb_params = gather_flow_infos_for_wandb()
    protocol_cfg = apply_smoke_overrides(normalize_protocol_cfg(_plain(cfg)))
    mode = protocol_cfg["protocol"]["mode"]

    logger.info("Starting tuned reconstruction protocol flow in mode=%s", mode)
    if mode == "standard":
        run_standard(protocol_cfg, wandb_cfg, local_cache_path, bart_path, logger, wandb_params)
    elif mode == "index_holdout":
        run_index_holdout(protocol_cfg, wandb_cfg, local_cache_path, bart_path, logger, wandb_params)
    else:
        raise ValueError(f"Unknown tuned reconstruction protocol mode: {mode}")
    logger.info("tuned_recon_protocol_flow completed.")

