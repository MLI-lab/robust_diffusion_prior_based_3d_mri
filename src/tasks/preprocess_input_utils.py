from __future__ import annotations

from typing import Optional

from omegaconf import OmegaConf

from src.prefect.caching import CacheableDictConfig


def input_cfg_from_converted_path(
    input_cfg: CacheableDictConfig,
    mode: str,
    converted_dataset_path: str,
    fold: Optional[str],
) -> CacheableDictConfig:
    """Return a copied input config that points at an upstream converted fold."""
    cfg = OmegaConf.create(OmegaConf.to_container(input_cfg.cfg, resolve=False))
    fold_name = str(fold or ("test_val" if mode in ("test", "recon_observation", "recon") else "train"))
    if mode in ("test", "recon_observation", "recon"):
        cfg.data_path_recon = f"{converted_dataset_path}/{fold_name}"
        cfg.data_path_sensmaps_recon = None
    else:
        cfg.data_path_train = f"{converted_dataset_path}/{fold_name}"
        cfg.data_path_val = None
        cfg.data_path_sensmaps_train = None
        cfg.data_path_sensmaps_val = None
    return CacheableDictConfig(cfg=cfg)
