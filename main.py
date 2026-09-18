import numpy
import hydra
from omegaconf import DictConfig, OmegaConf

from hydra._internal.hydra import Hydra
from src.flows.flow_resolver import resolve_flow
from src.prefect.api_helper import PREFECT_API_URL
import os
from src.prefect.setup import get_and_setup_task_runner, get_caching_storage
from src.utils.path_utils import get_path_by_cluster_name_and_hierarchy

def _cluster_name_from_cfg_or_env(cfg: DictConfig) -> str:
    cluster_name_env = os.environ.get("CLUSTER_NAME")
    if cluster_name_env:
        return cluster_name_env
    if OmegaConf.is_missing(cfg, "cluster_name"):
        raise ValueError(
            "cluster_name is required. Set it in Hydra, e.g. cluster_name=node, "
            "or export CLUSTER_NAME."
        )
    return cfg.cluster_name


def _resolve_infra_path(value, cfg: DictConfig):
    cluster_name = _cluster_name_from_cfg_or_env(cfg)
    try:
        return get_path_by_cluster_name_and_hierarchy(
            value,
            cluster_hierarchy=cfg.cluster_hierarchy,
            cluster_name=cluster_name,
        )
    except ValueError:
        short_name = str(cluster_name).rsplit("_", 1)[-1]
        if short_name == str(cluster_name):
            raise
        return get_path_by_cluster_name_and_hierarchy(
            value,
            cluster_hierarchy=cfg.cluster_hierarchy,
            cluster_name=short_name,
        )


@hydra.main(config_path='hydra', config_name='config', version_base='1.2')
def coordinator(cfg : DictConfig) -> None:
    os.environ["RAY_RUNTIME_ENV_IGNORE_GITIGNORE"] = "1"
    resolve_flow(
        options={"task_runner": get_and_setup_task_runner(cfg.task_runner), "result_storage": get_caching_storage()},
        wandb_cfg=cfg.wandb,
        local_cache_path=_resolve_infra_path(cfg.local_cache_path, cfg),
        bart_path=_resolve_infra_path(cfg.bart_path, cfg),
        **cfg.flow,
    )()

if __name__ == "__main__":
    coordinator() 