from __future__ import annotations

from typing import Any, Dict, Optional

from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE

from src.prefect.caching import CacheableDictConfig
from src.utils.wandb_utils import WandbParamsTask
from src.tasks.preprocess_input_utils import input_cfg_from_converted_path


@task(
    cache_policy=TASK_SOURCE + INPUTS - "wandb_params_task" - "bart_path",
    name="Preprocess Recon Dataset Task",
    tags=["preprocess", "recon-dataset"],
    version="1.0",
    retries=0,
)
def preprocess_recon_dataset_task(
    input_cfg: CacheableDictConfig,
    local_cache_path: str,
    output_cache_subfolder: str = "preprocess_cache",
    preprocess_cfg: Optional[CacheableDictConfig] = None,
    dataset_cfg: Optional[CacheableDictConfig] = None,
    bart_path: Optional[str] = None,
    wandb_params_task: Optional[WandbParamsTask] = None,
    converted_dataset_path: Optional[str] = None,
    fold: str = "test_val",
    mode: Optional[str] = None,
    **unused: Any,
) -> Dict[str, Any]:
    if preprocess_cfg is None or dataset_cfg is None:
        raise ValueError("preprocess_recon_dataset_task requires preprocess_cfg and dataset_cfg.")
    if converted_dataset_path is not None:
        input_cfg = input_cfg_from_converted_path(
            input_cfg=input_cfg,
            mode="recon_observation",
            converted_dataset_path=converted_dataset_path,
            fold=fold,
        )

    from src.tasks.preprocess_impl.recon_preprocess import run_recon_observation

    result = run_recon_observation(
        input_cfg=input_cfg,
        output_cache_subfolder=output_cache_subfolder,
        preprocess_cfg=preprocess_cfg,
        dataset_cfg=dataset_cfg,
        local_cache_path=local_cache_path,
        bart_path=bart_path,
    )

    from src.tasks.preprocess_visualization import log_preprocess_outputs_to_wandb

    log_preprocess_outputs_to_wandb(
        output_paths=result,
        visualize_cfg=getattr(preprocess_cfg.cfg, "visualize", None),
        wandb_params_task=wandb_params_task,
        key_candidates=[
            str(getattr(preprocess_cfg.cfg, "output_recons_key", "reconstruction_mvue")),
            str(getattr(preprocess_cfg.cfg, "output_pseudoinverse_key", "pseudoinverse")),
            str(getattr(preprocess_cfg.cfg, "output_sensmaps_key", "sens_maps")),
            "kspace",
        ],
        name_aux="preprocess_recon",
        summary_stage="preprocess_recon",
        dataset_name="preprocess_recon",
        counts=None,
    )
    return result
