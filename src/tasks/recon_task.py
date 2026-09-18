from itertools import islice
from functools import partial

from typing import Optional, List, Any, Dict, Sequence

import os
import math
import re

import torch.nn as nn

import logging

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import wandb
from src.prefect.wandb_lock import locked_wandb_init
import numpy as np

from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.problem_trafos.trafo_resolver import (
    get_fwd_trafo, get_dataset_trafo,
    get_target_trafo, get_prior_trafo)

from src.diffmodels.sampler.sampler_resolver import get_sampler
from src.diffmodels.sde import SDE
from src.reconstruction.posterior_sampling.conditioner_resolver import get_conditioning_method
from src.representations.representation_resolver import (
    get_representation,
    get_mesh,
    get_mesh_from_model,
    get_slice_method
)
from src.datasets import dataset_resolver

from src.reconstruction.variational.fit import fit
from src.reconstruction.utils.pass_through import ScoreWithIdentityGradWrapper
from src.utils.nfe_counter import NfeCountingScoreWrapper
from src.reconstruction.variational.var_objectives import get_variational_objective
from src.utils.wandb_utils import wandb_kwargs_via_cfg
from src.utils.device_utils import get_all_devices
from src.tasks.dataset_pipeline_utils import concrete_path_resolver
from src.problem_trafos.utils.bart_utils import import_bart
from src.problem_trafos.prior_target_trafo.stacked_prior_trafo_adapter import (
    StackedPriorTrafoAdapter,
)

from src.diffmodels.diffmodels_resolver import load_score_model, create_model
from src.diffmodels import load_sde_model
from src.sample_logger.sample_logger_resolver import get_sample_logger
from src.representations.fixed_grid_representation import FixedGridRepresentation

import hydra

from prefect import task
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE

from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask

from src.prefect.setup import StorageSettings, StoragePath, load_file_system
from src.prefect.caching import CacheableDict, CacheableDictConfig

# here we must now use the pretrained path
from src.prefect.setup import download_directory_to_temp_on_enter, switch_dir_and_upload_directory_on_exit, retry_s3_transfer

import tempfile
from datetime import datetime


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _normalize_bool_sequence(value: Any) -> List[bool]:
    if isinstance(value, str):
        value = value.split(",")
    return [_as_bool(item) for item in value]


_MIN_NFE_SLICE_BUDGET_PER_DIRECTION = 5
_CHAINED_VARIATIONAL_PRIOR_LOSSES = {"infusion_loss", "sampled_prior_loss"}


def _normalize_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    return OmegaConf.select(cfg, key, default=default)


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    value = _cfg_select(cfg, key, default=default)
    if value is None:
        return int(default)
    return int(value)


def _sequence_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len([item for item in value.split(",") if item.strip()])
    try:
        return len(value)
    except TypeError:
        return 1


def _has_score_variational_regularizer(reconstruction_cfg: DictConfig) -> bool:
    regularization_name = _normalize_name(
        _cfg_select(reconstruction_cfg, "variational.regularization.name", default=None)
    )
    return regularization_name in {"diffusion", "diffusion_sampled", "infusion"}


def _enabled_prior_reg_directions(reconstruction_cfg: DictConfig) -> int:
    prior_reg_cfg = _cfg_select(reconstruction_cfg, "slice_methods.prior_reg", default=None)
    if prior_reg_cfg is None:
        return 0
    slice_enabled = _cfg_select(prior_reg_cfg, "slice_enabled", default=[True, True, True])
    return sum(1 for enabled in _normalize_bool_sequence(slice_enabled) if enabled)


def _variational_nfe_per_slice(reconstruction_cfg: DictConfig) -> int:
    loss_cfg_key = "variational.regularization.loss_cfg"
    loss_name = _normalize_name(_cfg_select(reconstruction_cfg, f"{loss_cfg_key}.name", default=None))
    repetition = max(1, _cfg_int(reconstruction_cfg, f"{loss_cfg_key}.repetition", default=1))
    if loss_name in _CHAINED_VARIATIONAL_PRIOR_LOSSES:
        prior_num_steps = max(1, _cfg_int(reconstruction_cfg, f"{loss_cfg_key}.prior_num_steps", default=1))
        return repetition * prior_num_steps
    return repetition


def _resolve_variational_nfe_choice(
    *,
    limit_nfes: int,
    iterations: int,
    n_reg_calls: int,
    n_dirs: int,
    nfe_per_slice: int,
    min_slices_per_direction: int = _MIN_NFE_SLICE_BUDGET_PER_DIRECTION,
) -> Dict[str, Any]:
    iterations = max(1, int(iterations))
    n_reg_calls = max(1, int(n_reg_calls))
    n_dirs = max(1, int(n_dirs))
    nfe_per_slice = max(1, int(nfe_per_slice))
    min_slices_per_direction = max(1, int(min_slices_per_direction))

    min_nfes_per_iteration = n_reg_calls * n_dirs * nfe_per_slice * min_slices_per_direction
    if iterations * min_nfes_per_iteration > limit_nfes:
        resolved_iterations = max(1, int(limit_nfes) // min_nfes_per_iteration)
        resolved_slice_budget = min_slices_per_direction
    else:
        resolved_iterations = iterations
        denom = resolved_iterations * n_reg_calls * n_dirs * nfe_per_slice
        resolved_slice_budget = max(min_slices_per_direction, int(limit_nfes) // denom)

    estimated_nfes = resolved_iterations * n_reg_calls * n_dirs * nfe_per_slice * resolved_slice_budget
    return {
        "iterations": resolved_iterations,
        "slice_budget": resolved_slice_budget,
        "estimated_nfes": estimated_nfes,
        "feasible": estimated_nfes <= int(limit_nfes),
        "min_slices_per_direction": min_slices_per_direction,
    }


def _sampling_avg_nfe_per_step(diffmodels_cfg: DictConfig, prior_shape: Any) -> float:
    cycling = _as_bool(_cfg_select(diffmodels_cfg, "sampler.cycling", default=False))
    sampling_in_3d = _as_bool(_cfg_select(diffmodels_cfg, "sampler.sampling_in_3d", default=False))
    if sampling_in_3d and cycling and len(prior_shape) >= 4:
        return float(prior_shape[0] + prior_shape[2] + prior_shape[3]) / 3.0
    return float(prior_shape[0]) if len(prior_shape) >= 1 else 1.0


def _resolve_sampling_steps_for_nfe_budget(
    *,
    limit_nfes: int,
    avg_nfe_per_step: float,
    sde_max: int,
) -> int:
    return max(1, min(int(limit_nfes // max(1.0, avg_nfe_per_step)), int(sde_max)))


def _apply_nfe_budget_choice(
    *,
    reconstruction_cfg: DictConfig,
    diffmodels_cfg: DictConfig,
    prior_shape: Any,
    sde: Optional[SDE],
    min_slices_per_direction: int = _MIN_NFE_SLICE_BUDGET_PER_DIRECTION,
) -> Dict[str, Any]:
    """Resolve NFE-limited variational slice/iteration and sampler-step choices."""

    limit_nfes = _cfg_int(reconstruction_cfg, "limit_nfes", default=-1)
    if limit_nfes <= 0 or not _as_bool(_cfg_select(reconstruction_cfg, "use_score_regularisation", default=False)):
        return {}

    method = _normalize_name(_cfg_select(reconstruction_cfg, "method", default=None))
    summary: Dict[str, Any] = {"limit_nfes": int(limit_nfes)}
    avg_nfe_per_step = _sampling_avg_nfe_per_step(diffmodels_cfg, prior_shape)
    sde_max = int(sde.num_steps) if sde is not None else 1000

    if method in {"variational", "hybrid_variational_sampling"} and _has_score_variational_regularizer(reconstruction_cfg):
        prior_reg_cfg = _cfg_select(reconstruction_cfg, "slice_methods.prior_reg", default=None)
        if prior_reg_cfg is None:
            logging.warning(
                "[limit_nfes=%s] Skipping variational NFE budget resolution because slice_methods.prior_reg is None.",
                limit_nfes,
            )
        else:
            configured_iterations = _cfg_int(
                reconstruction_cfg,
                "variational.fitting.optimizer.iterations",
                default=1,
            )
            configured_slice_budget = _cfg_int(prior_reg_cfg, "slice_budget", default=1)
            n_dirs = max(1, _enabled_prior_reg_directions(reconstruction_cfg))
            reg_steps = _cfg_select(
                reconstruction_cfg,
                "variational.fitting.optimizer.gradient_acc_steps_prior_reg",
                default=[0],
            )
            n_reg_calls = max(1, _sequence_len(reg_steps))
            nfe_per_slice = _variational_nfe_per_slice(reconstruction_cfg)
            variational_limit_nfes = limit_nfes
            if method == "hybrid_variational_sampling":
                configured_start_timestep = _cfg_int(reconstruction_cfg, "hybrid_refinement.start_timestep", default=20)
                configured_refinement_steps = min(configured_start_timestep + 1, sde_max)
                configured_refinement_nfes = int(math.ceil(configured_refinement_steps * avg_nfe_per_step))
                min_nfes_per_iteration = (
                    max(1, n_reg_calls)
                    * max(1, n_dirs)
                    * max(1, nfe_per_slice)
                    * max(1, int(min_slices_per_direction))
                )
                configured_min_variational_nfes = configured_iterations * min_nfes_per_iteration
                if configured_min_variational_nfes < limit_nfes:
                    remaining_after_min_variational = limit_nfes - configured_min_variational_nfes
                    if remaining_after_min_variational >= configured_refinement_nfes:
                        variational_limit_nfes = limit_nfes - configured_refinement_nfes
                    else:
                        variational_limit_nfes = configured_min_variational_nfes

            variational_choice = _resolve_variational_nfe_choice(
                limit_nfes=variational_limit_nfes,
                iterations=configured_iterations,
                n_reg_calls=n_reg_calls,
                n_dirs=n_dirs,
                nfe_per_slice=nfe_per_slice,
                min_slices_per_direction=min_slices_per_direction,
            )

            reconstruction_cfg.variational.fitting.optimizer.iterations = variational_choice["iterations"]
            reconstruction_cfg.slice_methods.prior_reg.slice_budget = variational_choice["slice_budget"]
            logging.info(
                "[limit_nfes=%s] varrecon: iterations %s -> %s, slice_budget %s -> %s "
                "(n_dirs=%s, reg_calls=%s, nfe_per_slice=%s, min_slices_per_direction=%s, estimated_nfes=%s)",
                limit_nfes,
                configured_iterations,
                variational_choice["iterations"],
                configured_slice_budget,
                variational_choice["slice_budget"],
                n_dirs,
                n_reg_calls,
                nfe_per_slice,
                variational_choice["min_slices_per_direction"],
                variational_choice["estimated_nfes"],
            )
            if not variational_choice["feasible"]:
                logging.warning(
                    "[limit_nfes=%s] Minimum stable variational setting still estimates %s NFEs.",
                    limit_nfes,
                    variational_choice["estimated_nfes"],
                )

            summary.update(
                {
                    "resolved_slice_budget": variational_choice["slice_budget"],
                    "resolved_variational_iterations": variational_choice["iterations"],
                    "estimated_variational_nfes": variational_choice["estimated_nfes"],
                    "variational_budget_nfes": variational_limit_nfes,
                    "variational_nfe_per_slice": nfe_per_slice,
                    "min_slices_per_direction": variational_choice["min_slices_per_direction"],
                    "nfe_budget_feasible": variational_choice["feasible"],
                }
            )

    if method == "sampling":
        configured_num_steps = _cfg_int(diffmodels_cfg, "sampler.num_steps", default=1)
        resolved_num_steps = _resolve_sampling_steps_for_nfe_budget(
            limit_nfes=limit_nfes,
            avg_nfe_per_step=avg_nfe_per_step,
            sde_max=sde_max,
        )
        diffmodels_cfg.sampler.num_steps = resolved_num_steps
        estimated_sampling_nfes = int(math.ceil(resolved_num_steps * avg_nfe_per_step))
        logging.info(
            "[limit_nfes=%s] sampling: num_steps %s -> %s (avg_nfe/step=%.1f, sde_max=%s)",
            limit_nfes,
            configured_num_steps,
            resolved_num_steps,
            avg_nfe_per_step,
            sde_max,
        )
        summary.update(
            {
                "resolved_num_steps": resolved_num_steps,
                "estimated_sampling_nfes": estimated_sampling_nfes,
                "estimated_total_nfes": estimated_sampling_nfes,
                "nfe_budget_feasible": estimated_sampling_nfes <= limit_nfes,
            }
        )

    elif method == "hybrid_variational_sampling":
        configured_start_timestep = _cfg_int(reconstruction_cfg, "hybrid_refinement.start_timestep", default=20)
        configured_refinement_steps = configured_start_timestep + 1
        variational_nfes = int(summary.get("estimated_variational_nfes", 0))
        remaining_nfes = max(0, limit_nfes - variational_nfes)
        allowed_refinement_steps = _resolve_sampling_steps_for_nfe_budget(
            limit_nfes=remaining_nfes,
            avg_nfe_per_step=avg_nfe_per_step,
            sde_max=sde_max,
        )
        resolved_refinement_steps = max(
            1,
            min(configured_refinement_steps, allowed_refinement_steps),
        )
        resolved_start_timestep = resolved_refinement_steps - 1
        OmegaConf.update(
            reconstruction_cfg,
            "hybrid_refinement.start_timestep",
            resolved_start_timestep,
            merge=False,
            force_add=True,
        )
        estimated_refinement_nfes = int(math.ceil(resolved_refinement_steps * avg_nfe_per_step))
        logging.info(
            "[limit_nfes=%s] hybrid refinement: start_timestep %s -> %s "
            "(steps=%s, remaining_nfes=%s, avg_nfe/step=%.1f)",
            limit_nfes,
            configured_start_timestep,
            resolved_start_timestep,
            resolved_refinement_steps,
            remaining_nfes,
            avg_nfe_per_step,
        )
        estimated_total_nfes = variational_nfes + estimated_refinement_nfes
        if estimated_total_nfes > limit_nfes:
            logging.warning(
                "[limit_nfes=%s] Minimum hybrid setting still estimates %s NFEs.",
                limit_nfes,
                estimated_total_nfes,
            )
        summary.update(
            {
                "resolved_hybrid_start_timestep": resolved_start_timestep,
                "resolved_hybrid_refinement_steps": resolved_refinement_steps,
                "estimated_hybrid_refinement_nfes": estimated_refinement_nfes,
                "estimated_total_nfes": estimated_total_nfes,
                "nfe_budget_feasible": bool(summary.get("nfe_budget_feasible", True)) and estimated_total_nfes <= limit_nfes,
            }
        )

    return summary


def _extract_checkpoint_subfolder_names(filesystem: Any, fs_folder: str) -> List[str]:
    def _normalize_entries(entries: Any) -> List[str]:
        names: List[str] = []
        if entries is None:
            return names
        for entry in entries:
            raw_name = None
            if isinstance(entry, str):
                raw_name = entry
            elif isinstance(entry, dict):
                raw_name = (
                    entry.get("name")
                    or entry.get("path")
                    or entry.get("Key")
                    or entry.get("key")
                )
            if raw_name is None:
                continue
            basename = os.path.basename(str(raw_name).rstrip("/"))
            if basename:
                names.append(basename)
        return names

    discovered: List[str] = []

    for method_name in ("list_directory", "listdir", "ls"):
        method = getattr(filesystem, method_name, None)
        if callable(method):
            try:
                discovered.extend(_normalize_entries(method(fs_folder)))
            except Exception:
                pass

    backend_fs = getattr(filesystem, "filesystem", None)
    if backend_fs is not None:
        for method_name in ("listdir", "ls"):
            method = getattr(backend_fs, method_name, None)
            if callable(method):
                try:
                    discovered.extend(_normalize_entries(method(fs_folder)))
                except Exception:
                    pass

    seen = set()
    unique: List[str] = []
    for name in discovered:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _scaled_prior_matrix_size(
    data_matrix_size: Sequence[int],
    scale_factor: Any,
) -> List[int]:
    """Matrix size of the prior mesh, as the data mesh scaled by ``scale_factor``."""
    if isinstance(scale_factor, (int, float)) and not isinstance(scale_factor, bool):
        factors = [float(scale_factor)] * len(data_matrix_size)
    else:
        factors = [float(value) for value in scale_factor]
        if len(factors) != len(data_matrix_size):
            raise ValueError(
                "representation.mesh_prior_scale_factor must be a scalar or hold one value "
                f"per axis; got {factors} for a data mesh of shape {tuple(data_matrix_size)}."
            )
    if any(factor <= 0.0 for factor in factors):
        raise ValueError(
            f"representation.mesh_prior_scale_factor must be positive, got {factors}."
        )
    return [max(1, int(round(size * factor))) for size, factor in zip(data_matrix_size, factors)]


def _attrs_get_float(attrs: Any, key: str, default: float) -> float:
    if attrs is None or key not in attrs:
        return default
    value = attrs[key]
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return _attrs_get_float({key: value[0]}, key, default)
    return float(value)


def _forward_diffuse_from_x0(
    x0: torch.Tensor,
    sde: SDE,
    start_timestep: int,
    noise_mode: str = "forward_marginal",
) -> torch.Tensor:
    if noise_mode != "forward_marginal":
        raise ValueError(f"Unsupported hybrid_refinement.noise_mode={noise_mode!r}.")

    start_timestep = int(start_timestep)
    if start_timestep < 0 or start_timestep >= int(sde.num_steps):
        raise ValueError(
            f"hybrid_refinement.start_timestep must satisfy 0 <= start_timestep < {sde.num_steps}, "
            f"got {start_timestep}."
        )

    t = torch.full((1,), start_timestep, device=x0.device, dtype=torch.long)
    mean = sde.marginal_prob_mean(t).to(device=x0.device, dtype=x0.dtype)
    std = sde.marginal_prob_std(t).to(device=x0.device, dtype=x0.dtype)
    view_shape = (1,) + (1,) * (x0.ndim - 1)
    return x0 * mean.reshape(view_shape) + torch.randn_like(x0) * std.reshape(view_shape)


def _resolve_model_subfolder_candidates(
    diffmodels_cfg: CacheableDictConfig,
    diffmodels_train_cfg: DictConfig,
    filesystem: Any,
    pretrained_fs_folder: str,
) -> List[str]:
    model_load_nr = OmegaConf.select(diffmodels_cfg.cfg, "model_load_nr", default=None)
    if model_load_nr is not None:
        model_load_nr = int(model_load_nr)
        if model_load_nr >= 0:
            raise ValueError(f"diffmodels.model_load_nr must be <= -1, got {model_load_nr}.")

        if model_load_nr == -1:
            return ["final_model"]

        previous_rank = -model_load_nr - 1
        subfolders = _extract_checkpoint_subfolder_names(filesystem, pretrained_fs_folder)
        step_numbers = sorted(
            [
                int(m.group(1))
                for name in subfolders
                for m in [re.fullmatch(r"step_(\d+)", name)]
                if m is not None
            ],
            reverse=True,
        )

        if step_numbers:
            if previous_rank > len(step_numbers):
                raise ValueError(
                    f"Requested diffmodels.model_load_nr={model_load_nr}, but only "
                    f"{len(step_numbers)} step checkpoints are available: {step_numbers}."
                )
            selected_step = step_numbers[previous_rank - 1]
            return [f"step_{selected_step}", f"epoch_{selected_step}"]

        training_steps_cfg = OmegaConf.select(diffmodels_train_cfg, "train.training_steps", default=None)
        n_save_total = int(OmegaConf.select(diffmodels_train_cfg, "train.n_save_total", default=10))
        if training_steps_cfg is not None:
            max_steps = int(training_steps_cfg)
            save_every_n_steps = max(max_steps // max(n_save_total, 1), 1)
            expected_steps = sorted(
                set(list(range(save_every_n_steps, max_steps + 1, save_every_n_steps)) + [max_steps]),
                reverse=True,
            )
            if previous_rank > len(expected_steps):
                raise ValueError(
                    f"Requested diffmodels.model_load_nr={model_load_nr}, but fallback from "
                    f"train config yields only {len(expected_steps)} step checkpoints: {expected_steps}."
                )
            selected_step = expected_steps[previous_rank - 1]
            logging.warning(
                "Could not list checkpoint subfolders in storage. Falling back to train config "
                "derived step index: step_%s",
                selected_step,
            )
            return [f"step_{selected_step}", f"epoch_{selected_step}"]

        raise ValueError(
            "diffmodels.model_load_nr was set, but no step_* checkpoints could be discovered "
            "and train.training_steps is unavailable for fallback resolution."
        )

    legacy_model_from_epoch = int(OmegaConf.select(diffmodels_cfg.cfg, "model_from_epoch", default=-1))
    if legacy_model_from_epoch < 0:
        return ["final_model"]
    return [f"epoch_{legacy_model_from_epoch}", f"step_{legacy_model_from_epoch}"]

@task(cache_policy=INPUTS - "wandb_params_task" - "sweep_cfg", name="Recon Task", tags=["recon"], version="1.0", retries=2, retry_delay_seconds=[60, 300])
def recon_task(
    seed: int,
    problem_trafos_cfg : CacheableDictConfig,
    reconstruction_cfg : CacheableDictConfig,
    representation_cfg : CacheableDictConfig,
    sample_logger_cfg : CacheableDictConfig,
    dataset_cfg : CacheableDictConfig,
    diffmodels_cfg : CacheableDictConfig,
    storage_settings_cfg : CacheableDictConfig,
    wandb_params_task : WandbParamsTask,
    sample_idx: int = 0,
    pretrained_model_path: Optional[StoragePath] = None,
    sweep_cfg : Optional[CacheableDictConfig] = None,
    data_path_recon_override: Optional[Any] = None,
    local_cache_path: Optional[str] = None,
) -> CacheableDict:

    """
        Pretrained model  
    """
    devices = get_all_devices()
    device = devices[0]

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        wandb_kwargs = wandb_kwargs_for_prefect_task(wandb_params_task)
        
        # setting up the storage for configs and model checkpoints
        output_dir_in_fs = os.path.join(
            wandb_kwargs["config"]["flow_name"] + "_" + wandb_kwargs["config"]["flow_run_name"],
            str(datetime.now().strftime('%Y%m%d_%H%M%S')) + "_" + wandb_kwargs["name"]
        )
        storage_settings = StorageSettings(**storage_settings_cfg.cfg)
        filesystem = load_file_system(storage_settings)
        
        #  upload config files to filesystem
        OmegaConf.save(config=wandb_params_task.config, f=os.path.join(tmp_dir_name, "wandb_kwargs.yaml"))
        retry_s3_transfer(lambda: filesystem.put_directory(tmp_dir_name, output_dir_in_fs))

        switch_dir_and_upload_directory_on_exit_mgr = partial(
            switch_dir_and_upload_directory_on_exit,
            filesystem=filesystem,
            temp_folder=tmp_dir_name,
            fs_folder=output_dir_in_fs
        )

        with locked_wandb_init(**wandb_kwargs_for_prefect_task(wandb_params_task)):
            import time as _time
            _task_wall_t0 = _time.perf_counter()

            dtype = torch.get_default_dtype()

            # trafo: object to measurement data (e.g. complex fourier data or sinogram)
            fwd_trafo = get_fwd_trafo(**problem_trafos_cfg.cfg.fwd_trafo)
            # trafo: object to target data (e.g. magnitude images)
            target_trafo = get_target_trafo(**problem_trafos_cfg.cfg.target_trafo)
            # trafo: object to prior (e.g. magnitude images)
            prior_trafo = get_prior_trafo(**problem_trafos_cfg.cfg.prior_trafo)
            # trafo: dataset trafo (preprocessing loaded volumes)
            dataset_trafo = get_dataset_trafo(**problem_trafos_cfg.cfg.dataset_trafo,
                provide_pseudoinverse=True, provide_measurement=True, device=device)

            # loading the datasets
            _t0 = _time.perf_counter()
            path_resolver = concrete_path_resolver(local_cache_path)
            dataset_kwargs = OmegaConf.to_container(dataset_cfg.cfg, resolve=True)
            if data_path_recon_override is not None:
                dataset_kwargs["data_path_recon"] = data_path_recon_override
                dataset_kwargs["data_path_sensmaps_recon"] = None
            dataset = dataset_resolver.get_dataset(
                **dataset_kwargs, dataset_trafo=dataset_trafo, path_resolver=path_resolver)
            logging.info(f"[TIMING] get_dataset (incl. calc_sensmap_files): {_time.perf_counter()-_t0:.1f}s")
            # loading score model and sde
            score : Optional[nn.Module] = None
            sde : Optional[SDE] = None
            nfe_counter: Optional[NfeCountingScoreWrapper] = None
            model_in_channels: Optional[int] = None
            _saved_stack_num_slices: Optional[int] = None
            if reconstruction_cfg.cfg.use_score_regularisation:

                assert pretrained_model_path is not None, "pretrained_model_path must be provided when using score regularization."

                with tempfile.TemporaryDirectory() as tmp_pretrained_dir_name:
                    filesystem_pretrained = load_file_system(pretrained_model_path.storage_settings)

                    download_directory_to_temp_on_enter_mgr = partial(
                        download_directory_to_temp_on_enter,
                        filesystem=filesystem_pretrained,
                        temp_folder=tmp_pretrained_dir_name,
                        fs_folder=pretrained_model_path.storage_path
                    )

                    # download configs
                    _t1 = _time.perf_counter()
                    with download_directory_to_temp_on_enter_mgr(ex_subfolder="configs"):
                        diffmodels_train_cfg = OmegaConf.load("diffmodels_cfg.yaml")
                        assert isinstance(diffmodels_train_cfg, DictConfig), "diffmodels_train_cfg is not a DictConfig"
                        sde = load_sde_model(**diffmodels_train_cfg.sde) 
                        score = create_model(**dict(diffmodels_train_cfg.arch), arch_cfg=None).to(device)
                        model_in_channels = int(
                            OmegaConf.select(
                                diffmodels_train_cfg,
                                "arch.params.in_channels",
                                default=OmegaConf.select(diffmodels_train_cfg, "arch.in_channels", default=2),
                            )
                        )
                        _prior_trafo_cfg = None
                        if os.path.exists("prior_trafo_cfg.yaml"):
                            _prior_trafo_saved = OmegaConf.load("prior_trafo_cfg.yaml")
                            _prior_trafo_cfg = OmegaConf.to_container(_prior_trafo_saved, resolve=True)
                        _recon_target_type = OmegaConf.select(
                            problem_trafos_cfg.cfg, "dataset_trafo.target_type", default=None
                        )
                        if model_in_channels == 1 and _recon_target_type in (None, "rss"):
                            if _prior_trafo_cfg is None:
                                _prior_trafo_cfg = OmegaConf.to_container(
                                    OmegaConf.create(dict(problem_trafos_cfg.cfg.prior_trafo)), resolve=True
                                )
                            _prior_trafo_cfg.update(
                                {
                                    "magnitude_enabled": True,
                                    "move_axis": None,
                                    "stack_channel_into_batchdim": True,
                                    "stack_z_into_batchdim": False,
                                    "collapse_singleton_complex_channel": False,
                                }
                            )
                            prior_trafo = get_prior_trafo(**_prior_trafo_cfg)
                            logging.info(
                                "Loaded 1-channel RSS score model; using recon-time magnitude prior_trafo "
                                "so complex reconstruction variables are converted with complex_abs."
                            )
                        elif _prior_trafo_cfg is not None:
                            prior_trafo = get_prior_trafo(**_prior_trafo_cfg)
                            logging.info(
                                f"prior_trafo re-instantiated from saved prior_trafo_cfg.yaml "
                                f"(name={getattr(_prior_trafo_saved, 'name', '?')})"
                            )
                        _saved_stack_num_slices: Optional[int] = None
                        if os.path.exists("stack_num_slices.txt"):
                            with open("stack_num_slices.txt") as _f:
                                _saved_stack_num_slices = int(_f.read().strip())
                            logging.info("Loaded stack_num_slices=%s from saved stack_num_slices.txt", _saved_stack_num_slices)
                    logging.info(f"[TIMING] download configs + create model: {_time.perf_counter()-_t1:.1f}s")

                    # determine the path we want to load the model from
                    subfolder_candidates = _resolve_model_subfolder_candidates(
                        diffmodels_cfg=diffmodels_cfg,
                        diffmodels_train_cfg=diffmodels_train_cfg,
                        filesystem=filesystem_pretrained,
                        pretrained_fs_folder=pretrained_model_path.storage_path,
                    )
                    model_use_ema = OmegaConf.select(diffmodels_cfg.cfg, "model_use_ema", default=None)
                    if model_use_ema is None:
                        model_use_ema = OmegaConf.select(diffmodels_cfg.cfg, "use_ema", default=None)
                    if model_use_ema is None:
                        model_use_ema = OmegaConf.select(diffmodels_train_cfg, "train.use_ema", default=False)
                    model_use_ema = bool(model_use_ema)
                    _t2 = _time.perf_counter()
                    loaded_subfolder_name = None
                    last_exception = None
                    for subfolder_name in subfolder_candidates:
                        try:
                            with download_directory_to_temp_on_enter_mgr(ex_subfolder=subfolder_name):
                                load_score_model(score, model_use_ema=model_use_ema)
                            loaded_subfolder_name = subfolder_name
                            break
                        except Exception as exc:
                            last_exception = exc
                            logging.warning(
                                "Failed loading checkpoint from subfolder '%s' (will try next candidate if any): %s",
                                subfolder_name,
                                exc,
                            )

                    if loaded_subfolder_name is None:
                        raise RuntimeError(
                            f"Failed to load score model from checkpoint candidates {subfolder_candidates}."
                        ) from last_exception

                    logging.info(
                        f"[TIMING] download + load score model ({loaded_subfolder_name}): "
                        f"{_time.perf_counter()-_t2:.1f}s"
                    )
            
                if wandb.run is not None:
                    wandb.run.summary['num_params_score'] = sum(p.numel() for p in score.parameters() if p.requires_grad)
                if reconstruction_cfg.cfg.use_score_pass_through:
                    score = ScoreWithIdentityGradWrapper(module=score)
                # Wrap outermost with NFE counter so every score call is measured
                nfe_counter = NfeCountingScoreWrapper(score)
                score = nfe_counter

            if sample_idx >= len(dataset):
                raise ValueError(f"sample_idx={sample_idx} is out of range for dataset of size {len(dataset)}.")

            # create and init logging
            sample_logger = get_sample_logger(device=device, devices=devices, **sample_logger_cfg.cfg,
                fwd_trafo=fwd_trafo, target_trafo=target_trafo)
            
            with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="init"):
                sample_logger.init_run(num_samples=1)

            if seed is not None:
                torch.manual_seed(seed)

            _t3 = _time.perf_counter()
            data_sample = next(islice(DataLoader(dataset), sample_idx, sample_idx + 1))
            logging.info(f"[TIMING] DataLoader / dataset __getitem__ (sample_idx={sample_idx}): {_time.perf_counter()-_t3:.1f}s")

            observation, ground_truth, filtbackproj, attrs = data_sample

            observation = observation.to(dtype=dtype, device=device).squeeze()
            filtbackproj = filtbackproj.to(dtype=dtype, device=device).squeeze()
            ground_truth = ground_truth.to(dtype=dtype, device=device).squeeze()

            # Real-valued optimisation variable for a synthetic forward model
            if _as_bool(getattr(fwd_trafo, "use_synth_forward", False)) and filtbackproj.shape[-1] == 2:
                filtbackproj = filtbackproj[..., 0].contiguous()
                logging.info(
                    "fwd_trafo.use_synth_forward=True: projecting the pseudo-inverse onto the "
                    "real axis, rep_shape=%s.",
                    tuple(filtbackproj.shape),
                )
                if int(OmegaConf.select(representation_cfg.cfg, "arch.out_features", default=1)) != 1:
                    logging.warning(
                        "Overriding representation.arch.out_features=%s -> 1 for the real-valued "
                        "use_synth_forward variable.",
                        representation_cfg.cfg.arch.out_features,
                    )
                    representation_cfg.cfg.arch.out_features = 1

            detected_stack_num_slices = 1
            if reconstruction_cfg.cfg.use_score_regularisation and model_in_channels is not None:
                # Determine stack_num_slices
                if _saved_stack_num_slices is not None and _saved_stack_num_slices > 0:
                    detected_stack_num_slices = _saved_stack_num_slices
                    prior_channels_per_slice = model_in_channels // max(detected_stack_num_slices, 1)
                    logging.info(
                        "Using saved stack_num_slices=%s (prior_channels_per_slice=%s).",
                        detected_stack_num_slices,
                        prior_channels_per_slice,
                    )
                else:
                    if filtbackproj.ndim >= 4:
                        center_index = int(filtbackproj.shape[0] // 2)
                        sample_slice = filtbackproj[center_index]
                    else:
                        sample_slice = filtbackproj

                    # Probe prior_trafo with a batched single-slice tensor so that move_axis
                    # (e.g. [-1, 1]) maps the channel dim correctly: (1, H, W, 2) -> (1, 2, H, W).
                    if sample_slice.ndim == 2:
                        probe_slice = sample_slice.unsqueeze(-1).expand(*sample_slice.shape, 2).clone()
                    else:
                        probe_slice = sample_slice
                    probe_slice = probe_slice.unsqueeze(0)  # add batch dim
                    prior_sample = prior_trafo(probe_slice)
                    if prior_sample.ndim < 4:
                        raise ValueError(
                            f"prior_trafo(batched probe) is expected to produce (1, C, H, W), got shape {prior_sample.shape}."
                        )
                    prior_channels_per_slice = int(prior_sample.shape[1])

                    if model_in_channels % max(prior_channels_per_slice, 1) != 0:
                        raise ValueError(
                            "Loaded model channel count is incompatible with prior_trafo output channels: "
                            f"model_in_channels={model_in_channels}, prior_channels_per_slice={prior_channels_per_slice}."
                        )

                    detected_stack_num_slices = model_in_channels // max(prior_channels_per_slice, 1)
                    if detected_stack_num_slices < 1:
                        detected_stack_num_slices = 1

                if detected_stack_num_slices > 1:
                    logging.info(
                        "Detected stack-based model: in_channels=%s, prior_channels_per_slice=%s, stack_num_slices=%s",
                        model_in_channels,
                        prior_channels_per_slice,
                        detected_stack_num_slices,
                    )
                    prior_trafo = StackedPriorTrafoAdapter(
                        base_prior_trafo=prior_trafo,
                        stack_num_slices=detected_stack_num_slices,
                        base_channels_per_slice=prior_channels_per_slice,
                        stack_padding_mode="edge",
                    )

                    if reconstruction_cfg.cfg.method == "sampling":
                        if bool(getattr(diffmodels_cfg.cfg.sampler, "cycling", False)):
                            logging.info(
                                "Disabling sampler.cycling for stack-based model (fixed-direction stack context)."
                            )
                            diffmodels_cfg.cfg.sampler.cycling = False

            rep_shape = filtbackproj.shape
            prior_shape = prior_trafo(filtbackproj).shape
            if (
                reconstruction_cfg.cfg.use_score_regularisation
                and model_in_channels is not None
                and len(prior_shape) >= 2
                and int(prior_shape[1]) != int(model_in_channels)
            ):
                raise ValueError(
                    "prior_trafo output channels do not match the loaded score model: "
                    f"prior_shape={tuple(prior_shape)}, model_in_channels={model_in_channels}. "
                    "For Luesebrink/RSS magnitude models this should be a single-channel "
                    "magnitude tensor; check problem_trafos.prior_trafo.magnitude_enabled."
                )

            base_mesh_shape = rep_shape[:-1] if representation_cfg.cfg.arch.out_features > 1 else rep_shape 
            if representation_cfg.cfg.mesh_data.matrix_size is None:
                logging.info(f"Matrix-size is None, take base_mesh_shape: {base_mesh_shape}.")
                representation_cfg.cfg.mesh_data.matrix_size = tuple(base_mesh_shape)
            elif base_mesh_shape != representation_cfg.cfg.mesh_data.matrix_size:
                logging.error(f"base_mesh_shape: {base_mesh_shape} and matrix_size: {representation_cfg.cfg.mesh_data.matrix_size} do not match.  Use base_mesh_shape for mesh creation.")
                representation_cfg.cfg.mesh_data.matrix_size = tuple(base_mesh_shape)
            # mesh_data_con = get_mesh(representation_cfg.cfg.mesh_data_name, representation_cfg.cfg.mesh_data, device=device)
            mesh_data_con = get_mesh(representation_cfg.cfg.mesh_data, device=device)

            logging.info("Calibrating trafo")
            _t4 = _time.perf_counter()
            fwd_trafo.calibrate(observation, attrs)
            logging.info(f"[TIMING] fwd_trafo.calibrate: {_time.perf_counter()-_t4:.1f}s")

            has_precomputed_observation_scaling = (
                attrs is not None and "observation_scaling_factor" in attrs
            )
            preprocessing_observation_scaling_factor = _attrs_get_float(
                attrs, "observation_scaling_factor", 1.0
            )
            _scaling_reference_numel = _attrs_get_float(attrs, "observation_scaling_rep_numel", 0.0)
            if has_precomputed_observation_scaling and _scaling_reference_numel > 0.0:
                _rep_shape_ratio = math.sqrt(
                    float(np.prod(rep_shape).item()) / _scaling_reference_numel
                )
                if abs(_rep_shape_ratio - 1.0) > 1e-6:
                    logging.info(
                        "Rescaling prepared observation scaling_factor by %.6f: the factor was "
                        "normalised against numel=%d but rep_shape=%s has numel=%d.",
                        _rep_shape_ratio,
                        int(_scaling_reference_numel),
                        tuple(rep_shape),
                        int(np.prod(rep_shape).item()),
                    )
                    preprocessing_observation_scaling_factor *= _rep_shape_ratio
                    if wandb.run is not None:
                        wandb.run.summary["observation_scaling_rep_shape_ratio"] = _rep_shape_ratio
            if has_precomputed_observation_scaling:
                logging.info(
                    "Using prepared observation scaling_factor=%s as recon multiplier baseline.",
                    preprocessing_observation_scaling_factor,
                )
                if wandb.run is not None:
                    wandb.run.summary["preprocess_observation_scaling_factor"] = preprocessing_observation_scaling_factor
                    wandb.run.summary["recon_scaling_factor"] = math.sqrt(float(np.prod(rep_shape).item())) / observation.detach().cpu().norm() * float(reconstruction_cfg.cfg.constant_scaling_factor)

            if reconstruction_cfg.cfg.rescale_observation and has_precomputed_observation_scaling:
                scaling_factor = preprocessing_observation_scaling_factor * float(
                    reconstruction_cfg.cfg.constant_scaling_factor
                )
            elif reconstruction_cfg.cfg.rescale_observation:
                scaling_factor = math.sqrt(float(np.prod(rep_shape).item())) / observation.detach().cpu().norm() * float(reconstruction_cfg.cfg.constant_scaling_factor)
            else:
                scaling_factor = float(reconstruction_cfg.cfg.constant_scaling_factor)

            with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="init_sample" + str(sample_idx)):
                sample_logger.init_sample_log(
                    observation=observation,
                    ground_truth=ground_truth,
                    filtbackproj=filtbackproj,
                    sample_nr=sample_idx,
                    scaling_factor = scaling_factor,
                    mesh=mesh_data_con,
                    attrs=attrs
                )

            final_representation : CoordBasedRepresentation

            # NFE budget control
            # limit_nfes = -1 (default) keeps the configured parameters unchanged.
            nfe_budget_summary = _apply_nfe_budget_choice(
                reconstruction_cfg=reconstruction_cfg.cfg,
                diffmodels_cfg=diffmodels_cfg.cfg,
                prior_shape=prior_shape,
                sde=sde,
            )
            if nfe_budget_summary and wandb.run is not None:
                for key, value in nfe_budget_summary.items():
                    wandb.run.summary[key] = value

            if reconstruction_cfg.cfg.method == 'sampling':
                ####################################
                ### Sampling based methods
                ####################################

                if reconstruction_cfg.cfg.sampling is not None:
                    cond_method = get_conditioning_method(
                        fwd_trafo=fwd_trafo,
                        prior_trafo=prior_trafo,
                        im_shape=rep_shape,
                        observation=observation * scaling_factor,
                        sde=sde,
                        **reconstruction_cfg.cfg.sampling,
                    )
                else:
                    cond_method = None

                sampler = get_sampler(
                    sample_logger=sample_logger,
                    conditioning_method=cond_method,
                    fwd_trafo=fwd_trafo,
                    prior_trafo=prior_trafo,
                    score=score,
                    sde=sde,
                    device=device,
                    im_shape=prior_shape,
                    **diffmodels_cfg.cfg.sampler
                )
                sample = sampler.sample().detach()
                final_representation = FixedGridRepresentation(
                    in_shape=tuple(sample.shape[:-1]),
                    out_features=sample.shape[-1],
                    warm_start=sample,
                )

            elif reconstruction_cfg.cfg.method in ('variational', 'hybrid_variational_sampling'):
                ####################################
                ### Variational methods
                ####################################

                # datacon mesh is already initialized
                mesh_prior_reg = None
                if representation_cfg.cfg.mesh_prior_use_same_as_data:
                    logging.warning("Using same mesh for data and prior.")
                    mesh_prior_reg = get_mesh(
                        mesh_cfg=representation_cfg.cfg.mesh_data,
                        device=device,
                    )
                elif (
                    representation_cfg.cfg.mesh_prior.matrix_size is not None
                    and representation_cfg.cfg.mesh_prior.field_of_view is not None
                ):
                    logging.info(
                        f"Taking fixed mesh of size: {representation_cfg.cfg.mesh_prior.matrix_size} and fov: {representation_cfg.cfg.mesh_prior.field_of_view} for prior."
                    )
                    mesh_prior_reg = get_mesh(
                        representation_cfg.cfg.mesh_prior,
                        device=device,
                    )
                elif OmegaConf.select(
                    representation_cfg.cfg, "mesh_prior_scale_factor", default=None
                ) is not None:
                    if representation_cfg.cfg.mesh_data.field_of_view is None:
                        raise ValueError(
                            "representation.mesh_prior_scale_factor needs representation.mesh_data."
                            "field_of_view to be set, since the prior mesh inherits it (the k-space "
                            "cropping that produces the resolution shift preserves the FOV)."
                        )
                    prior_scale_factor = OmegaConf.to_container(
                        representation_cfg.cfg.mesh_prior_scale_factor, resolve=True
                    ) if OmegaConf.is_config(
                        representation_cfg.cfg.mesh_prior_scale_factor
                    ) else representation_cfg.cfg.mesh_prior_scale_factor
                    representation_cfg.cfg.mesh_prior.matrix_size = _scaled_prior_matrix_size(
                        mesh_data_con.matrix_size, prior_scale_factor
                    )
                    representation_cfg.cfg.mesh_prior.field_of_view = list(
                        representation_cfg.cfg.mesh_data.field_of_view
                    )
                    logging.info(
                        "Scaling the prior mesh by %s: data matrix %s -> prior matrix %s (fov %s).",
                        prior_scale_factor,
                        tuple(mesh_data_con.matrix_size),
                        tuple(representation_cfg.cfg.mesh_prior.matrix_size),
                        list(representation_cfg.cfg.mesh_prior.field_of_view),
                    )
                    if wandb.run is not None:
                        wandb.run.summary["mesh_prior_scale_factor"] = prior_scale_factor
                        wandb.run.summary["mesh_prior_matrix_size"] = list(
                            representation_cfg.cfg.mesh_prior.matrix_size
                        )
                    mesh_prior_reg = get_mesh(
                        representation_cfg.cfg.mesh_prior,
                        device=device,
                    )
                else:
                    logging.info("Trying to derive mesh from prior.")
                    mesh_data_per_model = OmegaConf.select(
                        diffmodels_cfg.cfg, "mesh_data_per_model", default=None
                    )
                    if mesh_data_per_model is None:
                        raise ValueError(
                            "No prior mesh could be resolved. Set exactly one of: "
                            "representation.mesh_prior_use_same_as_data=True (prior at the data "
                            "resolution), representation.mesh_prior_scale_factor=<float> (prior at "
                            "the data resolution scaled by that factor), "
                            "representation.mesh_prior.matrix_size + .field_of_view (explicit mesh), "
                            "or diffmodels.mesh_data_per_model (per-model mesh lookup)."
                        )
                    mesh_prior_reg = get_mesh_from_model(
                        mesh_cfg = representation_cfg.cfg.mesh_prior,
                        device=device,
                        model_key=diffmodels_cfg.cfg.model_key,
                        mesh_data_per_model=mesh_data_per_model,
                    )

                initialise_with = None
                if reconstruction_cfg.cfg.variational.fitting.use_filterbackproj_as_init: 
                    initialise_with = torch.clone(filtbackproj) * scaling_factor
                    logging.info("initialise with pseudoinverse")
                elif reconstruction_cfg.cfg.variational.fitting.use_l1wavelet_as_init:
                    from src.problem_trafos.utils.bart_utils import compute_l1_wavelet_solution
                    l1_wavelet_solution = compute_l1_wavelet_solution(observation, attrs["sens_maps"], reg_param=4e-4)
                    initialise_with = l1_wavelet_solution * scaling_factor
                    logging.info("initialise with l1 wavelet")
                filtbackproj = None; torch.cuda.empty_cache()

                representation = get_representation(
                    representation_cfg=representation_cfg.cfg,
                    mesh_data = mesh_data_con,
                    mesh_prior = mesh_prior_reg,
                    initialise_with = initialise_with,
                    device=device,
                )

                if wandb.run is not None:
                    params = representation.get_optimizer_params()
                    if isinstance(params, list):
                        wandb.run.summary['num_params_representation'] = sum(
                            sum(p.numel() for p in group['params'] if p.requires_grad) 
                            for group in params
                        )
                    else:
                        wandb.run.summary['num_params_representation'] = sum(
                            p.numel() for p in params if p.requires_grad
                        )

                slice_method_data_con = get_slice_method(**reconstruction_cfg.cfg.slice_methods.data_con)

                if (
                    reconstruction_cfg.cfg.use_score_regularisation
                    and detected_stack_num_slices > 1
                    and reconstruction_cfg.cfg.slice_methods.prior_reg is not None
                    and getattr(reconstruction_cfg.cfg.slice_methods.prior_reg, "name", None) == "rnd_slicing"
                ):
                    import omegaconf as _oc
                    # Disable struct validation temporarily or use OmegaConf.update to write keys that don't exist in the current struct schema
                    _oc.OmegaConf.set_struct(reconstruction_cfg.cfg, False)
                    reconstruction_cfg.cfg.slice_methods.prior_reg.stack_num_slices = int(detected_stack_num_slices)
                    reconstruction_cfg.cfg.slice_methods.prior_reg.stack_padding_mode = "edge"
                    reconstruction_cfg.cfg.slice_methods.prior_reg.slice_enabled = [True, False, False]
                    reconstruction_cfg.cfg.slice_methods.prior_reg.swapaxis = [False, False, False]
                    _oc.OmegaConf.set_struct(reconstruction_cfg.cfg, True)
                    logging.info(
                        "Auto-configured variational prior slicing with stack_num_slices=%s and fixed primary slice direction.",
                        detected_stack_num_slices,
                    )

                slice_method_prior_reg = get_slice_method(**reconstruction_cfg.cfg.slice_methods.prior_reg)

                var_objective = get_variational_objective(
                    # base objective args
                    observation=observation * scaling_factor,
                    mesh_data_con=mesh_data_con,
                    mesh_data_reg=mesh_prior_reg,
                    fwd_trafo=fwd_trafo,
                    prior_trafo=prior_trafo,
                    steps_data_con=reconstruction_cfg.cfg.variational.fitting.optimizer.gradient_acc_steps_data_con,
                    steps_data_reg=reconstruction_cfg.cfg.variational.fitting.optimizer.gradient_acc_steps_prior_reg,
                    slice_method_data_con=slice_method_data_con,
                    slice_method_prior_reg=slice_method_prior_reg,
                    outer_iterations_max = reconstruction_cfg.cfg.variational.fitting.optimizer.iterations,
                    score=score, 
                    sde=sde,
                    # reg-dependent args
                    cfg_regularization = reconstruction_cfg.cfg.variational.regularization
                )

                with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="fitting_sample_" + str(sample_idx)):
                    final_representation = fit(
                        representation=representation, 
                        var_objective=var_objective,
                        cfg_fitting = reconstruction_cfg.cfg.variational.fitting,
                        sample_logger=sample_logger,
                    )

                if reconstruction_cfg.cfg.method == 'hybrid_variational_sampling':
                    if sde is None:
                        raise ValueError("hybrid_variational_sampling requires an SDE-backed score model.")

                    hybrid_start_timestep = int(
                        OmegaConf.select(
                            reconstruction_cfg.cfg,
                            "hybrid_refinement.start_timestep",
                            default=20,
                        )
                    )
                    hybrid_noise_mode = str(
                        OmegaConf.select(
                            reconstruction_cfg.cfg,
                            "hybrid_refinement.noise_mode",
                            default="forward_marginal",
                        )
                    )

                    logging.info(
                        "Starting hybrid DDS refinement from variational reconstruction at timestep %s.",
                        hybrid_start_timestep,
                    )
                    with torch.no_grad():
                        var_sample = final_representation.forward_splitted(
                            mesh_data_con,
                            device,
                            getattr(sample_logger, "sample_gen_split", 1),
                        ).detach()
                        var_prior_sample = prior_trafo(var_sample).detach()
                        init_x = _forward_diffuse_from_x0(
                            x0=var_prior_sample,
                            sde=sde,
                            start_timestep=hybrid_start_timestep,
                            noise_mode=hybrid_noise_mode,
                        ).detach()

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "hybrid_var_rec_mean": float(var_sample.detach().mean().cpu().item()),
                                "hybrid_var_rec_std": float(var_sample.detach().std().cpu().item()),
                                "hybrid_start_timestep": hybrid_start_timestep,
                                "global_step": sample_idx,
                            }
                        )
                        wandb.run.summary["hybrid_start_timestep"] = hybrid_start_timestep

                    if reconstruction_cfg.cfg.sampling is not None:
                        cond_method = get_conditioning_method(
                            fwd_trafo=fwd_trafo,
                            prior_trafo=prior_trafo,
                            im_shape=rep_shape,
                            observation=observation * scaling_factor,
                            sde=sde,
                            x_var_anchor=var_sample,
                            **reconstruction_cfg.cfg.sampling,
                        )
                    else:
                        cond_method = None

                    sampler = get_sampler(
                        sample_logger=sample_logger,
                        conditioning_method=cond_method,
                        fwd_trafo=fwd_trafo,
                        prior_trafo=prior_trafo,
                        score=score,
                        sde=sde,
                        device=device,
                        im_shape=prior_shape,
                        **diffmodels_cfg.cfg.sampler
                    )
                    sample = sampler.sample(
                        init_x=init_x,
                        start_timestep=hybrid_start_timestep,
                    ).detach()
                    final_representation = FixedGridRepresentation(
                        in_shape=tuple(sample.shape[:-1]),
                        out_features=sample.shape[-1],
                        warm_start=sample,
                    )

            else:
                raise NotImplementedError(f'Reconstruction method {reconstruction_cfg.cfg.method} not implemented.')

            # Measure actual NFEs and log to W&B
            measured_nfes: int = nfe_counter.nfe if nfe_counter is not None else -1
            if measured_nfes > 0:
                wandb.log({"nfe": measured_nfes, "global_step": sample_idx})
                if wandb.run is not None:
                    wandb.run.summary["nfes"] = measured_nfes

            with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="log_sample_" + str(sample_idx)):
                sample_logger.close_sample_log(representation=final_representation)

            with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="close"):
                sample_logger.close_run()
                final_stats = sample_logger.get_final_stats()
                runtime_total_s = float(_time.perf_counter() - _task_wall_t0)

                # Expose measured NFE count as an output metric alongside time
                if measured_nfes > 0:
                    final_stats["nfes"] = float(measured_nfes)

                final_stats["runtime_total_s"] = runtime_total_s
                final_stats["times"] = runtime_total_s
                if "time" in final_stats and final_stats["time"] is not None:
                    final_stats["runtime_recon_s"] = float(final_stats["time"])
                else:
                    final_stats["time"] = runtime_total_s
                    final_stats["runtime_recon_s"] = runtime_total_s

                if wandb.run is not None:
                    wandb.run.summary["runtime_total_s"] = runtime_total_s
                    wandb.run.summary["times"] = runtime_total_s
                    if "runtime_recon_s" in final_stats:
                        wandb.run.summary["runtime_recon_s"] = float(final_stats["runtime_recon_s"])

                # If the sample_logger saved numpy volumes, expose the storage
                # location so downstream visual tasks can retrieve them.
                if final_stats.get("saved_final_sample_npy_files"):
                    final_stats["storage_path"] = output_dir_in_fs
                    final_stats["storage_settings"] = dict(storage_settings_cfg.cfg)
                final_stats["sample_idx"] = sample_idx
                return CacheableDict(cfg=final_stats)
    raise RuntimeError("Temporary directory context exited before task end.")