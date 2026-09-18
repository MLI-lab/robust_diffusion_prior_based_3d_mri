"""
    Module: train_diff_models.py
"""
import logging
import os
from typing import Callable, Optional, Any, Union, List
from prefect.cache_policies import TASK_SOURCE, INPUTS, NONE

import wandb

from src.prefect.wandb_lock import locked_wandb_init

import torch
from torch.utils.data import DataLoader

from omegaconf import DictConfig, OmegaConf
from src.utils.wandb_utils import wandb_kwargs_for_prefect_task, WandbParamsTask
import hydra
from prefect import task
from src.prefect.caching import CacheableDict, CacheableDictConfig, CacheableListConfig

from src.diffmodels.diffmodels_resolver import create_model
from src.diffmodels import score_model_trainer, load_sde_model
from src.utils.wandb_utils import wandb_kwargs_via_cfg
from src.utils.device_utils import get_free_cuda_devices, get_all_devices
from src.tasks.dataset_pipeline_utils import cache_base_from_subfolder, concrete_path_resolver

from src.datasets.dataset_resolver import get_dataset
from src.problem_trafos.trafo_resolver import (get_dataset_trafo,
    get_prior_trafo)

from src.diffmodels.trainer.in_memory_dataset import cache_iterable_in_memory

from src.diffmodels.sampler.sampler_resolver import get_sampler
from src.sample_logger.sample_logger_resolver import get_sample_logger
from src.prefect.setup import StoragePath, StorageSettings, load_file_system
from datetime import datetime
import tempfile
from src.prefect.setup import switch_dir_and_upload_directory_on_exit, retry_s3_transfer
from functools import partial

def get_dataloader(dataset_cfg, dataset_trafo, cfg_dl, problem_trafos_cfg, fold, device : str, path_resolver : Callable, cache_base_path : str,
                   data_path_override: Optional[Any] = None,
                   data_path_train_override: Optional[Any] = None):

    def _extract_tensor_from_sample(sample: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(sample):
            return sample
        if isinstance(sample, (tuple, list)):
            for elem in sample:
                if torch.is_tensor(elem):
                    return elem
        return None

    cache_limit = cfg_dl.get('cache_limit', -1)
    cache_limit_suffix = f"_limit{cache_limit}" if cache_limit != -1 else ""
    _key_path = data_path_override if data_path_override is not None else data_path_train_override
    if _key_path is not None:
        import hashlib, json
        _path_str = json.dumps(_key_path, sort_keys=True) if not isinstance(_key_path, str) else _key_path
        _path_hash = hashlib.sha1(_path_str.encode()).hexdigest()[:10]
        _path_suffix = f"_prep{_path_hash}"
    else:
        _path_suffix = ""
    cache_path = os.path.join(cache_base_path,
        cfg_dl.cache_dataset_disk_path,
        f"{dataset_cfg.ser()}_{problem_trafos_cfg.ser()}{cache_limit_suffix}{_path_suffix}_cache.pt")
    
    # first check if we need to cache the dataset in gpu
    cache_in_gpu = False
    cache_device = None
    if cfg_dl.cache_dataset:
        cache_in_gpu = cfg_dl.cache_dataset_in_gpu
        cache_device = device if cache_in_gpu  else "cpu"

    # first try to load the dataset from the fs cache
    dataset = None
    if cfg_dl.cache_dataset and cfg_dl.cache_dataset_load_from_disk:
        if os.path.exists(cache_path):
            dataset = torch.load(cache_path, map_location=cache_device)

            _stack_num_slices = int(getattr(dataset_cfg.cfg, "stack_num_slices", 1) or 1)
            if _stack_num_slices > 1:
                try:
                    _cached_sample = dataset[0]
                    _cached_tensor = _extract_tensor_from_sample(_cached_sample)
                    _invalid_cached_shape = (
                        _cached_tensor is None
                        or _cached_tensor.ndim != 3
                    )
                except Exception:
                    _invalid_cached_shape = True

                if _invalid_cached_shape:
                    logging.warning(
                        "Invalid cached stack dataset detected at %s (expected sample tensor ndim=3 for stack training). Rebuilding cache.",
                        cache_path,
                    )
                    dataset = None
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
        else:
            logging.warning(f"Could not find cached dataset at {cache_path} -> Loading it from scratch.")

    # create the dataset as usual when it has not been loaded from disk
    if dataset is None:
        dataset_kwargs = dict(OmegaConf.to_container(dataset_cfg.cfg, resolve=True))
        if data_path_train_override is not None:
            dataset_kwargs["data_path_train"] = data_path_train_override
        if data_path_override is not None:
            if "train" in fold:
                dataset_kwargs["data_path_train"] = data_path_override
            elif "val" in fold:
                dataset_kwargs["data_path_val"] = data_path_override
        iterable_dataset = get_dataset(**dataset_kwargs, fold_overwrite=fold, dataset_trafo=dataset_trafo, path_resolver=path_resolver)

        if cfg_dl.cache_dataset:
            dataset = cache_iterable_in_memory(
                iterable_ds=iterable_dataset, use_tqdm=True, device=cache_device, repeat_dataset=cfg_dl.cache_dataset_repeats,
                cache_limit=cache_limit
                )

            if cfg_dl.cache_dataset_store_on_disk:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(dataset, cache_path)
        else:
            dataset = iterable_dataset

    # create the dataloader
    if cfg_dl.use_batch_sampler_same_shape:
        from src.diffmodels.trainer.batch_sampler_same_shape import BatchSamplerSameShape
        sampler = BatchSamplerSameShape(dataset,
            shuffle        =    cfg_dl.shuffle,
            batch_size     =    cfg_dl.batch_size,
            group_shape_by =    cfg_dl.group_shape_by)

        dataloader = DataLoader(
            dataset,
            pin_memory=False,
            num_workers=cfg_dl.num_workers if not cache_in_gpu else 0,
            batch_sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg_dl.batch_size,
            shuffle=cfg_dl.shuffle,
            num_workers=cfg_dl.num_workers if not cache_in_gpu else 0,
            pin_memory=False
        )

    return dataloader

@task(cache_policy=INPUTS - "wandb_params_task", name="Train Task", tags=["train"], version="1.0", retries=0)
def train_task(
    seed: int,
    problem_trafos_cfg: CacheableDictConfig,
    diffmodels_cfg: CacheableDictConfig,
    dataset_cfg: CacheableDictConfig,
    sample_logger_cfg: CacheableDictConfig,
    storage_settings_cfg: CacheableDictConfig,
    wandb_params_task: WandbParamsTask,
    local_cache_path: str,
    train_cache_subfolder: str = "train_cache",
    data_path_train_override: Optional[Any] = None,
    data_path_val_override: Optional[Any] = None,
    skip_validation: bool = False,
) -> StoragePath:

    # resolve and prepare
    # pick device(s)
    devices = get_all_devices()
    if not devices:
        devices = ["cpu"]
    device = devices[0]

    logging.getLogger().setLevel(logging.INFO)

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

        with locked_wandb_init(**wandb_kwargs):

            if seed is not None:
                torch.manual_seed(seed)

            # resolving paths, e.g. for the datasets and loaded models, depending on the cluster
            path_resolver = concrete_path_resolver(local_cache_path)

            # get prior and dataset trafos
            prior_trafo = get_prior_trafo(**problem_trafos_cfg.cfg.prior_trafo)
            _prior_trafo_cfg_to_save = OmegaConf.create(dict(problem_trafos_cfg.cfg.prior_trafo))

            # Auto-detect magnitude-only preprocessed data
            _preprocess_prior_mode: Optional[str] = None
            _data_path_for_meta = data_path_train_override
            if isinstance(_data_path_for_meta, (list, tuple)):
                _data_path_for_meta = next((p for p in _data_path_for_meta if p is not None), None)
            if _data_path_for_meta is not None:
                import json as _json
                from pathlib import Path as _Path
                _meta_path = _Path(str(_data_path_for_meta)) / "_preprocess_meta.json"
                if _meta_path.exists():
                    with open(_meta_path) as _f:
                        _meta = _json.load(_f)
                    _preprocess_prior_mode = _meta.get("prior_mode")
            if _preprocess_prior_mode == "magnitude_1ch":
                _pt_cfg = OmegaConf.to_container(
                    OmegaConf.create(dict(problem_trafos_cfg.cfg.prior_trafo)), resolve=True
                )
                _is_complex_cfg = _pt_cfg.get("stack_z_into_batchdim", False) or not _pt_cfg.get("magnitude_enabled", False)
                if _is_complex_cfg:
                    _data_object_type = getattr(dataset_cfg.cfg, "data_object_type", None)
                    if _data_object_type == "slices":
                        _mag_layout_cfg = {
                            "magnitude_enabled": False,
                            "stack_channel_into_batchdim": False,
                            "stack_z_into_batchdim": False,
                            "move_axis": [-1, 1],
                            "collapse_singleton_complex_channel": True,
                        }
                    else:
                        _mag_layout_cfg = {
                            "magnitude_enabled": True,
                            "stack_channel_into_batchdim": True,
                            "stack_z_into_batchdim": False,
                            "move_axis": None,
                        }
                    logging.warning(
                        "Auto-switching prior_trafo to magnitude_1ch mode because "
                        "_preprocess_meta.json reports prior_mode=magnitude_1ch but "
                        "the configured prior_trafo appears complex. "
                        "Using layout overrides %s for dataset.data_object_type=%s."
                        % (_mag_layout_cfg, _data_object_type)
                    )
                    _pt_cfg.update(_mag_layout_cfg)
                    prior_trafo = get_prior_trafo(**_pt_cfg)
                    _prior_trafo_cfg_to_save = OmegaConf.create(_pt_cfg)
                    # Check arch channels to warn if model won't match
                    _arch_in_ch = getattr(getattr(diffmodels_cfg.cfg, "arch", None), "params", None)
                    if _arch_in_ch is not None:
                        _in_ch = getattr(_arch_in_ch, "in_channels", None)
                        if _in_ch is not None and int(_in_ch) != 1:
                            logging.error(
                                f"magnitude_1ch prior_mode detected but diffmodels.arch.params.in_channels={_in_ch} "
                                "instead of 1. Use '+exps=prepped/train_dense_mag' or set arch.params.in_channels=1."
                            )

            dataset_trafo = get_dataset_trafo(**problem_trafos_cfg.cfg.dataset_trafo,
                provide_pseudoinverse=False, provide_measurement=False, device=device)

            # dataloaders
            cache_base_path = str(cache_base_from_subfolder(local_cache_path, train_cache_subfolder, "train_cache"))
            dataloader_train = get_dataloader(dataset_cfg=dataset_cfg,
                dataset_trafo=dataset_trafo, cfg_dl=diffmodels_cfg.cfg.train, problem_trafos_cfg=problem_trafos_cfg,
                fold="train", device=device, path_resolver=path_resolver, cache_base_path=cache_base_path,
                data_path_override=data_path_train_override,
                data_path_train_override=data_path_train_override)
            dataloader_val = None
            if not skip_validation:
                dataloader_val = get_dataloader(dataset_cfg=dataset_cfg,
                    dataset_trafo=dataset_trafo, cfg_dl=diffmodels_cfg.cfg.val, problem_trafos_cfg=problem_trafos_cfg,
                    fold="val", device=device, path_resolver=path_resolver, cache_base_path=cache_base_path,
                    data_path_override=data_path_val_override,
                    data_path_train_override=data_path_train_override)
            else:
                logging.info("No validation dataset configured; skipping validation loss evaluation.")

            _stack_num_slices = getattr(dataset_cfg.cfg, "stack_num_slices", 1)
            if _stack_num_slices is not None and int(_stack_num_slices) > 1:
                _target_type = getattr(problem_trafos_cfg.cfg.dataset_trafo, "target_type", None)
                _channels_per_slice = 2 if _target_type == "mvue" else 1
                _expected_channels = int(_stack_num_slices) * _channels_per_slice

                try:
                    _sample_target = next(iter(dataloader_train))
                    if isinstance(_sample_target, (tuple, list)):
                        _sample_target = _sample_target[0]
                    if torch.is_tensor(_sample_target):
                        _sample_target_single = _sample_target[:1]
                        _sample_prior = prior_trafo(_sample_target_single)
                        if torch.is_tensor(_sample_prior) and _sample_prior.ndim >= 2:
                            _expected_channels = int(_sample_prior.shape[1])
                except Exception as _exc:
                    logging.warning(
                        "Could not infer stack channels from sample/prior_trafo; falling back to target_type-based estimate (%s channels). Reason: %s",
                        _expected_channels,
                        _exc,
                    )

                _arch_params = diffmodels_cfg.cfg.arch.params
                _current_in = int(_arch_params.in_channels)
                _current_out = int(_arch_params.out_channels)
                if _current_in != _expected_channels or _current_out != _expected_channels:
                    logging.warning(
                        "Detected stack_num_slices=%s with target_type=%s. "
                        "Overriding diffmodels.arch.params.in_channels/out_channels from (%s, %s) to (%s, %s).",
                        _stack_num_slices,
                        _target_type,
                        _current_in,
                        _current_out,
                        _expected_channels,
                        _expected_channels,
                    )
                    _arch_params.in_channels = _expected_channels
                    _arch_params.out_channels = _expected_channels

            try:
                _shape_probe = next(iter(dataloader_train))
                if isinstance(_shape_probe, (tuple, list)):
                    _shape_probe = _shape_probe[0]
                if torch.is_tensor(_shape_probe):
                    _shape_probe_prior = prior_trafo(_shape_probe[:1].to(device))
                    if _shape_probe_prior.ndim != 4:
                        raise ValueError(
                            f"prior_trafo must produce a 4D NCHW tensor for diffusion training, "
                            f"got shape {tuple(_shape_probe_prior.shape)} from input "
                            f"{tuple(_shape_probe[:1].shape)}."
                        )
                    _arch_params = diffmodels_cfg.cfg.arch.params
                    _expected_in_channels = int(_arch_params.in_channels)
                    if int(_shape_probe_prior.shape[1]) != _expected_in_channels:
                        raise ValueError(
                            f"prior_trafo output channels ({int(_shape_probe_prior.shape[1])}) do not match "
                            f"diffmodels.arch.params.in_channels ({_expected_in_channels}); "
                            f"prior shape={tuple(_shape_probe_prior.shape)}, input shape={tuple(_shape_probe[:1].shape)}."
                        )
            except ValueError:
                raise
            except Exception as _exc:
                logging.warning("Could not validate train prior_trafo output shape before model creation: %s", _exc)

            # load models
            sde = load_sde_model(**diffmodels_cfg.cfg.sde)
            #score = load_score_model(diffmodels_cfg.cfg, device=device, path_resolver=path_resolver)
            score = create_model(**dict(diffmodels_cfg.cfg.arch), arch_cfg=None).to(device)
            if wandb.run is not None:
                wandb.run.summary['num_params_score'] = sum(p.numel() for p in score.parameters() if p.requires_grad)

            with switch_dir_and_upload_directory_on_exit_mgr(new_subfolder="configs"):
                with open("diffmodels_cfg.yaml", "w") as f:
                    OmegaConf.save(config=diffmodels_cfg.cfg, f=f)
                # Save prior_trafo config so recon stage can auto-match the training setup.
                with open("prior_trafo_cfg.yaml", "w") as f:
                    OmegaConf.save(config=_prior_trafo_cfg_to_save, f=f)
                # Save stack_num_slices so recon can detect stack models reliably.
                _save_stack_num_slices = int(getattr(dataset_cfg.cfg, "stack_num_slices", 1) or 1)
                with open("stack_num_slices.txt", "w") as f:
                    f.write(str(_save_stack_num_slices))

            # setup logging and sampler
            sample_logger = get_sample_logger(device=device, devices=devices, **sample_logger_cfg.cfg)
            sampler = get_sampler(
                score=score,
                sde=sde,
                im_shape=prior_trafo(next(iter(dataloader_val if dataloader_val is not None else dataloader_train))).shape,
                device=device,
                sample_logger=sample_logger,
                prior_trafo=prior_trafo,
                **diffmodels_cfg.cfg.sampler
            )

            # start training
            score_model_trainer(
                score=score,
                sde=sde,
                dataloader_train=dataloader_train,
                dataloader_val=dataloader_val,
                sample_logger=sample_logger,
                sampler=sampler,
                optim_kwargs=diffmodels_cfg.cfg.train,
                val_kwargs=diffmodels_cfg.cfg.val,
                prior_trafo=prior_trafo,
                device=device,
                switch_dir_and_upload_directory_on_exit_mgr=switch_dir_and_upload_directory_on_exit_mgr
            )

        return StoragePath(
            storage_path=output_dir_in_fs,
            storage_settings=storage_settings
        )