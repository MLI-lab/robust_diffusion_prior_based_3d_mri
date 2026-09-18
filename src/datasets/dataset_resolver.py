import json
import logging
from pathlib import Path
from typing import Tuple, List, Callable

import numpy as np

from torch import Tensor

from torch.utils.data import ConcatDataset
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import h5py
import numpy as np
import os

from omegaconf import DictConfig

from src.datasets.fastmri_slice_dataset import ExtSliceDataset
from src.datasets.fastmri_volume_dataset import FastMRIVolumeDataset
from fastmri.data.mri_data import FastMRIRawDataSample
from .base_dataset import BaseDataset

from torch.utils.data import Dataset

def _load_preprocess_meta(data_path) -> dict:
    """If any of the data root directory(ies) contains a ``_preprocess_meta.json`` written by ``preprocess_h5_directory``, load and return it.
    """
    if data_path is None:
        return {}
    paths = [data_path] if isinstance(data_path, str) else list(data_path)
    for p in paths:
        for meta_name in ("_preprocess_recon_meta.json", "_preprocess_meta.json"):
            meta_file = Path(p) / meta_name
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                logging.info(f"Auto-loaded dataset metadata from {meta_file}: {meta}")
                return meta
    return {}


def get_fastmri_dataset(
        fold_overwrite : Optional[str],
        fold : str,
        dataset_trafo : Optional[Any],
        data_path_train : Dict,
        data_path_val : Dict,
        data_path_recon : Dict,
        data_path_sensmaps_train : Dict,
        data_path_sensmaps_val : Dict,
        data_path_sensmaps_recon : Dict,
        volume_filter_train : Optional[str],
        volume_filter_val : Optional[str],
        volume_filter_test : Optional[str],
        path_resolver : Optional[Callable],
        raw_sample_filter : Dict,
        # raw_sample_filter_enabled : bool,
        # raw_sample_filter_encoding_size: Optional[int],
        data_object_type : str = "slices",
        **dataset_kwargs
        ):

    dataset = None
    data_path = None
    data_path_sensmaps = None
    volume_filter = None

    requires_sensmaps = dataset_trafo.requires_sensmaps()

    if fold_overwrite is not None and fold != fold_overwrite:
        logging.info(f"Overwriting fold {fold} with {fold_overwrite}")
        fold = fold_overwrite
    
    if "train" in fold:
        data_path = path_resolver(data_path_train)
        data_path_sensmaps = path_resolver(data_path_sensmaps_train) if requires_sensmaps and data_path_sensmaps_train is not None else None
        volume_filter = volume_filter_train
    elif "val" in fold:
        data_path = path_resolver(data_path_val)
        data_path_sensmaps = path_resolver(data_path_sensmaps_val) if requires_sensmaps and data_path_sensmaps_val is not None else None
        volume_filter = volume_filter_val
        if data_path is None:
            logging.info("data_path_val is null, falling back to data_path_train.")
            data_path = path_resolver(data_path_train)
            data_path_sensmaps = path_resolver(data_path_sensmaps_train) if requires_sensmaps and data_path_sensmaps_train is not None else None
    elif "test" in fold:
        data_path = path_resolver(data_path_recon)
        data_path_sensmaps = path_resolver(data_path_sensmaps_recon) if requires_sensmaps and data_path_sensmaps_recon is not None else None
        volume_filter = volume_filter_test
        # If test paths are not configured (null), fall back to train paths and rely on volume_filter_test
        if data_path is None:
            logging.info("data_path_recon is null, falling back to data_path_train with volume_filter_test.")
            data_path = path_resolver(data_path_train)
            data_path_sensmaps = path_resolver(data_path_sensmaps_train) if requires_sensmaps and data_path_sensmaps_train is not None else None
    else:
        raise NotImplementedError(f"Fold {fold} not supported")

    if data_path is None:
        raise ValueError(
            f"data_path resolved to None for fold='{fold}'. "
            f"Check that the dataset config has a path entry for the current cluster."
        )

    preprocess_meta = _load_preprocess_meta(data_path)
    if preprocess_meta:
        # If preprocessed/prepared data omitted or already embeds sensmaps, do not generate them here.
        if preprocess_meta.get("skip_sensmaps", False):
            logging.info("Preprocessed data detected with skip_sensmaps=True; will not request sensmaps")
            requires_sensmaps = False
        if preprocess_meta.get("prepared_recon_dataset", False):
            logging.info("Prepared test data detected; using stored observations/sensmaps and skipping sensmap generation")
            requires_sensmaps = preprocess_meta.get("sensmaps_key") is not None
            dataset_kwargs = {
                **dataset_kwargs,
                "apply_fft1c_on_readout_dim": False,
                "readout_dim_keep_spatial": False,
            }
            if preprocess_meta.get("sampling_kind") == "noncartesian":
                dataset_kwargs = {**dataset_kwargs, "dataset_is_3d": False}
        # recons_key maps directly; sensmaps_key -> sensmaps_key_in_h5 (dataset param name).
        # Only inject keys the dataset actually understands; ignore the rest.
        if "recons_key" in preprocess_meta:
            dataset_kwargs = {**dataset_kwargs, "recons_key": preprocess_meta["recons_key"]}
        if "sensmaps_key" in preprocess_meta:
            dataset_kwargs = {**dataset_kwargs, "sensmaps_key_in_h5": preprocess_meta["sensmaps_key"]}
        if "sensmap_coil_dim_nr" in preprocess_meta:
            dataset_kwargs = {**dataset_kwargs, "sensmap_coil_dim_nr": preprocess_meta["sensmap_coil_dim_nr"]}
        if "observation_key" in preprocess_meta:
            dataset_kwargs = {**dataset_kwargs, "kspace_key": preprocess_meta["observation_key"]}
        if "pseudoinverse_key" in preprocess_meta:
            dataset_kwargs = {**dataset_kwargs, "pseudoinverse_key_in_h5": preprocess_meta["pseudoinverse_key"]}

    if data_object_type == "slices":

        if raw_sample_filter["enabled"]:

            def raw_sample_filter_func(sample : FastMRIRawDataSample, root : str, sensemap_root : str):

                # skip_first_last = 0
                # slice_ind_condition = sample.slice_ind >= skip_first_last and sample.slice_ind <= sample.metadata["num_slices"] - skip_first_last

                encoding_condition = sample.metadata["encoding_size"][1] == raw_sample_filter["encoding_size"] if raw_sample_filter["encoding_size"] is not None else True

                norms = sample.metadata["target_slice_norms"]
                mean, std = norms.mean(), norms.std()
                cutoff_threshold_factor_lower = raw_sample_filter["slice_norm_cutoff_factor_lower"]
                # cutoff_threshold_factor_upper = raw_sample_filter["slice_norm_cutoff_factor_upper"]
                lower_bound = mean - cutoff_threshold_factor_lower * std
                # upper_bound = mean + cutoff_threshold_factor_upper * std

                slice_norm_condition = lower_bound <= norms[sample.slice_ind] if raw_sample_filter["slice_norm_filter_enabled"] else True

                return encoding_condition and slice_norm_condition
        else:
            raw_sample_filter_func = None

        if isinstance(data_path, str):
            dataset = ExtSliceDataset(root=data_path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=data_path_sensmaps, return_sensmaps=requires_sensmaps, volume_filter=volume_filter, **dataset_kwargs)

        else:
            _sensmap_paths = data_path_sensmaps if data_path_sensmaps is not None else [None] * len(data_path)
            dataset = ConcatDataset([ExtSliceDataset(root=path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=path_sense, return_sensmaps=requires_sensmaps, volume_filter=volume_filter, **dataset_kwargs) for path, path_sense in zip(data_path, _sensmap_paths)])

    elif data_object_type == "volumes":

        if raw_sample_filter["enabled"]:
            def raw_sample_filter_func(sample : FastMRIRawDataSample, root : str, sensemap_root : str):
                encoding_condition = sample.metadata["encoding_size"][1] == raw_sample_filter["encoding_size"] if raw_sample_filter["encoding_size"] is not None else True
                return encoding_condition
        else:
            raw_sample_filter_func = None

        if isinstance(data_path, str):
            dataset = FastMRIVolumeDataset(root=data_path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=data_path_sensmaps, volume_filter=volume_filter, return_sensmaps=requires_sensmaps, **dataset_kwargs)
        else:
            _sensmap_paths = data_path_sensmaps if data_path_sensmaps is not None else [None] * len(data_path)
            dataset = ConcatDataset([FastMRIVolumeDataset(root=path, raw_sample_filter=raw_sample_filter_func, transform=dataset_trafo, sensmap_files_root=path_sense, volume_filter=volume_filter, return_sensmaps=requires_sensmaps, **dataset_kwargs) for path, path_sense in zip(data_path, _sensmap_paths)])

    else:
        raise NotImplementedError(f"Dataset type {data_object_type} not supported")

    # check if we need to calculate sensemaps    
    if requires_sensmaps:
        logging.info("Sensmaps required, generate if necessary...")
        if isinstance(data_path, str):
            dataset.calc_sensmap_files()
        elif isinstance(data_path, list):
            for ds in dataset.datasets:
                ds.calc_sensmap_files()


    return dataset

def get_dataset(
        name : str,
        dataset_trafo : Optional[Any] = None,
        path_resolver : Optional[Callable] = None,
        fold_overwrite : Optional[str] = None,
        **cfg_kwargs
        ) -> BaseDataset:
    if name == "FastMRIDataset":

        from src.datasets.dataset_resolver import get_fastmri_dataset
        dataset = get_fastmri_dataset(
            dataset_trafo=dataset_trafo,
            path_resolver=path_resolver,
            fold_overwrite=fold_overwrite,
            **cfg_kwargs
        )

    else: 
        raise NotImplementedError(f"Dataset {name} not supported")

    return dataset