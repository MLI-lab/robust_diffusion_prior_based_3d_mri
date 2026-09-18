
import torch
import numpy as np
from fastmri.data.subsample import MaskFunc
from typing import Optional, Tuple, Union, Sequence

def _normalize_calib(calib):
    if calib is None:
        return None
    if isinstance(calib, int):
        return (calib, calib)
    if len(calib) != 2:
        raise ValueError(f"calib must be an int or a length-2 sequence, got {calib}.")
    return (int(calib[0]), int(calib[1]))


def get_gaussian_2d_mask_rej(
        shape,
        acc_factor,
        seed=0,
        cov_factor=12,
        base_size=320,
        max_retries=100,
        calib=None,
):
    # adapted from https://github.com/HJ-harry/DDS/blob/main/utils.py
    mux_in = shape[-2] * shape[-1]
    Nsamp = int(mux_in // acc_factor)
    mask = torch.zeros(shape)
    mean = [shape[-2] // 2, shape[-1] // 2]
    cov = [[shape[-2] * cov_factor * shape[-2] / base_size, 0],
         [0, shape[-1] * cov_factor * shape[-1] / base_size] ]

    calib = _normalize_calib(calib)
    if calib is not None:
        calib_y, calib_x = calib
        if calib_y < 0 or calib_x < 0:
            raise ValueError(f"calib entries must be non-negative, got {calib}.")
        calib_y = min(calib_y, shape[-2])
        calib_x = min(calib_x, shape[-1])
        y0 = (shape[-2] - calib_y) // 2
        x0 = (shape[-1] - calib_x) // 2
        mask[..., y0:y0 + calib_y, x0:x0 + calib_x] = 1

    Nsamp_success = int(mask.sum().item())
    if Nsamp_success > Nsamp:
        raise ValueError(
            f"calib={calib} selects {Nsamp_success} samples, which exceeds the "
            f"requested total sample count {Nsamp} for shape={shape} and acc_factor={acc_factor}."
        )

    rng = np.random.default_rng(seed)

    Nsamp_it = 1
    retries = 0

    while Nsamp_success < Nsamp:
        samples = rng.multivariate_normal(mean, cov, int(Nsamp_it))
        int_samples = samples.astype(int)
    
        mask_samples_out_of_bounds = (int_samples < 0).sum() > 0 or (int_samples[:,0] >= shape[-2]).any() or (int_samples[:,1] >= shape[-1]).any()
        if not mask_samples_out_of_bounds:
            mask_entries_exists = mask[..., int_samples[:, 0], int_samples[:, 1]].sum() > 0
        else:
            mask_entries_exists = True

        if not mask_entries_exists and not mask_samples_out_of_bounds:
            mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
            Nsamp_success += int_samples.shape[0]
            retries = 0
        else:
            retries += 1
            if retries > max_retries:
                raise ValueError(f"Reached maximum of retries.")

    return mask


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    # based on the fastMRI github repo
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    if data.get_device() >= 0:
        mask = mask.to(data.get_device())
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies