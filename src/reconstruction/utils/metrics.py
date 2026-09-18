from typing import Dict, Optional, Any, Tuple
from functools import lru_cache
from torch import Tensor
import os
import torch

from skimage.metrics import structural_similarity

import numpy as np
import scipy
import logging


def _slices_to_3ch_01(t: Tensor) -> Tensor:
    """Reshape to (N, H, W), normalise to [0,1], expand to 3-channel (N,3,H,W)."""
    t = t.detach().cpu().reshape(-1, t.shape[-2], t.shape[-1]).float()
    t_min = t.flatten(1).min(dim=1).values[:, None, None]
    t_max = t.flatten(1).max(dim=1).values[:, None, None]
    t = (t - t_min) / (t_max - t_min + 1e-8)
    return t.unsqueeze(1).expand(-1, 3, -1, -1)  # (N,3,H,W)


def PSNR(
    rec: Tensor,
    gt: Tensor,
) -> Tensor:
    return PSNR_pt(reconstruction=rec, ground_truth=gt, data_range=torch.max(gt))


def PSNR_pt(
    reconstruction: Tensor, ground_truth: Tensor, data_range: Optional[Tensor] = None
) -> Tensor:

    mse = (reconstruction - ground_truth).square().mean()
    if data_range is None:
        data_range = torch.max(ground_truth) - np.min(ground_truth)
    return 20 * torch.log10(data_range) - 10 * torch.log10(mse)


def normalize(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


def _ssim_win_size_for_shape(shape: Tuple[int, ...]) -> Optional[int]:
    min_side = min(shape)
    if min_side < 3:
        return None
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    return max(3, win_size)


def _ssim_data_range(gt: np.ndarray) -> float:
    finite = gt[np.isfinite(gt)]
    if finite.size == 0:
        return 1.0
    data_range = float(np.max(finite) - np.min(finite))
    if data_range <= 0.0 or not np.isfinite(data_range):
        max_val = float(np.max(np.abs(finite)))
        return max(max_val, 1.0)
    return data_range


FOREGROUND_MASK_THRESHOLD = 0.05


def foreground_mask(gt: np.ndarray, threshold: float = FOREGROUND_MASK_THRESHOLD) -> np.ndarray:
    """Boolean foreground mask ``|gt| > threshold * max(|gt|)``, per volume."""
    mag = np.abs(np.asarray(gt))
    finite = mag[np.isfinite(mag)]
    if finite.size == 0:
        return np.zeros(mag.shape, dtype=bool)
    max_mag = float(np.max(finite))
    if not np.isfinite(max_mag) or max_mag <= 0.0:
        return np.zeros(mag.shape, dtype=bool)
    return mag > threshold * max_mag


def masked_volume_ssim(
    rec: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    win_size: int,
    average_over_mask: bool = True,
    min_foreground_voxels: int = 10,
) -> Optional[float]:
    """3D SSIM restricted to the foreground `mask`."""
    if mask.shape != gt.shape or mask.shape != rec.shape:
        return None
    if int(mask.sum()) < min_foreground_voxels:
        logging.warning(
            "Skipping foreground SSIM: only %d foreground voxels (min %d).",
            int(mask.sum()),
            min_foreground_voxels,
        )
        return None

    gt_masked = np.where(mask, gt, 0.0)
    rec_masked = np.where(mask, rec, 0.0)
    data_range = float(np.max(gt_masked))
    if not np.isfinite(data_range) or data_range <= 0.0:
        logging.warning("Skipping foreground SSIM: non-positive data range %s.", data_range)
        return None

    mssim, ssim_map = structural_similarity(
        gt_masked, rec_masked, data_range=data_range, win_size=win_size, full=True
    )
    if not average_over_mask:
        return float(mssim) if np.isfinite(mssim) else None

    # skimage averages its SSIM map over the valid (unpadded) region only; crop
    # the map and the mask identically so both refer to the same voxels.
    pad = (win_size - 1) // 2
    valid = tuple(slice(pad, dim - pad) for dim in ssim_map.shape)
    map_valid = ssim_map[valid]
    mask_valid = mask[valid]
    if int(mask_valid.sum()) < min_foreground_voxels:
        logging.warning("Skipping foreground SSIM: too few foreground voxels after window cropping.")
        return None
    val = float(np.mean(map_valid[mask_valid]))
    return val if np.isfinite(val) else None


def _to_magnitude_array(volume: Any) -> np.ndarray:
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume)
    if np.iscomplexobj(volume):
        return np.abs(volume)
    if volume.ndim >= 4 and volume.shape[-1] == 2:
        return np.linalg.norm(volume, axis=-1)
    return volume


def _drop_leading_volume_dims(volume: Any) -> np.ndarray:
    vol = _to_magnitude_array(volume)

    # Preserve spatial volume axes while removing trivial batch/channel axes.
    # Common recon outputs are (D, H, W), (D, H, W, 1), or (1, D, H, W).
    while vol.ndim > 3 and vol.shape[-1] == 1:
        vol = np.squeeze(vol, axis=-1)
    while vol.ndim > 3 and vol.shape[0] == 1:
        vol = np.squeeze(vol, axis=0)
    while vol.ndim > 3:
        vol = vol[0]

    return vol.astype(np.float32, copy=False)


def _center_window_bounds(size: int, frac: float) -> Tuple[int, int]:
    """Return (start, end) bounds selecting a centered window of `size` that
    covers `frac` fraction of the slices (at least 1, clamped to [0, 1])."""
    frac = min(max(frac, 0.0), 1.0)
    num = max(1, min(size, round(size * frac)))
    start = (size - num) // 2
    return start, start + num


def _center_crop_pair_to_common_shape(
    rec: np.ndarray,
    gt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if rec.shape == gt.shape:
        return rec, gt

    common_shape = tuple(min(r, g) for r, g in zip(rec.shape, gt.shape))

    def _crop(vol: np.ndarray) -> np.ndarray:
        slices = []
        for dim, target in zip(vol.shape, common_shape):
            start = max(0, (dim - target) // 2)
            slices.append(slice(start, start + target))
        return vol[tuple(slices)]

    logging.warning(
        "Volume SSIM shape mismatch rec=%s gt=%s; center-cropping both to %s.",
        rec.shape,
        gt.shape,
        common_shape,
    )
    return _crop(rec), _crop(gt)


def _estimate_ncc_params_from_background(
    reference: np.ndarray,
    background_quantile: float = 0.10,
    min_background_voxels: int = 128,
    default_effective_channels: float = 1.0,
) -> Tuple[float, float]:
    ref = np.abs(np.asarray(reference, dtype=np.float64))
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return 1.0, default_effective_channels

    data_max = float(np.max(ref))
    data_min = float(np.min(ref))
    data_range = max(data_max - data_min, data_max, 1.0)
    sigma_floor = max(data_range * 1e-6, np.finfo(np.float64).tiny)

    quantiles = []
    for q in [background_quantile, 0.05, 0.10, 0.20, 0.30]:
        if 0.0 < q < 1.0 and q not in quantiles:
            quantiles.append(q)

    for q in quantiles:
        threshold = float(np.quantile(ref, q))
        background = ref[ref <= threshold]
        if background.size < min_background_voxels:
            continue

        background_power = np.square(background)
        mean_power = float(np.mean(background_power))
        var_power = float(np.var(background_power, ddof=1)) if background_power.size > 1 else 0.0
        if mean_power <= sigma_floor ** 2 or var_power <= sigma_floor ** 4:
            continue

        effective_channels = float(np.clip((mean_power ** 2) / var_power, 1.0, 256.0))
        sigma_sq = max(mean_power / effective_channels, sigma_floor ** 2)
        return float(np.sqrt(sigma_sq)), effective_channels

    positive = ref[ref > sigma_floor]
    if positive.size:
        sigma = max(float(np.percentile(positive, 5.0)) / np.sqrt(default_effective_channels), sigma_floor)
    else:
        sigma = sigma_floor
    return sigma, float(max(default_effective_channels, 1.0))


def _log_modified_bessel_iv(order: float, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = np.maximum(z, 0.0)
    out = np.empty_like(z, dtype=np.float64)

    small = z < 1e-5
    if np.any(small):
        z_small = np.maximum(z[small], np.finfo(np.float64).tiny)
        if abs(order) < 1e-12:
            out[small] = 0.25 * np.square(z_small)
        else:
            out[small] = order * np.log(0.5 * z_small) - scipy.special.gammaln(order + 1.0)

    if np.any(~small):
        z_large = z[~small]
        scaled = scipy.special.ive(order, z_large)
        vals = np.log(scaled) + z_large
        bad = ~np.isfinite(vals)
        if np.any(bad):
            zb = z_large[bad]
            vals[bad] = zb - 0.5 * np.log(2.0 * np.pi * zb)
        out[~small] = vals

    return out


def noncentral_chi_error(
    rec_volume: Any,
    gt_volume: Any,
    sigma: Optional[float] = None,
    effective_channels: Optional[float] = None,
) -> Optional[float]:
    """Compute mean noncentral-chi error (NCE) against a noisy magnitude reference."""
    rec = np.abs(_drop_leading_volume_dims(rec_volume).astype(np.float64, copy=False))
    gt = np.abs(_drop_leading_volume_dims(gt_volume).astype(np.float64, copy=False))
    if rec.ndim != 3 or gt.ndim != 3:
        logging.warning("Skipping NCE metric for non-3D shapes rec=%s gt=%s.", rec.shape, gt.shape)
        return None

    rec, gt = _center_crop_pair_to_common_shape(rec, gt)
    finite_mask = np.isfinite(rec) & np.isfinite(gt)
    if not finite_mask.any():
        logging.warning("Skipping NCE metric because no finite voxels were found.")
        return None

    x = rec[finite_mask]
    y = gt[finite_mask]
    if sigma is None or effective_channels is None:
        est_sigma, est_effective_channels = _estimate_ncc_params_from_background(y)
        sigma = est_sigma if sigma is None else sigma
        effective_channels = est_effective_channels if effective_channels is None else effective_channels

    sigma = float(sigma)
    effective_channels = float(effective_channels)
    if sigma <= 0.0 or not np.isfinite(sigma) or effective_channels < 1.0 or not np.isfinite(effective_channels):
        logging.warning("Skipping NCE metric for invalid NCC parameters sigma=%s L=%s.", sigma, effective_channels)
        return None

    sigma_sq = sigma ** 2
    intensity_floor = max(float(np.max(y)) * 1e-12, sigma * 1e-12, np.finfo(np.float64).tiny)
    x = np.maximum(x, intensity_floor)
    y = np.maximum(y, intensity_floor)

    order = effective_channels - 1.0
    z_xy = 2.0 * x * y / sigma_sq
    z_yy = 2.0 * y * y / sigma_sq

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nce_voxels = (
            (np.square(x) - np.square(y)) / sigma_sq
            + order * (np.log(x) - np.log(y))
            - (_log_modified_bessel_iv(order, z_xy) - _log_modified_bessel_iv(order, z_yy))
        )

    finite_nce = nce_voxels[np.isfinite(nce_voxels)]
    if finite_nce.size == 0:
        logging.warning("Skipping NCE metric because all per-voxel values were non-finite.")
        return None

    return float(np.mean(finite_nce))


def volume_nce_metrics(rec_volume: Any, gt_volume: Any) -> Dict[str, float]:
    """Compute the 3D reconstruction NCE metric."""
    rec_nce = noncentral_chi_error(rec_volume, gt_volume)
    if rec_nce is None:
        return {}
    return {"rec_nce": rec_nce}


def volume_ssim_metrics(
    rec_volume: Any,
    gt_volume: Any,
    center_frac: float = 0.5,
    fg_threshold: Optional[float] = None,
    fg_average_over_mask: bool = True,
) -> Dict[str, float]:
    """Compute whole-volume SSIM and center-region 2D average SSIM."""
    rec = _drop_leading_volume_dims(rec_volume)
    gt = _drop_leading_volume_dims(gt_volume)
    if rec.ndim != 3 or gt.ndim != 3:
        logging.warning("Skipping volume SSIM metrics for non-3D shapes rec=%s gt=%s.", rec.shape, gt.shape)
        return {}

    rec, gt = _center_crop_pair_to_common_shape(rec, gt)
    finite_mask = np.isfinite(rec) & np.isfinite(gt)
    if not finite_mask.any():
        logging.warning("Skipping volume SSIM metrics because no finite voxels were found.")
        return {}
    rec = np.where(np.isfinite(rec), rec, 0.0)
    gt = np.where(np.isfinite(gt), gt, 0.0)

    data_range = _ssim_data_range(gt)
    metrics: Dict[str, float] = {}

    win_3d = _ssim_win_size_for_shape(tuple(rec.shape))
    if win_3d is not None:
        try:
            metrics["rec_ssim3d"] = float(
                structural_similarity(gt, rec, data_range=data_range, win_size=win_3d)
            )
        except Exception as exc:
            logging.warning("Could not compute 3D SSIM for reconstructed volume: %s", exc)

    if win_3d is not None and fg_threshold is not None:
        try:
            ssim3d_fg = masked_volume_ssim(
                rec,
                gt,
                foreground_mask(gt, threshold=float(fg_threshold)),
                win_size=win_3d,
                average_over_mask=fg_average_over_mask,
            )
            if ssim3d_fg is not None:
                metrics["rec_ssim3d_fg"] = ssim3d_fg
        except Exception as exc:
            logging.warning("Could not compute foreground 3D SSIM for reconstructed volume: %s", exc)

    slice_ssims = []
    for axis in range(3):
        rec_axis = np.moveaxis(rec, axis, 0)
        gt_axis = np.moveaxis(gt, axis, 0)
        start, end = _center_window_bounds(rec_axis.shape[0], center_frac)
        for rec_slice, gt_slice in zip(rec_axis[start:end], gt_axis[start:end]):
            win_2d = _ssim_win_size_for_shape(tuple(rec_slice.shape))
            if win_2d is None:
                continue
            try:
                val = float(
                    structural_similarity(gt_slice, rec_slice, data_range=data_range, win_size=win_2d)
                )
            except Exception as exc:
                logging.warning("Could not compute 2D slice SSIM for axis=%d: %s", axis, exc)
                continue
            if np.isfinite(val):
                slice_ssims.append(val)

    if slice_ssims:
        metrics["rec_ssim2d_avg"] = float(np.mean(slice_ssims))

    return metrics


def PSNR_2D(
    rec: Tensor,
    gt: Tensor,
    axis: int = 0,
    use_vol_max: bool = False,
    take_abs_normalize: bool = False,
) -> Tuple[Tensor, Tensor]:

    if rec.ndim == 5:
        # (B, Z, X, Y, 2)
        rec = rec.squeeze(0)
        gt = gt.squeeze(0)

    if axis != 0:
        rec = rec.moveaxis(axis, 0)
        gt = gt.moveaxis(axis, 0)

    if take_abs_normalize:
        rec = normalize(torch.abs(rec))
        gt = normalize(torch.abs(gt))

    psnrs = PSNR_2D_pt(
        reconstruction=rec,
        ground_truth=gt,
        data_range=torch.max(gt) if use_vol_max else None,
    )

    return psnrs.mean(), psnrs.std()


def PSNR_2D_pt(
    reconstruction: Tensor, ground_truth: Tensor, data_range: Optional[Tensor] = None
) -> Tensor:

    reconstruction = reconstruction.reshape(reconstruction.shape[0], -1)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)

    mse = (reconstruction - ground_truth).square().mean(dim=1)
    if data_range is None:
        data_range = (
            torch.max(ground_truth, dim=1).values
            - torch.min(ground_truth, dim=1).values
        )
    return 20 * torch.log10(data_range) - 10 * torch.log10(mse)


def _center_slice(t: Tensor, axis: int) -> Tensor:
    """Return the single center slice along *axis* as a (1, H, W) tensor."""
    if t.ndim == 2:
        return t.detach().reshape(1, t.shape[0], t.shape[1])
    t = t.detach().moveaxis(axis, 0)
    mid = t.shape[0] // 2
    s = t[mid]  # should now be (H, W)
    return s.reshape(1, s.shape[-2], s.shape[-1])


def SSIM(
    rec: Tensor,
    gt: Tensor,
    axis: int = 0,
    max_val: Optional[float] = None,
    take_abs_normalize: bool = False,
    center_slices_only: bool = True,
) -> Tuple[float, float]:
    """Compute SSIM.  When *center_slices_only* is True (default) only the
    single center slice from each of the 3 spatial axes is used, which is
    ~D/3 times faster than iterating over all slices."""

    if rec.ndim == 4:
        # (B, Z, X, Y)
        rec = rec.squeeze(0)
        gt = gt.squeeze(0)

    if take_abs_normalize:
        rec = normalize(torch.abs(rec))
        gt = normalize(torch.abs(gt))

    if max_val is None:
        max_val = torch.max(gt).item()

    if center_slices_only and rec.ndim == 3 and gt.ndim == 3:
        ssim_vals = []
        for ax in range(3):
            r_c = _center_slice(rec, ax)  # (1, H, W)
            g_c = _center_slice(gt, ax)
            s = ssim_np(
                pred=r_c.cpu().numpy(),
                gt=g_c.cpu().numpy(),
                maxval=max_val,
            )
            ssim_vals.append(float(s[0]))
        mean_val = float(np.mean(ssim_vals))
        return mean_val, float(np.std(ssim_vals))

    if axis != 0:
        rec = rec.moveaxis(axis, 0)
        gt = gt.moveaxis(axis, 0)

    ssims = ssim_np(
        pred=rec.detach().cpu().reshape(-1, rec.shape[-2], rec.shape[-1]).numpy(),
        gt=gt.detach().cpu().reshape(-1, rec.shape[-2], rec.shape[-1]).numpy(),
        maxval=max_val,
    )

    return ssims.mean(), ssims.std()


def ssim_np(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    assert gt.ndim == pred.ndim, "Input images must have the same dimensions."
    if gt.ndim == 4 and gt.shape[1] == 1 and pred.shape[1] == 1 and pred.ndim == 4:
        # assume (B, C, H, W)
        gt = gt[:, 0, ...]
        pred = pred[:, 0, ...]
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssims = np.zeros(gt.shape[0])
    for slice_num in range(gt.shape[0]):
        h, w = gt[slice_num].shape[-2], gt[slice_num].shape[-1]
        min_side = min(h, w)
        # win_size must be odd, <= min_side, and >= 3 (win_size=1 -> NP=1 -> ZeroDivision)
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        win_size = max(win_size, 3)  # skimage requires NP = win_size**2 >= 4
        if min_side < 3:
            # image too small for any meaningful SSIM window; skip
            ssims[slice_num] = 0.0
            continue
        ssims[slice_num] = structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval, win_size=win_size
        )
    return ssims


def VIFP(
    rec: Tensor,
    gt: Tensor,
    center_slices_only: bool = True,
) -> np.number:
    """Compute VIFP.  When *center_slices_only* is True (default) only the
    single center slice from each of the 3 spatial axes is used."""
    if gt.ndim < 2 or rec.ndim < 2:
        return np.float64(0.0)

    if center_slices_only and rec.ndim == 3 and gt.ndim == 3:
        vifp_vals = []
        for ax in range(3):
            r_c = _center_slice(rec, ax).cpu().numpy()  # (1, H, W)
            g_c = _center_slice(gt, ax).cpu().numpy()
            vifp_vals.append(vifp_mscale_np(ref=g_c, dist=r_c))
        return np.mean(vifp_vals)

    return vifp_mscale_np(
        ref=gt.detach().cpu().reshape(-1, gt.shape[-2], gt.shape[-1]).numpy(),
        dist=rec.detach().cpu().reshape(-1, rec.shape[-2], rec.shape[-1]).numpy(),
    )


def vifp_mscale_np(ref, dist, eps=1e-10):
    if not ref.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not ref.ndim == dist.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    vifps = np.zeros(ref.shape[0])
    for slice_num in range(ref.shape[0]):
        vifps[slice_num] = vifp_mscale_single_np(
            ref[slice_num][None, ...],
            dist[slice_num][None, ...],
            sigma_nsq=ref[slice_num].mean(),
            eps=eps,
        )

    nr_nans = np.isnan(vifps).sum()
    if nr_nans > 0:
        logging.warn(f"Number of VIFP nans: {nr_nans} / {vifps.size}")

    return np.mean(vifps[~np.isnan(vifps)])


def vifp_mscale_single_np(ref, dist, sigma_nsq=1, eps=1e-10):
    ### from https://github.com/aizvorski/video-quality/blob/master/vifp.py
    sigma_nsq = sigma_nsq  ### tune this for your dataset to get reasonable numbers
    eps = eps

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if scale > 1:
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    return vifp

@lru_cache(maxsize=None)
def _lpips_net(net: str):
    import lpips as _lpips_lib

    loss_fn = _lpips_lib.LPIPS(net=net, verbose=False)
    loss_fn.eval()
    return loss_fn


@lru_cache(maxsize=None)
def _dists_net():
    import importlib.util
    from DISTS_pytorch import DISTS as _DISTS

    loss_fn = _DISTS(load_weights=False)
    # DISTS_pytorch looks for weights.pt via sys.prefix, which is wrong inside
    # Ray's pip virtualenv.  Load directly from the package directory instead.
    _dists_pkg_dir = os.path.dirname(importlib.util.find_spec("DISTS_pytorch").origin)
    _weights = torch.load(os.path.join(_dists_pkg_dir, "weights.pt"), weights_only=True)
    loss_fn.alpha.data = _weights["alpha"]
    loss_fn.beta.data = _weights["beta"]
    loss_fn.eval()
    return loss_fn


def LPIPS(
    rec: Tensor,
    gt: Tensor,
    net: str = "alex",
    center_slices_only: bool = True,
) -> float:
    """Mean LPIPS (lower = better).  When *center_slices_only* is True
    (default) only the center slice per spatial axis (3 slices total) is used."""
    loss_fn = _lpips_net(net)

    if center_slices_only and rec.ndim == 3:
        scores = []
        for ax in range(3):
            r_c = _center_slice(rec, ax)  # (1, H, W)
            g_c = _center_slice(gt, ax)
            r4 = (_slices_to_3ch_01(r_c) * 2 - 1)
            g4 = (_slices_to_3ch_01(g_c) * 2 - 1)
            with torch.no_grad():
                scores.append(float(loss_fn(r4, g4).mean().item()))
        return float(np.mean(scores))

    rec_4ch = _slices_to_3ch_01(rec)   # (N,3,H,W) in [0,1]
    gt_4ch  = _slices_to_3ch_01(gt)
    # lpips expects [-1,1]
    rec_4ch = rec_4ch * 2 - 1
    gt_4ch  = gt_4ch  * 2 - 1
    with torch.no_grad():
        scores = loss_fn(rec_4ch, gt_4ch)  # (N,1,1,1)
    return float(scores.mean().item())


def DISTS(
    rec: Tensor,
    gt: Tensor,
    center_slices_only: bool = True,
) -> float:
    """Mean DISTS (lower = better).  When *center_slices_only* is True
    (default) only the center slice per spatial axis (3 slices total) is used."""
    loss_fn = _dists_net()

    if center_slices_only and rec.ndim == 3:
        scores = []
        for ax in range(3):
            r_c = _center_slice(rec, ax)  # (1, H, W)
            g_c = _center_slice(gt, ax)
            r4 = _slices_to_3ch_01(r_c)
            g4 = _slices_to_3ch_01(g_c)
            with torch.no_grad():
                s = loss_fn(r4, g4)
                scores.append(float(s.mean().item() if s.ndim > 0 else s.item()))
        return float(np.mean(scores))

    rec_4ch = _slices_to_3ch_01(rec)   # (N,3,H,W) in [0,1]
    gt_4ch  = _slices_to_3ch_01(gt)
    with torch.no_grad():
        score = loss_fn(rec_4ch, gt_4ch)   # scalar or (N,)
    return float(score.mean().item() if score.ndim > 0 else score.item())