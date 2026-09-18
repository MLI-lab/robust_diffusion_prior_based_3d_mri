"""3D MRI multi-perspective visual plot task."""

import logging
import os
import re
import tempfile
import zipfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import wandb
from src.prefect.wandb_lock import locked_wandb_init
from prefect import task
from prefect.cache_policies import NONE

from src.prefect.caching import CacheableDict, CacheableDictConfig
from src.prefect.setup import StorageSettings, load_file_system
from src.utils.wandb_utils import WandbParamsTask, wandb_kwargs_for_prefect_task


# helpers

def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Unwrap CacheableDict / plain dict to a flat Python dict."""
    if isinstance(result, CacheableDict):
        return dict(result.cfg)
    if isinstance(result, dict):
        return dict(result)
    cfg = getattr(result, "cfg", None)
    if cfg is not None:
        return dict(cfg) if isinstance(cfg, dict) else {}
    return {}


def _to_magnitude(volume: np.ndarray) -> np.ndarray:
    """Return magnitude of a complex volume or the raw array if already real."""
    if np.iscomplexobj(volume):
        return np.abs(volume)
    # Detect real+imag interleaved storage: float array with last dim == 2 and ndim >= 4
    if volume.ndim >= 4 and volume.shape[-1] == 2 and not np.iscomplexobj(volume):
        return np.linalg.norm(volume, axis=-1)
    return volume


def _normalize(img: np.ndarray) -> np.ndarray:
    """Normalise a 2-D image to [0, 1] for display."""
    if img.size == 0:
        return img.astype(float)
    img_f = img.astype(float, copy=False)
    finite_mask = np.isfinite(img_f)
    if not finite_mask.any():
        return np.zeros_like(img_f, dtype=float)

    finite_vals = img_f[finite_mask]
    vmin, vmax = finite_vals.min(), finite_vals.max()
    if np.isclose(vmin, vmax):
        return np.zeros_like(img_f, dtype=float)

    clean = np.where(finite_mask, img_f, vmin)
    return (clean - vmin) / (vmax - vmin)


def _drop_leading_volume_dims(volume: np.ndarray) -> np.ndarray:
    """Reduce singleton/batch-like leading dimensions until a 3-D volume remains."""
    vol = _to_magnitude(np.asarray(volume))
    while vol.ndim > 3:
        vol = vol[0]
    return vol.astype(np.float32, copy=False)



def _compute_zoom_box(
    h: int, w: int,
    zoom_center_frac_h: float,
    zoom_center_frac_w: float,
    zoom_size_frac: float,
) -> Tuple[int, int, int, int]:
    """Return (row_start, row_end, col_start, col_end) for the zoom inset."""
    half = max(1, int(zoom_size_frac * min(h, w) / 2))
    cy = int(zoom_center_frac_h * h)
    cx = int(zoom_center_frac_w * w)
    r0 = max(0, cy - half)
    r1 = min(h, cy + half)
    c0 = max(0, cx - half)
    c1 = min(w, cx + half)
    return r0, r1, c0, c1


def _get_slice(volume: np.ndarray, axis: int, frac: float) -> np.ndarray:
    """Extract a 2-D slice from a 3-D volume along `axis` at position `frac`."""
    idx = int(frac * volume.shape[axis])
    idx = max(0, min(volume.shape[axis] - 1, idx))
    if axis == 0:
        return volume[idx, :, :]
    elif axis == 1:
        return volume[:, idx, :]
    else:
        return volume[:, :, idx]


def _try_download_volume(
    storage_path: str,
    storage_settings: Dict[str, Any],
    sample_idx: int,
) -> Optional[np.ndarray]:
    """Try to download `final_rec_{sample_idx}.npy` from storage and return the
    numpy array, or None if unavailable.
    """
    import tempfile, os
    from src.prefect.setup import download_directory_to_temp_on_enter
    try:
        fs_settings = StorageSettings(**storage_settings)
        fs = load_file_system(fs_settings)
        subfolder = f"log_sample_{sample_idx}"
        with tempfile.TemporaryDirectory() as tmp:
            with download_directory_to_temp_on_enter(fs, tmp, storage_path, subfolder):
                local_path = os.path.join(tmp, subfolder, f"final_rec_{sample_idx}.npy")
            return np.load(local_path)
    except Exception as exc:
        logging.warning("Could not download volume from %s: %s", storage_path, exc)
        return None


def _normalize_shared(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Apply a pre-computed [vmin, vmax] normalisation so all panels share the
    same intensity scale."""
    if np.isclose(vmin, vmax):
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr.astype(float) - vmin) / (vmax - vmin), 0.0, 1.0)


def _ax_label(ax: "plt.Axes", text: str, fontsize: int = 7) -> None:
    """Overlay a label in the top-left corner of an axes (no title spacing)."""
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        color="white", fontsize=fontsize,
        verticalalignment="top", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
    )


def _render_three_panels(
    volume: np.ndarray,
    axial_slice_frac: float,
    coronal_slice_frac: float,
    sagittal_slice_frac: float,
    zoom_box: Tuple[int, int, int, int],
    zoom_display_factor: float,
    cmap: str,
    shared_norm_percentile_low: float,
    shared_norm_percentile_high: float,
) -> Tuple["plt.Figure", Dict[str, np.ndarray]]:
    """
    Render a 2x2 figure with geometrically aligned panels:

      top-left     : axial slice HxW (with zoom rectangle + indicator lines)
      top-right    : sagittal NEIGHBOURHOOD (H x d_nbhd), centred on the
                     current axial slice; d_nbhd = zoom_h so the z-extent
                     shown matches the zoom patch height in the axial plane.
      bottom-left  : coronal NEIGHBOURHOOD (d_nbhd x W), same z-window.
      bottom-right : zoomed-in axial patch (the ROI "cube")

    All four panels share the same vmin/vmax computed robustly from volume
    percentiles (to avoid outlier-driven darkening) so brightness is consistent
    across panels. Figure background is black and
    there are no axis titles - labels are embedded as text overlays - so no
    white cross appears between panels.

    Returns the figure AND a dict of the four named panel arrays.
    """
    mag = _to_magnitude(volume)
    D, H, W = mag.shape

    # shared intensity scale from the entire volume
    finite_vals = mag[np.isfinite(mag)]
    if finite_vals.size:
        p_low = float(np.clip(shared_norm_percentile_low, 0.0, 100.0))
        p_high = float(np.clip(shared_norm_percentile_high, 0.0, 100.0))
        if p_high <= p_low:
            p_high = min(100.0, p_low + 1.0)
        g_vmin = float(np.percentile(finite_vals, p_low))
        g_vmax = float(np.percentile(finite_vals, p_high))
    else:
        g_vmin = 0.0
        g_vmax = 1.0
    if np.isclose(g_vmin, g_vmax):
        g_vmax = g_vmin + 1.0

    # primary (axial)
    axial_idx = int(axial_slice_frac * D)
    axial_idx = max(0, min(D - 1, axial_idx))
    axial = mag[axial_idx, :, :]                                  # HxW

    r0, r1, c0, c1 = zoom_box
    zoom_h = r1 - r0
    zoom_w = c1 - c0
    zoom_patch = axial[r0:r1, c0:c1]                             # zoom_h x zoom_w

    # neighbourhood window in D centred on axial_idx
    d_nbhd = max(4, zoom_h)
    d_half = d_nbhd // 2
    d_start = max(0, axial_idx - d_half)
    d_end   = min(D, d_start + d_nbhd)
    d_start = max(0, d_end - d_nbhd)          # keep size fixed near boundaries
    d_nbhd_actual = d_end - d_start
    axial_local = axial_idx - d_start          # indicator position within window

    # sagittal neighbourhood (H x d_nbhd)
    sag_idx = int(sagittal_slice_frac * W)
    sag_idx = max(0, min(W - 1, sag_idx))
    sagittal_panel = mag[d_start:d_end, :, sag_idx].T            # Hxd_nbhd

    # coronal neighbourhood (d_nbhd x W)
    cor_idx = int(coronal_slice_frac * H)
    cor_idx = max(0, min(H - 1, cor_idx))
    coronal_panel = mag[d_start:d_end, cor_idx, :]                # d_nbhdxW

    # shared normalisation
    panels = {
        "axial_full":     _normalize_shared(axial,         g_vmin, g_vmax),
        "sagittal_panel": _normalize_shared(sagittal_panel, g_vmin, g_vmax),
        "coronal_panel":  _normalize_shared(coronal_panel,  g_vmin, g_vmax),
        "zoom_patch":     _normalize_shared(zoom_patch,     g_vmin, g_vmax),
    }

    # Figure layout
    width_ratios  = [W, d_nbhd_actual]
    height_ratios = [H, d_nbhd_actual]
    total_w = W + d_nbhd_actual
    total_h = H + d_nbhd_actual
    fig_w = 10.0 * total_w / max(total_w, total_h)
    fig_h = 10.0 * total_h / max(total_w, total_h)

    fig = plt.figure(figsize=(max(4.0, fig_w), max(4.0, fig_h)),
                     facecolor="black")
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.0,
    )
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    for ax in [ax00, ax01, ax10, ax11]:
        ax.axis("off")
        ax.set_facecolor("black")

    # Kill ALL automatic spacing - titles and suptitle add height even with
    # hspace=0.  Subplots_adjust takes full ownership of the layout.
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    # top-left: axial
    ax00.imshow(panels["axial_full"], cmap=cmap, aspect="auto",
                vmin=0.0, vmax=1.0, interpolation="nearest")
    rect = mpatches.Rectangle(
        (c0, r0), zoom_w, zoom_h,
        linewidth=1.5, edgecolor="red", facecolor="none",
    )
    ax00.add_patch(rect)
    ax00.axhline(y=cor_idx, color="yellow", linewidth=0.8, linestyle="--", alpha=0.8)
    ax00.axvline(x=sag_idx, color="yellow", linewidth=0.8, linestyle="--", alpha=0.8)
    _ax_label(ax00, "Axial")

    # top-right: sagittal neighbourhood
    ax01.imshow(panels["sagittal_panel"], cmap=cmap, aspect="auto",
                vmin=0.0, vmax=1.0, interpolation="nearest")
    ax01.axhline(y=cor_idx,     color="yellow", linewidth=0.8, linestyle="--", alpha=0.8)
    ax01.axvline(x=axial_local, color="yellow", linewidth=1.0, linestyle="-",  alpha=0.9)
    _ax_label(ax01, f"Sag nbhd  w={sag_idx}  Δz={d_nbhd_actual}")

    # bottom-left: coronal neighbourhood
    ax10.imshow(panels["coronal_panel"], cmap=cmap, aspect="auto",
                vmin=0.0, vmax=1.0, interpolation="nearest")
    ax10.axvline(x=sag_idx,     color="yellow", linewidth=0.8, linestyle="--", alpha=0.8)
    ax10.axhline(y=axial_local, color="yellow", linewidth=1.0, linestyle="-",  alpha=0.9)
    _ax_label(ax10, f"Cor nbhd  h={cor_idx}  Δz={d_nbhd_actual}")

    # bottom-right: zoom patch
    ax11.imshow(panels["zoom_patch"], cmap=cmap, aspect="auto",
                vmin=0.0, vmax=1.0, interpolation="nearest")
    _ax_label(ax11, "Zoom ROI")

    return fig, panels


def _render_axial_zoom(
    volume: np.ndarray,
    axial_slice_frac: float,
    zoom_box: Tuple[int, int, int, int],
    zoom_display_factor: float,
    cmap: str,
) -> "plt.Figure":
    """Render a single-panel figure: full axial slice with red rectangle marking
    the zoom region, a zoom-patch inset in the bottom-right corner, and space
    reserved at the bottom-left for the metric overlay.
    """
    mag = _to_magnitude(volume)
    axial = _get_slice(mag, axis=0, frac=axial_slice_frac)       # HxW

    r0, r1, c0, c1 = zoom_box
    zoom_h = r1 - r0
    zoom_w = c1 - c0
    zoom_patch = _normalize(axial[r0:r1, c0:c1])
    axial_n    = _normalize(axial)

    H, W = axial_n.shape
    # Inset size as a fraction of the figure axes
    inset_w = min(0.45, zoom_display_factor * zoom_w / W)
    inset_h = min(0.45, zoom_display_factor * zoom_h / H)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * H / W))
    ax.imshow(axial_n, cmap=cmap, aspect="auto")
    ax.axis("off")

    # Red rectangle marking the zoom region
    rect = mpatches.Rectangle(
        (c0, r0), zoom_w, zoom_h,
        linewidth=1.5, edgecolor="red", facecolor="none",
    )
    ax.add_patch(rect)

    # Inset axes in the bottom-right corner [x0, y0, width, height] in axes fraction
    x0_inset = 1.0 - inset_w - 0.01
    y0_inset = 0.01
    ax_inset = ax.inset_axes([x0_inset, y0_inset, inset_w, inset_h])
    ax_inset.imshow(zoom_patch, cmap=cmap, aspect="auto")
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(1.5)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    plt.tight_layout(pad=0.3)
    return fig


def _add_metric_overlay(
    ax: "plt.Axes",
    result_dict: Dict[str, Any],
    metric_keys: List[str],
    text_color: str,
    font_scale: float,
) -> None:
    """Write metric values as text in the LOWER-LEFT corner of `ax`."""
    lines = []
    for key in metric_keys:
        if key in result_dict:
            val = result_dict[key]
            try:
                lines.append(f"{key}: {float(val):.3f}")
            except (TypeError, ValueError):
                lines.append(f"{key}: {val}")
    if not lines:
        return
    text = "\n".join(lines)
    ax.text(
        0.02, 0.02, text,
        transform=ax.transAxes,
        color=text_color,
        fontsize=max(5, int(8 * font_scale)),
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )


def _safe_filename(label: str, max_len: int = 80) -> str:
    """Convert a sweep label string to a safe filename stem."""
    s = re.sub(r"[^\w\-,=.]", "_", label)
    return s[:max_len]


def _canonical_param_name(name: str) -> str:
    """Normalize sweep parameter names for robust matching."""
    return str(name).lstrip("+")


def _infer_visual_grid_axes(df_sweep: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[Any], List[Any]]:
    """Infer (row_key, col_key, row_values, col_values) for the shared visual grid."""
    if df_sweep.empty:
        return None, None, [], []

    varying_cols: List[str] = []
    for col in df_sweep.columns:
        series = df_sweep[col]
        nunique = int(series.nunique(dropna=False))
        if nunique > 1:
            varying_cols.append(col)

    if not varying_cols:
        return None, None, [], []

    row_key: Optional[str] = None
    for col in varying_cols:
        canon = _canonical_param_name(col)
        if canon == "rec_method" or canon.endswith(".rec_method"):
            row_key = col
            break

    if row_key is None and varying_cols:
        row_key = varying_cols[0]

    col_key: Optional[str] = None
    for col in varying_cols:
        if col != row_key:
            col_key = col
            break

    row_values = list(pd.unique(df_sweep[row_key])) if row_key is not None else []
    if col_key is not None:
        col_values = list(pd.unique(df_sweep[col_key]))
    else:
        col_values = ["all"]

    return row_key, col_key, row_values, col_values


def _save_individual_pngs(
    label: str,
    panels: Dict[str, np.ndarray],
    out_dir: str,
    cmap: str,
    dpi: int,
) -> List[str]:
    """Save each panel array as an individual PNG.
    Returns a list of saved file paths.
    """
    stem = _safe_filename(label)
    paths = []
    panel_order = ["axial_full", "coronal_panel", "sagittal_panel", "zoom_patch"]
    for name in panel_order:
        img = panels.get(name)
        if img is None:
            continue
        fname = os.path.join(out_dir, f"{stem}__{name}.png")
        fig_tmp, ax_tmp = plt.subplots(1, 1, figsize=(4, 4), dpi=dpi)
        ax_tmp.imshow(img, cmap=cmap, aspect="auto")
        ax_tmp.axis("off")
        ax_tmp.set_title(f"{label[:40]}\n{name}", fontsize=6, pad=2)
        plt.tight_layout(pad=0.2)
        fig_tmp.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig_tmp)
        paths.append(fname)
    return paths


def _save_combined_png(
    label: str,
    fig: "plt.Figure",
    out_dir: str,
    dpi: int,
) -> str:
    """Save the combined perspective figure as a PNG and return the path."""
    stem = _safe_filename(label)
    fname = os.path.join(out_dir, f"{stem}__combined.png")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    return fname


def _load_volume_file(local_path: str) -> np.ndarray:
    """Load a numpy or torch-saved volume as a numpy array."""
    if local_path.endswith(".pt"):
        import torch
        t = torch.load(local_path, map_location="cpu", weights_only=False)
        return t.numpy() if hasattr(t, "numpy") else np.array(t)
    return np.load(local_path)


def _try_download_init_sample_volume(
    storage_path: str,
    storage_settings: Dict[str, Any],
    sample_idx: int,
    filename_template: str,
    volume_label: str,
) -> Optional[np.ndarray]:
    """Try to download a sample-fixed volume from init_sample{sample_idx}."""
    import tempfile as _tempfile
    from src.prefect.setup import download_directory_to_temp_on_enter
    try:
        fs_settings = StorageSettings(**storage_settings)
        fs = load_file_system(fs_settings)
        subfolder = f"init_sample{sample_idx}"
        filename = filename_template.format(sample_idx=sample_idx)
        with _tempfile.TemporaryDirectory() as tmp:
            with download_directory_to_temp_on_enter(fs, tmp, storage_path, subfolder):
                local_path = os.path.join(tmp, subfolder, filename)
                if not os.path.exists(local_path):
                    logging.warning("%s file not found at %s (subfolder=%s)", volume_label, local_path, subfolder)
                    return None
                return _load_volume_file(local_path)
    except Exception as exc:
        logging.warning("Could not download %s volume from %s: %s", volume_label, storage_path, exc)
        return None


def _try_download_gt_volume(
    storage_path: str,
    storage_settings: Dict[str, Any],
    sample_idx: int,
    filename_template: str = "ground_truth_{sample_idx}.pt",
) -> Optional[np.ndarray]:
    """Try to download a GT volume from storage."""
    return _try_download_init_sample_volume(
        storage_path=storage_path,
        storage_settings=storage_settings,
        sample_idx=sample_idx,
        filename_template=filename_template,
        volume_label="GT",
    )


def _try_download_pseudoinverse_volume(
    storage_path: str,
    storage_settings: Dict[str, Any],
    sample_idx: int,
    filename_template: str = "filtbackproj_{sample_idx}.pt",
) -> Optional[np.ndarray]:
    """Try to download a saved pseudo-inverse / filtered-backprojection volume."""
    return _try_download_init_sample_volume(
        storage_path=storage_path,
        storage_settings=storage_settings,
        sample_idx=sample_idx,
        filename_template=filename_template,
        volume_label="pseudoinverse",
    )


def _render_comparison_strip(
    gt_volume: Optional[np.ndarray],
    rec_volumes_with_labels: List[Tuple[str, Optional[np.ndarray]]],
    result_dicts: List[Dict[str, Any]],
    axial_slice_frac: float,
    zoom_box: Tuple[int, int, int, int],
    zoom_display_factor: float,
    cmap: str,
    metric_overlay_enabled: bool,
    metric_overlay_metrics: List[str],
    metric_overlay_text_color: str,
    metric_overlay_font_scale: float,
    dpi: int,
) -> "plt.Figure":
    """Create a side-by-side comparison figure placing GT (if available) as the
    first column and each method reconstruction in subsequent columns.
    """
    r0, r1, c0, c1 = zoom_box
    zoom_h = r1 - r0
    zoom_w = c1 - c0

    # Build ordered list: [(title, volume, result_dict_or_None), ...]
    all_cols: List[Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]] = []
    if gt_volume is not None:
        all_cols.append(("Ground Truth", gt_volume, None))
    for (label, vol), rdict in zip(rec_volumes_with_labels, result_dicts):
        all_cols.append((label, vol, rdict))

    n_cols = len(all_cols)
    if n_cols == 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
        ax.axis("off")
        ax.set_title("No volumes available")
        return fig

    col_width = 4.5
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(n_cols * col_width, 5.5),
        dpi=dpi,
        squeeze=False,
    )
    axes_row = axes[0]

    for col_idx, (label, vol, rdict) in enumerate(all_cols):
        ax = axes_row[col_idx]
        if vol is None:
            ax.axis("off")
            ax.set_title(label[:55], fontsize=7, pad=3)
            continue

        mag = _to_magnitude(vol)
        while mag.ndim > 3:
            mag = mag[0]
        axial = _get_slice(mag, axis=0, frac=axial_slice_frac)
        axial_n = _normalize(axial)
        zoom_patch_n = _normalize(axial[r0:r1, c0:c1])

        ax.imshow(axial_n, cmap=cmap, aspect="auto")
        ax.axis("off")
        ax.set_title(label[:55], fontsize=7, pad=3)

        # Red rectangle for zoom ROI
        rect = mpatches.Rectangle(
            (c0, r0), zoom_w, zoom_h,
            linewidth=1.5, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)

        # Zoom inset in bottom-right corner
        H, W = axial_n.shape
        inset_w = min(0.42, zoom_display_factor * zoom_w / max(W, 1))
        inset_h = min(0.42, zoom_display_factor * zoom_h / max(H, 1))
        inset_w = max(inset_w, 0.12)
        inset_h = max(inset_h, 0.12)
        ax_inset = ax.inset_axes([1.0 - inset_w - 0.01, 0.01, inset_w, inset_h])
        ax_inset.imshow(zoom_patch_n, cmap=cmap, aspect="auto")
        for spine in ax_inset.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

        # Metric overlay for non-GT columns only
        if metric_overlay_enabled and rdict is not None:
            _add_metric_overlay(
                ax=ax,
                result_dict=rdict,
                metric_keys=metric_overlay_metrics,
                text_color=metric_overlay_text_color,
                font_scale=metric_overlay_font_scale,
            )

    plt.suptitle("Visual Comparison: Methods vs Ground Truth", fontsize=10, y=1.01)
    plt.tight_layout(pad=0.5)
    return fig


def _unique_path(path: str) -> str:
    """Return a non-existing path by appending _N before the extension if needed."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    idx = 2
    while True:
        candidate = f"{root}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _save_named_magnitude_volume(label: str, volume: np.ndarray, out_dir: str) -> Optional[str]:
    mag = _drop_leading_volume_dims(volume)
    if mag.ndim != 3:
        logging.warning("Skipping magnitude volume upload for %s with shape %s.", label, mag.shape)
        return None
    stem = _safe_filename(label, max_len=120) or "volume"
    path = _unique_path(os.path.join(out_dir, f"{stem}.npy"))
    np.save(path, mag.astype(np.float32, copy=False))
    return path


def _save_magnitude_volumes_for_upload(
    volumes_with_labels: List[Tuple[str, Optional[np.ndarray]]],
    out_dir: str,
    reference_volumes_with_labels: Optional[List[Tuple[str, Optional[np.ndarray]]]] = None,
) -> List[str]:
    """Save reference and reconstruction magnitude volumes as .npy files."""
    os.makedirs(out_dir, exist_ok=True)
    saved_paths: List[str] = []

    for label, vol in reference_volumes_with_labels or []:
        if vol is None:
            continue
        path = _save_named_magnitude_volume(label, vol, out_dir)
        if path is not None:
            saved_paths.append(path)

    for label, vol in volumes_with_labels:
        if vol is None:
            continue
        method_name = _method_label_for_overlay(label, max_len=120)
        path = _save_named_magnitude_volume(method_name, vol, out_dir)
        if path is not None:
            saved_paths.append(path)

    return saved_paths


def _method_label_for_overlay(label: str, max_len: int = 28) -> str:
    """Extract a compact method label from a sweep label string."""
    # Try rec_method first
    m = re.search(r"(?:^|,\s*)(?:\+)?rec_method=([^,]+)", label)
    if m:
        method = m.group(1).strip()
        return method[:max_len]
    
    # Try exps
    m = re.search(r"(?:^|,\s*)(?:\+\+)?exps=([^,]+)", label)
    if m:
        method = m.group(1).strip()
        if "/" in method:
            method = method.split("/")[-1]
        return method[:max_len]
    
    return label[:max_len]


def _render_flythrough_side_by_side_video(
    gt_volume: Optional[np.ndarray],
    rec_volumes_with_labels: List[Tuple[str, Optional[np.ndarray]]],
    result_dicts: List[Dict[str, Any]],
    cmap: str,
    psnr_key: str,
    target_label: str,
    fps: int,
    frame_stride: int,
    z_start_frac: float,
    z_end_frac: float,
    panel_height_px: int,
    panel_width_px: int,
    video_dpi: int,
    video_annotation_font_scale: float,
    video_format: str,
) -> Optional[wandb.Video]:
    """Build one synchronized side-by-side fly-through video over z-slices."""
    cols: List[Tuple[str, np.ndarray, Optional[float], bool]] = []

    for (label, vol), rdict in zip(rec_volumes_with_labels, result_dicts):
        if vol is None:
            continue
        mag = _to_magnitude(vol)
        while mag.ndim > 3:
            mag = mag[0]
        if mag.ndim != 3:
            continue
        psnr_val: Optional[float] = None
        if psnr_key in rdict:
            try:
                psnr_val = float(rdict[psnr_key])
            except (TypeError, ValueError):
                psnr_val = None
        cols.append((_method_label_for_overlay(label), mag, psnr_val, False))

    if gt_volume is not None:
        gt_mag = _to_magnitude(gt_volume)
        while gt_mag.ndim > 3:
            gt_mag = gt_mag[0]
        if gt_mag.ndim == 3:
            cols.append((target_label, gt_mag, None, True))

    if not cols:
        return None

    min_depth = min(v.shape[0] for _label, v, _psnr, _is_target in cols)
    if min_depth <= 0:
        return None

    z_start = int(np.floor(float(z_start_frac) * min_depth))
    z_end = int(np.ceil(float(z_end_frac) * min_depth))
    z_start = max(0, min(min_depth - 1, z_start))
    z_end = max(z_start + 1, min(min_depth, z_end))

    z_indices = list(range(z_start, z_end, max(1, frame_stride)))
    if not z_indices:
        z_indices = [z_start]
    n_cols = len(cols)
    fig_w = max(6.0, n_cols * (panel_width_px / video_dpi))
    fig_h = max(3.0, panel_height_px / video_dpi)

    fontsize = max(6, int(9 * video_annotation_font_scale))
    frames: List[np.ndarray] = []

    for z_idx in z_indices:
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), dpi=video_dpi, squeeze=False)
        axes_row = axes[0]

        for ax, (ann_label, vol, psnr_val, is_target) in zip(axes_row, cols):
            sl = _normalize(vol[z_idx, :, :])
            ax.imshow(sl, cmap=cmap, aspect="auto", interpolation="nearest")
            ax.axis("off")

            ax.text(
                0.02,
                0.02,
                ann_label,
                transform=ax.transAxes,
                color="white",
                fontsize=fontsize,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55),
            )

            if not is_target:
                psnr_text = f"PSNR: {psnr_val:.2f}" if psnr_val is not None else "PSNR: n/a"
                ax.text(
                    0.98,
                    0.02,
                    psnr_text,
                    transform=ax.transAxes,
                    color="white",
                    fontsize=fontsize,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55),
                )

        fig.suptitle(f"Z Fly-through (slice {z_idx + 1}/{min_depth})", fontsize=max(8, fontsize))
        fig.tight_layout(pad=0.05)
        fig.canvas.draw()
        h, w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(frame)
        plt.close(fig)

    if not frames:
        return None

    video_arr = np.stack(frames, axis=0).astype(np.uint8, copy=False)  # T,H,W,3
    video_arr = np.moveaxis(video_arr, -1, 1)  # T,3,H,W (W&B-native)
    video_arr = np.ascontiguousarray(video_arr)
    fmt = str(video_format).lower()
    if fmt not in {"webm", "mp4", "gif"}:
        logging.warning("Unsupported flythrough_video_format=%s, falling back to webm", video_format)
        fmt = "webm"
    return wandb.Video(video_arr, fps=max(1, fps), format=fmt)


def _create_zip(paths: List[str], zip_path: str) -> str:
    """Zip a list of file paths into `zip_path`. Returns zip_path."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return zip_path


def _match_grouping_key_to_column(grouping_key: Optional[str], df_sweep: pd.DataFrame) -> Optional[str]:
    """Match a grouping key (with optional '+' / '=' prefix) to a column in df_sweep.
    Uses the same loose matching as _iter_split_groups in the flow.
    """
    if not grouping_key or df_sweep.empty:
        return None
    
    cols = df_sweep.columns.tolist()
    if grouping_key in cols:
        return grouping_key
    
    # Try stripping leading '+' / '~' / '='
    key_bare = grouping_key.lstrip("+~").lstrip("=")
    # Exact match after stripping
    if key_bare in cols:
        return key_bare
    candidates = [c for c in cols if c == key_bare or c.endswith(key_bare)]
    if candidates:
        return candidates[0]

    return None


def _local_disagreement_variance(patches: List[np.ndarray]) -> float:
    """Score local disagreement/variance across method patches.
    patches: list of 2-D arrays (one per method), all same shape
    Returns: mean variance (higher = more disagreement)
    """
    if len(patches) < 2:
        return 0.0
    patches = [p.astype(float) for p in patches]
    stacked = np.stack(patches, axis=0)  # (n_methods, h, w)
    # Variance across methods for each pixel, then mean
    variance = np.var(stacked, axis=0)  # (h, w)
    return float(np.mean(variance))


def _select_best_zoom_window_by_disagreement(
    volumes_with_labels: List[Tuple[str, Optional[np.ndarray]]],
    axial_slice_frac: float,
    zoom_size_frac: float,
    search_stride: int = 1,
) -> Optional[Tuple[int, int, int, int]]:
    """Automatically select a zoom window that maximizes disagreement across
    reconstruction methods for a given group (e.g. one acceleration level).
    """
    # Extract axial slices from all available volumes
    axial_slices: List[np.ndarray] = []
    for label, vol in volumes_with_labels:
        if vol is None:
            continue
        mag = _to_magnitude(vol)
        while mag.ndim > 3:
            mag = mag[0]
        if mag.ndim != 3:
            continue
        axial = _get_slice(mag, axis=0, frac=axial_slice_frac)
        axial_norm = _normalize(axial)
        axial_slices.append(axial_norm)

    if len(axial_slices) < 2:
        # Not enough methods to compute disagreement; fall back
        logging.info("Not enough methods (%d) for auto-zoom disagreement scoring.", len(axial_slices))
        return None

    shape_counts = Counter(s.shape for s in axial_slices)
    if len(shape_counts) > 1:
        ref_shape = max(shape_counts.items(), key=lambda item: (item[1], item[0]))[0]
        logging.warning(
            "Auto-zoom group mixes in-plane sizes %s; scoring on %s only and "
            "dropping %d slice(s). Consider grouping on the resolution axis.",
            dict(shape_counts), ref_shape, len(axial_slices) - shape_counts[ref_shape],
        )
        axial_slices = [s for s in axial_slices if s.shape == ref_shape]
        if len(axial_slices) < 2:
            return None

    h, w = axial_slices[0].shape
    half = max(1, int(zoom_size_frac * min(h, w) / 2))
    best_score = -1.0
    best_box: Optional[Tuple[int, int, int, int]] = None

    # Exhaustive or strided search over candidate window centers
    for cy in range(half, h - half, max(1, search_stride)):
        for cx in range(half, w - half, max(1, search_stride)):
            # Extract patches from all slices at this position
            patches = []
            for axial in axial_slices:
                r0 = max(0, cy - half)
                r1 = min(h, cy + half)
                c0 = max(0, cx - half)
                c1 = min(w, cx + half)
                patch = axial[r0:r1, c0:c1]
                patches.append(patch)

            # Score this candidate window
            score = _local_disagreement_variance(patches)
            if score > best_score:
                best_score = score
                r0 = max(0, cy - half)
                r1 = min(h, cy + half)
                c0 = max(0, cx - half)
                c1 = min(w, cx + half)
                best_box = (r0, r1, c0, c1)

    if best_box is not None:
        logging.info(
            "Auto-zoom selected best window at (cy=%d, cx=%d) with disagreement score=%.6f",
            (best_box[0] + best_box[1]) // 2,
            (best_box[2] + best_box[3]) // 2,
            best_score,
        )
    return best_box


# prefect task

@task(
    cache_policy=NONE,
    name="3D MRI Visual Examples",
    tags=["plot", "visual"],
    version="1.0",
    retries=0,
    description=(
        "Renders axial (main) + cropped coronal / sagittal ROI slices + zoom cube "
        "and assembles a comparison grid across all sweep configurations."
    ),
)
def plot_3d_mri_visual_task(
    results: List[CacheableDict],
    df_sweep: pd.DataFrame,
    wandb_params_task: WandbParamsTask,
    # --- from hydra config ---
    result_storage_path_key: str = "storage_path",
    result_storage_settings_key: str = "storage_settings",
    axial_slice_frac: float = 0.5,
    coronal_slice_frac: float = 0.5,
    sagittal_slice_frac: float = 0.5,
    zoom_center_frac_h: float = 0.5,
    zoom_center_frac_w: float = 0.5,
    zoom_size_frac: float = 0.10,
    zoom_display_factor: float = 2.0,
    cmap: str = "gray",
    metric_overlay_enabled: bool = True,
    metric_overlay_metrics: Optional[List[str]] = None,
    metric_overlay_text_color: str = "white",
    metric_overlay_font_scale: float = 0.6,
    shared_norm_percentile_low: float = 0.5,
    shared_norm_percentile_high: float = 99.5,
    # --- Automatic zoom window selection ---
    auto_zoom_enabled: bool = False,
    auto_zoom_grouping_key: Optional[str] = None,
    auto_zoom_scoring_mode: str = "disagreement_only",
    auto_zoom_search_stride: int = 2,
    auto_zoom_fallback_to_manual: bool = True,
    # --- GT / comparison strip ---
    gt_volume_filename_template: str = "ground_truth_{sample_idx}.npy",
    pseudoinverse_volume_filename_template: str = "filtbackproj_{sample_idx}.pt",
    comparison_figure_enabled: bool = True,
    flythrough_video_enabled: bool = False,
    flythrough_video_fps: int = 12,
    flythrough_frame_stride: int = 1,
    flythrough_z_start_frac: float = 0.0,
    flythrough_z_end_frac: float = 1.0,
    flythrough_psnr_key: str = "rec_psnr",
    flythrough_target_label: str = "Target",
    flythrough_panel_height_px: int = 256,
    flythrough_panel_width_px: int = 256,
    flythrough_video_dpi: int = 100,
    flythrough_video_annotation_font_scale: float = 1.0,
    flythrough_video_format: str = "webm",
    upload_magnitude_volumes_enabled: bool = False,
    upload_magnitude_volumes_artifact_name: str = "magnitude_volumes",
    upload_magnitude_volumes_dirname: str = "magnitude_volumes",
    dpi: int = 150,
) -> None:
    """For every result, load the saved 3-D reconstruction numpy and render a
    2x2 perspective figure (axial full + secondary/tertiary ROI crops + zoom cube).
    Saves individual panel PNGs and combined PNGs and uploads two zip archives to W&B.
    """
    if metric_overlay_metrics is None:
        metric_overlay_metrics = ["rec_psnr", "rec_ssim"]
    else:
        metric_overlay_metrics = list(metric_overlay_metrics)

    for visual_metric in ["rec_ssim3d", "rec_ssim2d_avg"]:
        if visual_metric not in metric_overlay_metrics:
            metric_overlay_metrics.append(visual_metric)

    result_dicts = [_result_to_dict(r) for r in results]

    with locked_wandb_init(**wandb_kwargs_for_prefect_task(wandb_params_task)):

        # Determine zoom box strategy
        # Either one shared box (manual) or per-group mapping (auto)
        zoom_box: Optional[Tuple[int, int, int, int]] = None
        zoom_boxes_by_group: Dict[Any, Tuple[int, int, int, int]] = {}
        group_col: Optional[str] = None

        if auto_zoom_enabled:
            # Auto-zoom: one box per group, or one shared auto-selected box
            # when no grouping key is provided/matched.
            group_col = _match_grouping_key_to_column(auto_zoom_grouping_key, df_sweep)
            if group_col is None:
                logging.info(
                    "Auto-zoom mode enabled without grouping; selecting one shared window "
                    "across all visual results."
                )
                grouped_indices = [("__all__", list(range(len(result_dicts))))]
            else:
                logging.info("Auto-zoom mode enabled with grouping on column: %s", group_col)
                grouped_indices = []
                for group_val in df_sweep[group_col].unique():
                    mask = df_sweep[group_col] == group_val
                    indices = [pos for pos, keep in enumerate(mask.to_numpy()) if keep]
                    grouped_indices.append((group_val, indices))

            for group_val, indices in grouped_indices:
                # Gather volumes for this group for disagreement scoring
                group_vols: List[Tuple[str, Optional[np.ndarray]]] = []
                for idx in indices:
                    if idx < len(result_dicts):
                        rdict = result_dicts[idx]
                        storage_path = rdict.get(result_storage_path_key)
                        storage_settings = rdict.get(result_storage_settings_key)
                        if storage_path is not None and storage_settings is not None:
                            vol = _try_download_volume(
                                storage_path, storage_settings,
                                sample_idx=int(rdict.get("sample_idx", 0))
                            )
                            label = df_sweep.iloc[idx].to_dict() if idx < len(df_sweep) else f"result_{idx}"
                            group_vols.append((str(label), vol))

                # Select best window for this group
                best_box = None
                if group_vols:
                    best_box = _select_best_zoom_window_by_disagreement(
                        group_vols,
                        axial_slice_frac=axial_slice_frac,
                        zoom_size_frac=zoom_size_frac,
                        search_stride=auto_zoom_search_stride,
                    )

                # Fallback to manual if auto selection failed
                if best_box is None:
                    if group_vols and group_vols[0][1] is not None:
                        vol = group_vols[0][1]
                        mag = _to_magnitude(vol)
                        while mag.ndim > 3:
                            mag = mag[0]
                        if mag.ndim >= 3:
                            h, w = mag.shape[1], mag.shape[2]
                            best_box = _compute_zoom_box(
                                h, w, zoom_center_frac_h,
                                zoom_center_frac_w, zoom_size_frac
                            )
                            logging.info(
                                "Auto-zoom selection failed for group=%s; "
                                "using manual fallback box=%s", group_val, best_box
                            )

                if best_box is not None:
                    zoom_boxes_by_group[group_val] = best_box
                    logging.info("Group %s: zoom_box=%s", group_val, best_box)

        if not auto_zoom_enabled:
            # Manual mode: single shared box for all results
            for i, rdict in enumerate(result_dicts):
                storage_path = rdict.get(result_storage_path_key)
                storage_settings = rdict.get(result_storage_settings_key)
                if storage_path is None or storage_settings is None:
                    continue
                _sidx = int(rdict.get("sample_idx", 0))
                vol = _try_download_volume(storage_path, storage_settings, sample_idx=_sidx)
                if vol is None:
                    continue
                mag = _to_magnitude(vol)
                if mag.ndim >= 3:
                    h, w = mag.shape[1], mag.shape[2]
                else:
                    continue
                zoom_box = _compute_zoom_box(h, w, zoom_center_frac_h, zoom_center_frac_w, zoom_size_frac)
                logging.info("Zoom box determined from result %d: (h=%d, w=%d) -> box=%s", i, h, w, zoom_box)
                break

        if zoom_box is None and not auto_zoom_enabled:
            logging.warning("No downloadable volumes found; logging only a metrics table.")
            if not df_sweep.empty:
                df_metrics = pd.DataFrame(result_dicts)
                df_table = pd.concat([df_metrics, df_sweep.reset_index(drop=True)], axis=1)
                wandb.log({"3d_visual_metrics": wandb.Table(dataframe=df_table)})
            return

        if auto_zoom_enabled and not zoom_boxes_by_group:
            logging.warning("Auto-zoom enabled but no groups found; logging only a metrics table.")
            if not df_sweep.empty:
                df_metrics = pd.DataFrame(result_dicts)
                df_table = pd.concat([df_metrics, df_sweep.reset_index(drop=True)], axis=1)
                wandb.log({"3d_visual_metrics": wandb.Table(dataframe=df_table)})
            return

        # Load GT / pseudoinverse volumes for comparison and upload
        gt_volume: Optional[np.ndarray] = None
        pseudoinverse_volume: Optional[np.ndarray] = None
        for i, rdict in enumerate(result_dicts):
            storage_path_gt = rdict.get(result_storage_path_key)
            storage_settings_gt = rdict.get(result_storage_settings_key)
            if storage_path_gt is None or storage_settings_gt is None:
                continue
            gt_volume = _try_download_gt_volume(
                storage_path=storage_path_gt,
                storage_settings=storage_settings_gt,
                sample_idx=int(rdict.get("sample_idx", 0)),
                filename_template=gt_volume_filename_template,
            )
            if gt_volume is not None:
                logging.info("GT volume loaded from result %d for comparison strip / visual SSIM metrics.", i)
                break
        if gt_volume is None:
            logging.warning(
                "GT volume not found (template=%s); comparison strip and visual SSIM metrics will skip GT comparison.",
                gt_volume_filename_template,
            )

        for i, rdict in enumerate(result_dicts):
            storage_path_pi = rdict.get(result_storage_path_key)
            storage_settings_pi = rdict.get(result_storage_settings_key)
            if storage_path_pi is None or storage_settings_pi is None:
                continue
            pseudoinverse_volume = _try_download_pseudoinverse_volume(
                storage_path=storage_path_pi,
                storage_settings=storage_settings_pi,
                sample_idx=int(rdict.get("sample_idx", 0)),
                filename_template=pseudoinverse_volume_filename_template,
            )
            if pseudoinverse_volume is not None:
                logging.info("Pseudoinverse volume loaded from result %d for magnitude upload.", i)
                break

        # Build per-result figures
        per_result_figs: List[Tuple[str, Optional["plt.Figure"], Optional[Dict[str, np.ndarray]]]] = []
        per_result_figs_az: List[Tuple[str, Optional["plt.Figure"]]] = []  # axial+zoom view
        per_result_volumes: List[Tuple[str, Optional[np.ndarray]]] = []  # for comparison strip
        for i, rdict in enumerate(result_dicts):
            if i < len(df_sweep):
                row = df_sweep.iloc[i]
                label = ", ".join(f"{k}={v}" for k, v in row.items())
            else:
                label = f"result_{i}"

            storage_path = rdict.get(result_storage_path_key)
            storage_settings = rdict.get(result_storage_settings_key)

            # Determine which zoom_box to use for this result
            result_zoom_box = zoom_box
            if auto_zoom_enabled:
                if group_col is not None and i < len(df_sweep):
                    group_val = df_sweep.iloc[i][group_col]
                    result_zoom_box = zoom_boxes_by_group.get(group_val, zoom_box)
                elif zoom_boxes_by_group:
                    result_zoom_box = next(iter(zoom_boxes_by_group.values()), zoom_box)

            fig = None
            panels = None
            fig_az = None
            vol = None
            if storage_path is not None and storage_settings is not None and result_zoom_box is not None:
                vol = _try_download_volume(storage_path, storage_settings, sample_idx=int(rdict.get("sample_idx", 0)))
                if vol is not None and _to_magnitude(vol).ndim >= 3:
                    mag = _to_magnitude(vol)
                    while mag.ndim > 3:
                        mag = mag[0]
                    fig, panels = _render_three_panels(
                        mag,
                        axial_slice_frac=axial_slice_frac,
                        coronal_slice_frac=coronal_slice_frac,
                        sagittal_slice_frac=sagittal_slice_frac,
                        zoom_box=result_zoom_box,
                        zoom_display_factor=zoom_display_factor,
                        cmap=cmap,
                        shared_norm_percentile_low=shared_norm_percentile_low,
                        shared_norm_percentile_high=shared_norm_percentile_high,
                    )
                    fig_az = _render_axial_zoom(
                        mag,
                        axial_slice_frac=axial_slice_frac,
                        zoom_box=result_zoom_box,
                        zoom_display_factor=zoom_display_factor,
                        cmap=cmap,
                    )
                    if metric_overlay_enabled:
                        # Metric overlay on lower-left of the main (axial) panel
                        _add_metric_overlay(
                            ax=fig.axes[0],
                            result_dict=rdict,
                            metric_keys=metric_overlay_metrics,
                            text_color=metric_overlay_text_color,
                            font_scale=metric_overlay_font_scale,
                        )
                        # Same overlay on the axial+zoom figure
                        _add_metric_overlay(
                            ax=fig_az.axes[0],
                            result_dict=rdict,
                            metric_keys=metric_overlay_metrics,
                            text_color=metric_overlay_text_color,
                            font_scale=metric_overlay_font_scale,
                        )

            per_result_figs.append((label, fig, panels))
            per_result_figs_az.append((label, fig_az))
            # Collect volume for the side-by-side comparison strip
            if fig is not None and vol is not None and _to_magnitude(vol).ndim >= 3:
                strip_mag = _to_magnitude(vol).copy()
                while strip_mag.ndim > 3:
                    strip_mag = strip_mag[0]
                per_result_volumes.append((label, strip_mag))
            else:
                per_result_volumes.append((label, None))

        # Save PNGs and create zip archives
        with tempfile.TemporaryDirectory() as tmp_dir:
            individual_dir = os.path.join(tmp_dir, "individual")
            combined_dir = os.path.join(tmp_dir, "combined")
            os.makedirs(individual_dir, exist_ok=True)
            os.makedirs(combined_dir, exist_ok=True)

            all_individual_paths: List[str] = []
            all_combined_paths: List[str] = []

            for label, fig, panels in per_result_figs:
                if fig is None:
                    continue
                # individual panels
                if panels is not None:
                    ind_paths = _save_individual_pngs(label, panels, individual_dir, cmap=cmap, dpi=dpi)
                    all_individual_paths.extend(ind_paths)
                # combined figure
                combined_path = _save_combined_png(label, fig, combined_dir, dpi=dpi)
                all_combined_paths.append(combined_path)

            zip_individual = os.path.join(tmp_dir, "panels_individual.zip")
            zip_combined = os.path.join(tmp_dir, "panels_combined.zip")
            if all_individual_paths:
                _create_zip(all_individual_paths, zip_individual)
                art_ind = wandb.Artifact("panels_individual", type="visual_panels")
                art_ind.add_file(zip_individual, name="panels_individual.zip")
                wandb.log_artifact(art_ind)
            if all_combined_paths:
                _create_zip(all_combined_paths, zip_combined)
                art_comb = wandb.Artifact("panels_combined", type="visual_panels")
                art_comb.add_file(zip_combined, name="panels_combined.zip")
                wandb.log_artifact(art_comb)

            if upload_magnitude_volumes_enabled and per_result_volumes:
                volume_dir = os.path.join(tmp_dir, upload_magnitude_volumes_dirname)
                reference_volumes = [
                    ("ground_truth", gt_volume),
                    ("pseudoinverse", pseudoinverse_volume),
                ]
                volume_paths = _save_magnitude_volumes_for_upload(
                    per_result_volumes,
                    volume_dir,
                    reference_volumes_with_labels=reference_volumes,
                )
                if volume_paths:
                    zip_volumes = os.path.join(tmp_dir, f"{upload_magnitude_volumes_dirname}.zip")
                    _create_zip(volume_paths, zip_volumes)
                    art_vol = wandb.Artifact(upload_magnitude_volumes_artifact_name, type="magnitude_volumes")
                    art_vol.add_file(zip_volumes, name=f"{upload_magnitude_volumes_dirname}.zip")
                    wandb.log_artifact(art_vol)
                else:
                    logging.warning("Magnitude volume upload enabled, but no valid volumes were available.")

            # Render individual result figures to W&B images
            for idx, (label, fig, panels) in enumerate(per_result_figs):
                if fig is not None:
                    safe_label = re.sub(r"[^\w\-,=.]", "_", label)[:60]
                    wandb.log({f"3d_visual/result_{idx:03d}__{safe_label}": wandb.Image(fig, caption=label)})
                    plt.close(fig)
            for idx, (label, fig_az) in enumerate(per_result_figs_az):
                if fig_az is not None:
                    safe_label = re.sub(r"[^\w\-,=.]", "_", label)[:60]
                    wandb.log({f"3d_visual_axzoom/result_{idx:03d}__{safe_label}": wandb.Image(fig_az, caption=label)})
                    plt.close(fig_az)

            # Side-by-side comparison strip (all methods + GT)
            if comparison_figure_enabled and per_result_volumes:
                # For comparison strip, use the zoom box. In auto mode with multiple groups,
                # use the first available group box (typical case is single group per split run).
                strip_zoom_box = zoom_box
                if auto_zoom_enabled and zoom_boxes_by_group:
                    strip_zoom_box = next(iter(zoom_boxes_by_group.values()), None)
                
                if strip_zoom_box is not None:
                    fig_strip = _render_comparison_strip(
                        gt_volume=gt_volume,
                        rec_volumes_with_labels=per_result_volumes,
                        result_dicts=result_dicts,
                        axial_slice_frac=axial_slice_frac,
                        zoom_box=strip_zoom_box,
                        zoom_display_factor=zoom_display_factor,
                        cmap=cmap,
                        metric_overlay_enabled=metric_overlay_enabled,
                        metric_overlay_metrics=metric_overlay_metrics,
                        metric_overlay_text_color=metric_overlay_text_color,
                        metric_overlay_font_scale=metric_overlay_font_scale,
                        dpi=dpi,
                    )
                    wandb.log({"3d_visual_comparison": wandb.Image(fig_strip, caption="Side-by-side: GT vs methods")})
                    plt.close(fig_strip)

            # Side-by-side z-flythrough video (methods + target on right)
            if flythrough_video_enabled and per_result_volumes:
                flythrough_video = _render_flythrough_side_by_side_video(
                    gt_volume=gt_volume,
                    rec_volumes_with_labels=per_result_volumes,
                    result_dicts=result_dicts,
                    cmap=cmap,
                    psnr_key=flythrough_psnr_key,
                    target_label=flythrough_target_label,
                    fps=flythrough_video_fps,
                    frame_stride=flythrough_frame_stride,
                    z_start_frac=flythrough_z_start_frac,
                    z_end_frac=flythrough_z_end_frac,
                    panel_height_px=flythrough_panel_height_px,
                    panel_width_px=flythrough_panel_width_px,
                    video_dpi=flythrough_video_dpi,
                    video_annotation_font_scale=flythrough_video_annotation_font_scale,
                    video_format=flythrough_video_format,
                )
                if flythrough_video is not None:
                    wandb.log({"3d_visual_flythrough_comparison": flythrough_video})
                else:
                    logging.warning("Could not render flythrough comparison video (no valid volumes).")

        # Assemble combined comparison grid
        panel_order = ["axial_full", "sagittal_panel", "coronal_panel", "zoom_patch"]
        panel_labels = ["Axial", "Sagittal", "Coronal", "Zoom"]

        method_entries: List[Tuple[str, Optional[Dict[str, np.ndarray]], Dict[str, Any]]] = []
        for idx, (label, _fig_single, panels_dict) in enumerate(per_result_figs):
            sweep_row = dict(df_sweep.iloc[idx]) if idx < len(df_sweep) else {}
            method_entries.append((label, panels_dict, sweep_row))

        if not method_entries:
            return

        row_key, col_key, row_values, col_values = _infer_visual_grid_axes(df_sweep)

        # Fallback to a simple sequential layout if no varying sweep axis exists.
        if not row_values:
            row_values = ["all"]
        if not col_values:
            col_values = ["all"]

        # Map entries into a 2-D grid by inferred sweep axes.
        grid_cells: Dict[Tuple[Any, Any], Tuple[str, Optional[Dict[str, np.ndarray]]]] = {}
        for label, panels_dict, sweep_row in method_entries:
            row_val = sweep_row.get(row_key, "all") if row_key is not None else "all"
            col_val = sweep_row.get(col_key, "all") if col_key is not None else "all"
            key = (row_val, col_val)
            if key in grid_cells:
                logging.warning("Duplicate visual grid cell for row=%s col=%s; keeping first entry.", row_val, col_val)
                continue
            grid_cells[key] = (label, panels_dict)

        grid_rows = len(row_values)
        grid_cols = len(col_values)
        fig_grid = plt.figure(
            figsize=(max(5, grid_cols) * 5, max(1, grid_rows) * 5),
            dpi=dpi,
            facecolor="black",
        )
        outer_gs = fig_grid.add_gridspec(
            grid_rows, grid_cols,
            hspace=0.0, wspace=0.0,
            left=0, right=1, top=1, bottom=0,
        )

        row_name = _canonical_param_name(row_key) if row_key is not None else "row"
        col_name = _canonical_param_name(col_key) if col_key is not None else "group"

        for outer_r, row_val in enumerate(row_values):
            for outer_c, col_val in enumerate(col_values):
                label, panels_dict = grid_cells.get((row_val, col_val), ("missing", None))

                if panels_dict is not None and all(k in panels_dict for k in panel_order):
                    H, W = panels_dict["axial_full"].shape
                    # sagittal_panel shape: H x d_nbhd, coronal_panel shape: d_nbhd x W
                    d_nbhd = panels_dict["sagittal_panel"].shape[1]
                else:
                    H, W, d_nbhd = 1, 1, 1

                inner_gs = outer_gs[outer_r, outer_c].subgridspec(
                    2, 2,
                    width_ratios=[W, d_nbhd],
                    height_ratios=[H, d_nbhd],
                    hspace=0.0,
                    wspace=0.0,
                )

                sub_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for (dr, dc), pname, plabel in zip(sub_positions, panel_order, panel_labels):
                    ax = fig_grid.add_subplot(inner_gs[dr, dc])
                    ax.axis("off")
                    ax.set_facecolor("black")
                    if panels_dict is not None and pname in panels_dict:
                        ax.imshow(
                            panels_dict[pname],
                            cmap=cmap,
                            aspect="auto",
                            vmin=0.0,
                            vmax=1.0,
                            interpolation="nearest",
                        )

                    if dr == 0 and dc == 0:
                        cell_title = f"{row_name}={row_val} | {col_name}={col_val}"
                        ax.text(
                            0.02,
                            0.98,
                            cell_title[:60],
                            transform=ax.transAxes,
                            color="white",
                            fontsize=5,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.55),
                        )
                    else:
                        ax.text(
                            0.02,
                            0.98,
                            plabel,
                            transform=ax.transAxes,
                            color="white",
                            fontsize=5,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.55),
                        )

        wandb.log({"3d_visual_grid": wandb.Image(fig_grid)})
        plt.close(fig_grid)

        # Also log a metrics summary table
        df_metrics = pd.DataFrame(result_dicts)
        if not df_metrics.empty and not df_sweep.empty:
            df_table = pd.concat([df_metrics.reset_index(drop=True), df_sweep.reset_index(drop=True)], axis=1)
            wandb.log({"3d_visual_metrics_table": wandb.Table(dataframe=df_table)})

        logging.info("plot_3d_mri_visual_task completed for %d results.", len(per_result_figs))
