# %%
from typing import Any, Dict, Optional, Tuple, Optional

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.representations.mesh import SliceableMesh
import time

import wandb
from tqdm import tqdm

from src.reconstruction.utils.metrics import PSNR, PSNR_2D, SSIM, VIFP, LPIPS, DISTS, volume_ssim_metrics, volume_nce_metrics, _center_window_bounds, FOREGROUND_MASK_THRESHOLD
from src.utils.wandb_utils import tensor_to_wandbimages_dict

import logging

from .base_sample_logger import BaseSampleLogger

class SampleLoggerWithTarget(BaseSampleLogger):
    def __init__(
        self,
        device: str,
        devices: str,
        target_trafo: Any,
        fwd_trafo: Any,
        show_phase: bool,
        sample_gen_split: int,
        volume_stats_period: int,
        medslice_stats_period: int,
        volume_stats__wandb_take_mean_slice_period: int = 5,
        volume_stats__wandb_video_period: int = 300,
        take_abs_normalize: bool = False,
        log_psnr=True,
        log_ssim=False,
        log_vifp=False,
        log_lpips=False,
        log_dists=False,
        foreach_data=True,
        final_reco=False,
        save_observation=False,
        save_filtbackproj=False,
        save_ground_truth=False,
        save_final_sample=False,
        store_k3d_volume=False,
        log_gt_fbp_to_wandb=False,
        use_second_cuda=False,
        log_data_is_complex=True,
        log_3d_include_final_slice_averages_for_volumes=False,
        log_3d_slice_metrics_use_vol_max: bool = False,
        avg_metrics_center_frac: float = 0.5,
        log_ssim3d_fg: bool = False,
        fg_mask_threshold: float = FOREGROUND_MASK_THRESHOLD,
        fg_ssim_average_over_mask: bool = True,
    ):

        super().__init__()
        self.device = device
        self.devices = devices
        self.target_trafo = target_trafo
        self.fwd_trafo = fwd_trafo

        self.show_phase = show_phase
        self.sample_gen_split = sample_gen_split
        self.volume_stats_period = volume_stats_period
        self.medslice_stats_period = medslice_stats_period
        self.volume_stats__wandb_take_mean_slice_period = (
            volume_stats__wandb_take_mean_slice_period
        )
        self.volume_stats__wandb_video_period = volume_stats__wandb_video_period
        self.take_abs_normalize = take_abs_normalize
        self.log_psnr = log_psnr
        self.log_ssim = log_ssim
        self.log_vifp = log_vifp
        self.log_lpips = log_lpips
        self.log_dists = log_dists
        self.foreach_data = foreach_data
        self.final_reco = final_reco
        self.save_observation = save_observation
        self.save_filtbackproj = save_filtbackproj
        self.save_ground_truth = save_ground_truth
        self.save_final_sample = save_final_sample
        self.store_k3d_volume = store_k3d_volume
        self.log_gt_fbp_to_wandb = log_gt_fbp_to_wandb
        self.use_second_cuda = use_second_cuda
        self.log_3d_include_final_slice_averages_for_volumes = (
            log_3d_include_final_slice_averages_for_volumes
        )
        self.log_3d_slice_metrics_use_vol_max = log_3d_slice_metrics_use_vol_max
        self.log_data_is_complex = log_data_is_complex
        # fraction of slices (centered on the middle slice, per axis) used for
        # rec_lpips_avg / rec_ssim2d_avg, so edge slices don't dilute the score
        self.avg_metrics_center_frac = avg_metrics_center_frac
        self.log_ssim3d_fg = log_ssim3d_fg
        self.fg_mask_threshold = fg_mask_threshold
        self.fg_ssim_average_over_mask = fg_ssim_average_over_mask

        # tracks numpy files saved per sample (populated when save_final_sample=True)
        self._saved_final_sample_files: list = []

        # sample-dependent state - plain scalars (single-sample design)
        self.mesh = None
        self.ground_truth = None
        self.scaling_factor = None
        self.sample_nr = None
        self.time_start = None
        self.time: Optional[float] = None
        self.reff: Optional[float] = None
        self.reff_metrics: Dict[str, Any] = {}

        # per-metric scalars populated by init_sample_log / close_sample_log
        self.fbp_psnr: Optional[float] = None
        self.rec_psnr: Optional[float] = None
        self.fbp_ssim: Optional[float] = None
        self.rec_ssim: Optional[float] = None
        self.rec_ssim3d: Optional[float] = None
        self.rec_ssim3d_fg: Optional[float] = None
        self.rec_ssim2d_avg: Optional[float] = None
        self.fbp_vifp: Optional[float] = None
        self.rec_vifp: Optional[float] = None
        self.fbp_lpips: Optional[float] = None
        self.rec_lpips: Optional[float] = None
        self.rec_lpips_avg: Optional[float] = None
        self.fbp_dists: Optional[float] = None
        self.rec_dists: Optional[float] = None
        self.rec_nce: Optional[float] = None

    def _to_mag(self, t: "Tensor") -> "Tensor":
        """Return magnitude of t."""
        if self.log_data_is_complex and t.ndim >= 1 and t.shape[-1] == 2:
            return t.norm(dim=-1)
        return t

    def _lpips_all_slices_avg(self, rec: "Tensor", gt: "Tensor") -> float:
        """Average LPIPS over slices within a centered window (covering `self.avg_metrics_center_frac` of each axis) rather than the full volume, so it better reflects quality near the anatomically relevant center rather than being diluted by edge slices.
        """
        if rec.ndim == 3 and gt.ndim == 3:
            axis_scores = []
            for axis in range(3):
                rec_axis = rec.moveaxis(axis, 0)
                gt_axis = gt.moveaxis(axis, 0)
                start, end = _center_window_bounds(rec_axis.shape[0], self.avg_metrics_center_frac)
                axis_scores.append(
                    LPIPS(
                        rec_axis[start:end],
                        gt_axis[start:end],
                        center_slices_only=False,
                    )
                )
            return float(sum(axis_scores) / len(axis_scores))

        return float(LPIPS(rec, gt, center_slices_only=False))

    def init_run(self, num_samples: int = 1):
        """Called once before starting reconstruction.
        num_samples is accepted for API compatibility but ignored -
        this logger is single-sample only.
        """
        if self.use_second_cuda:
            free_devices = [d for d in self.devices if d != self.device]
            assert len(free_devices) > 0, "No free cuda devices available"
            self.eval_device = free_devices[0]
        else:
            self.eval_device = self.device
        logging.info("Using device %s", self.eval_device)

    def init_sample_log(
        self,
        observation: Tensor,
        filtbackproj: Tensor,
        ground_truth: Tensor,
        sample_nr: int,
        scaling_factor: float,
        mesh: SliceableMesh,
        attrs: Optional[Dict[str, Any]] = None,
    ):

        # save sample-fixed vars for later
        self.observation = observation
        self.ground_truth = ground_truth
        self.scaling_factor = scaling_factor
        self.mesh = mesh

        # optionally saving to disk
        if self.save_observation:
            torch.save(observation, f"observation_{sample_nr}.pt")
        if self.save_filtbackproj:
            torch.save(filtbackproj, f"filtbackproj_{sample_nr}.pt")
        if self.save_ground_truth:
            torch.save(ground_truth, f"ground_truth_{sample_nr}.pt")

        fbp_loss: float = float(
            torch.nn.functional.mse_loss(
                self.fwd_trafo(filtbackproj * scaling_factor),
                observation * scaling_factor,
            )
        )
        trafo_adjoint: Optional[Tensor] = None

        extra_dict = {}
        if self.log_gt_fbp_to_wandb:
            extra_dict = {
                **tensor_to_wandbimages_dict(
                    "ground_truth",
                    ground_truth.unsqueeze(0),
                    take_meanslices=True,
                    take_videos=False,
                    show_phase=self.show_phase,
                ),
                **tensor_to_wandbimages_dict(
                    "fbp",
                    filtbackproj.unsqueeze(0),
                    take_meanslices=True,
                    take_videos=False,
                    show_phase=self.show_phase,
                ),
            }

        trafo_adjoint = self.fwd_trafo.trafo_adjoint(observation)
        if self.log_psnr:
            self.fbp_psnr = float(PSNR(self.target_trafo(trafo_adjoint), ground_truth))
            extra_dict["fbp_psnr"] = self.fbp_psnr
        if self.log_ssim:
            _fbp_adj = self.target_trafo(trafo_adjoint)
            fbp_ssim = self._to_mag(_fbp_adj) if _fbp_adj.shape[-1] == 2 else _fbp_adj / self.scaling_factor
            gt_ssim = self._to_mag(ground_truth)
            self.fbp_ssim = float(SSIM(fbp_ssim, gt_ssim, take_abs_normalize=self.take_abs_normalize)[0])
            extra_dict["fbp_ssim"] = self.fbp_ssim
        if self.log_vifp:
            _fbp_adj = self.target_trafo(trafo_adjoint)
            fbp_vifp_rec = self._to_mag(_fbp_adj) if _fbp_adj.shape[-1] == 2 else _fbp_adj / self.scaling_factor
            fbp_vifp_gt = self._to_mag(ground_truth)
            self.fbp_vifp = float(VIFP(fbp_vifp_rec, fbp_vifp_gt))
            extra_dict["fbp_vifp"] = self.fbp_vifp
        if self.log_lpips:
            _fbp_adj = self.target_trafo(trafo_adjoint)
            _fbp_mag = self._to_mag(_fbp_adj)
            _gt_mag = self._to_mag(ground_truth)
            self.fbp_lpips = float(LPIPS(_fbp_mag, _gt_mag))
            extra_dict["fbp_lpips"] = self.fbp_lpips
        if self.log_dists:
            _fbp_adj = self.target_trafo(trafo_adjoint)
            _fbp_mag = self._to_mag(_fbp_adj)
            _gt_mag = self._to_mag(ground_truth)
            self.fbp_dists = float(DISTS(_fbp_mag, _gt_mag))
            extra_dict["fbp_dists"] = self.fbp_dists

        self.reff = None
        self.reff_metrics = {}
        reff_metrics = self._extract_effective_acceleration_metrics(attrs)
        if reff_metrics:
            self.reff_metrics = dict(reff_metrics)
            if "Reff" in reff_metrics:
                self.reff = float(reff_metrics["Reff"])
            extra_dict.update(reff_metrics)
            if wandb.run is not None:
                for key, value in reff_metrics.items():
                    wandb.run.summary[key] = value

        wandb.log(
            {
                "fbp_loss": fbp_loss,
                "scaling_factor": scaling_factor,
                "global_step": sample_nr,
                "step": sample_nr,
                **extra_dict,
            }
        )
        torch.cuda.empty_cache()

        self.sample_nr = sample_nr
        self.time_start = time.time()

    def __call__(
        self,
        representation: CoordBasedRepresentation,
        step: int,
        pbar: tqdm,
        log_dict: Dict = {},
    ):

        if not self.foreach_data:
            return

        if log_dict is not None and len(log_dict) > 0:
            wandb.log({"global_step": step, **log_dict})

        if step % self.volume_stats_period == 0:

            with torch.no_grad():

                sample = representation.forward_splitted(
                    self.mesh, self.eval_device, self.sample_gen_split
                )
                tf_sample = self.target_trafo(sample) / self.scaling_factor
                gt = self.ground_truth.cpu().to(self.eval_device)

                extra_log_dict = {}
                pbar_strs = []
                if self.log_psnr:
                    extra_log_dict["rec_psnr"] = PSNR(tf_sample, gt)
                    pbar_strs.append(f'rec_psnr={extra_log_dict["rec_psnr"]:.1f}')
                if self.log_ssim:
                    _rec_tf_ssim = self.target_trafo(sample)
                    tf_sample_ssim = self._to_mag(_rec_tf_ssim) / self.scaling_factor if _rec_tf_ssim.shape[-1] == 2 else _rec_tf_ssim / self.scaling_factor
                    gt_ssim = self._to_mag(self.ground_truth.cpu())
                    extra_log_dict["rec_ssim"] = SSIM(tf_sample_ssim, gt_ssim)[0]
                    pbar_strs.append(f', rec_ssim={extra_log_dict["rec_ssim"]:.2f}')
                if self.log_vifp:
                    _rec_tf_vifp = self.target_trafo(sample)
                    tf_sample_vifp = self._to_mag(_rec_tf_vifp) if _rec_tf_vifp.shape[-1] == 2 else _rec_tf_vifp / self.scaling_factor
                    gt_vifp = self._to_mag(self.ground_truth.cpu())
                    extra_log_dict["rec_vifp"] = VIFP(tf_sample_vifp, gt_vifp)
                    pbar_strs.append(f', rec_vifp={extra_log_dict["rec_vifp"]:.2f}')
                if self.log_lpips:
                    _mag = self._to_mag(tf_sample)
                    _gt_mag = self._to_mag(gt)
                    extra_log_dict["rec_lpips"] = LPIPS(_mag, _gt_mag)
                    pbar_strs.append(f', rec_lpips={extra_log_dict["rec_lpips"]:.3f}')
                if self.log_dists:
                    _mag = self._to_mag(tf_sample)
                    _gt_mag = self._to_mag(gt)
                    extra_log_dict["rec_dists"] = DISTS(_mag, _gt_mag)
                    pbar_strs.append(f', rec_dists={extra_log_dict["rec_dists"]:.3f}')

                pbar.set_description(",".join(pbar_strs), refresh=False)

                images = tensor_to_wandbimages_dict(
                    "reco",
                    sample.unsqueeze(0),
                    take_meanslices=step
                    % self.volume_stats__wandb_take_mean_slice_period
                    == 0
                    and step > 0,
                    take_videos=step % self.volume_stats__wandb_video_period == 0
                    and step > 0,
                    show_phase=self.show_phase,
                )

                wandb.log(
                    {
                        "rec_mean": sample.detach().cpu().numpy().mean(),
                        "rec_std": sample.detach().cpu().numpy().std(),
                        "global_step": step,
                        **(images),
                        **extra_log_dict,
                    }
                )

        if step % self.medslice_stats_period == 0:

            with torch.no_grad():
                sample_mean_slice = representation.forward(
                    self.mesh.add_index_select(
                        axis=0,
                        indices=torch.Tensor([self.mesh.matrix_size[0] // 2]).int(),
                    )
                )

                extra_log_dict = {}
                pbar_strs = []
                if self.log_psnr:
                    sample_mean_slice_psnr = PSNR_2D(
                        self.target_trafo(sample_mean_slice) / self.scaling_factor,
                        self.ground_truth[self.ground_truth.shape[0] // 2, ...][None]
                        .cpu()
                        .to(self.eval_device),
                        take_abs_normalize=self.take_abs_normalize,
                    )[0]
                    extra_log_dict["rec_medslice_psnr"] = sample_mean_slice_psnr
                    pbar_strs.append(
                        f'rec_medslice_psnr={extra_log_dict["rec_medslice_psnr"]:.1f}'
                    )
                if self.log_ssim:
                    _ms_tf_ssim = self.target_trafo(sample_mean_slice)
                    _ms_gt_ssim = self._to_mag(self.ground_truth[self.ground_truth.shape[0] // 2, ...].cpu())
                    sample_mean_slice_ssim = SSIM(
                        self._to_mag(_ms_tf_ssim) / self.scaling_factor if _ms_tf_ssim.shape[-1] == 2 else _ms_tf_ssim / self.scaling_factor,
                        _ms_gt_ssim,
                        take_abs_normalize=self.take_abs_normalize,
                        center_slices_only=False,
                    )[0]
                    extra_log_dict["rec_medslice_ssim"] = sample_mean_slice_ssim
                    pbar_strs.append(
                        f', rec_medslice_ssim={extra_log_dict["rec_medslice_ssim"]:.2f}'
                    )
                if self.log_vifp:
                    _ms_tf_vifp = self.target_trafo(sample_mean_slice)
                    _ms_gt_vifp = self._to_mag(self.ground_truth[self.ground_truth.shape[0] // 2, ...].cpu())
                    sample_mean_slice_vifp = VIFP(
                        self._to_mag(_ms_tf_vifp) if _ms_tf_vifp.shape[-1] == 2 else _ms_tf_vifp / self.scaling_factor,
                        _ms_gt_vifp,
                        center_slices_only=False,
                    )
                    extra_log_dict["rec_medslice_vifp"] = sample_mean_slice_vifp
                    pbar_strs.append(
                        f', rec_medslice_vifp={extra_log_dict["rec_medslice_vifp"]:.2f}'
                    )

                pbar.set_description(",".join(pbar_strs), refresh=False)

                wandb.log(
                    {
                        "global_step": step,
                        "rec_medslice_mean": sample_mean_slice.detach()
                        .cpu()
                        .numpy()
                        .mean(),
                        "rec_medslice_std": sample_mean_slice.detach()
                        .cpu()
                        .numpy()
                        .std(),
                        **(
                            tensor_to_wandbimages_dict(
                                "reco_medslice",
                                sample_mean_slice.unsqueeze(0),
                                show_phase=self.show_phase,
                            )
                        ),
                        **extra_log_dict,
                    }
                )

    def close_sample_log(self, representation: CoordBasedRepresentation):
        """
        Called once after reconstruction of a single sample.
        """
        self.time = time.time() - self.time_start if self.time_start is not None else None

        sample = representation.forward_splitted(
            self.mesh, self.eval_device, self.sample_gen_split
        )
        tf_sample = self.target_trafo(sample) / self.scaling_factor
        gt = self.ground_truth.cpu().to(self.eval_device)

        if not self.final_reco:
            return

        if self.save_final_sample:
            import numpy as np
            # filename uses sample_nr (global dataset index) so the visual task can find it
            npy_filename = f"final_rec_{self.sample_nr}.npy"
            np.save(npy_filename, tf_sample.detach().cpu().numpy())
            self._saved_final_sample_files.append(npy_filename)

        extra_dict = {}

        if self.log_ssim:
            tf_sample_mag = self._to_mag(tf_sample)
            gt_mag = self._to_mag(gt)
            self.rec_ssim = float(SSIM(
                tf_sample_mag, gt_mag,
                take_abs_normalize=self.take_abs_normalize,
            )[0])
            extra_dict["rec_ssim"] = self.rec_ssim
            extra_dict["fbp_ssim"] = self.fbp_ssim

            rec_volume_ssim = volume_ssim_metrics(
                tf_sample_mag,
                gt_mag,
                center_frac=self.avg_metrics_center_frac,
                fg_threshold=self.fg_mask_threshold if self.log_ssim3d_fg else None,
                fg_average_over_mask=self.fg_ssim_average_over_mask,
            )
            if "rec_ssim3d" in rec_volume_ssim:
                self.rec_ssim3d = rec_volume_ssim["rec_ssim3d"]
            if "rec_ssim3d_fg" in rec_volume_ssim:
                self.rec_ssim3d_fg = rec_volume_ssim["rec_ssim3d_fg"]
            if "rec_ssim2d_avg" in rec_volume_ssim:
                self.rec_ssim2d_avg = rec_volume_ssim["rec_ssim2d_avg"]
            extra_dict.update(rec_volume_ssim)

        if self.log_vifp:
            self.rec_vifp = float(VIFP(self._to_mag(tf_sample), self._to_mag(gt)))
            extra_dict["rec_vifp"] = self.rec_vifp
            extra_dict["fbp_vifp"] = self.fbp_vifp

        if self.log_lpips:
            tf_sample_mag = self._to_mag(tf_sample)
            gt_mag = self._to_mag(gt)
            self.rec_lpips = float(LPIPS(tf_sample_mag, gt_mag))
            self.rec_lpips_avg = self._lpips_all_slices_avg(tf_sample_mag, gt_mag)
            extra_dict["rec_lpips"] = self.rec_lpips
            extra_dict["rec_lpips_avg"] = self.rec_lpips_avg
            extra_dict["fbp_lpips"] = self.fbp_lpips

        if self.log_dists:
            self.rec_dists = float(DISTS(self._to_mag(tf_sample), self._to_mag(gt)))
            extra_dict["rec_dists"] = self.rec_dists
            extra_dict["fbp_dists"] = self.fbp_dists

        tf_sample_mag = self._to_mag(tf_sample)
        gt_mag = self._to_mag(gt)
        rec_nce_metrics = volume_nce_metrics(tf_sample_mag, gt_mag)
        if "rec_nce" in rec_nce_metrics:
            self.rec_nce = rec_nce_metrics["rec_nce"]
        extra_dict.update(rec_nce_metrics)

        if self.log_psnr:
            self.rec_psnr = float(PSNR(tf_sample, gt))
            extra_dict["rec_psnr"] = self.rec_psnr
            extra_dict["fbp_psnr"] = self.fbp_psnr

        if self.log_3d_include_final_slice_averages_for_volumes:
            with torch.no_grad():
                if self.log_ssim:
                    tf_s = self._to_mag(tf_sample)
                    gt_s = self._to_mag(gt)
                    ssim_dim1, _ = SSIM(tf_s, gt_s, axis=0, take_abs_normalize=self.take_abs_normalize, center_slices_only=False)
                    ssim_dim2, _ = SSIM(tf_s, gt_s, axis=1, take_abs_normalize=self.take_abs_normalize, center_slices_only=False)
                    ssim_dim3, _ = SSIM(tf_s, gt_s, axis=2, take_abs_normalize=self.take_abs_normalize, center_slices_only=False)
                    extra_dict.update({"ssim_dim1": ssim_dim1, "ssim_dim2": ssim_dim2, "ssim_dim3": ssim_dim3})

                if self.log_psnr:
                    tf_p = self._to_mag(tf_sample)
                    gt_p = self._to_mag(gt)
                    psnr_dim1, _ = PSNR_2D(tf_p, gt_p, axis=0, use_vol_max=self.log_3d_slice_metrics_use_vol_max, take_abs_normalize=self.take_abs_normalize)
                    psnr_dim2, _ = PSNR_2D(tf_p, gt_p, axis=1, use_vol_max=self.log_3d_slice_metrics_use_vol_max, take_abs_normalize=self.take_abs_normalize)
                    psnr_dim3, _ = PSNR_2D(tf_p, gt_p, axis=2, use_vol_max=self.log_3d_slice_metrics_use_vol_max, take_abs_normalize=self.take_abs_normalize)
                    extra_dict.update({"psnr_dim1": psnr_dim1, "psnr_dim2": psnr_dim2, "psnr_dim3": psnr_dim3})

        wandb.log(
            {
                "rec_loss": float(torch.nn.functional.mse_loss(self.fwd_trafo(sample), self.observation)),
                "rec_mse": float(torch.nn.functional.mse_loss(self.target_trafo(sample), self.ground_truth)),
                "time": self.time,
                "global_step": self.sample_nr,
                "step": self.sample_nr,
                **extra_dict,
            }
        )

    def close_run(self):
        """
        Called once after the reconstruction is finished.
        """
        if not self.final_reco:
            return

        if wandb.run is not None:
            if self.log_psnr and self.rec_psnr is not None:
                wandb.run.summary["fbp_psnr"] = self.fbp_psnr
                wandb.run.summary["rec_psnr"] = self.rec_psnr
            if self.log_ssim and self.rec_ssim is not None:
                wandb.run.summary["fbp_ssim"] = self.fbp_ssim
                wandb.run.summary["rec_ssim"] = self.rec_ssim
                if self.rec_ssim3d is not None:
                    wandb.run.summary["rec_ssim3d"] = self.rec_ssim3d
                if self.rec_ssim3d_fg is not None:
                    wandb.run.summary["rec_ssim3d_fg"] = self.rec_ssim3d_fg
                if self.rec_ssim2d_avg is not None:
                    wandb.run.summary["rec_ssim2d_avg"] = self.rec_ssim2d_avg
            if self.log_vifp and self.rec_vifp is not None:
                wandb.run.summary["fbp_vifp"] = self.fbp_vifp
                wandb.run.summary["rec_vifp"] = self.rec_vifp
            if self.log_lpips and self.rec_lpips is not None:
                wandb.run.summary["fbp_lpips"] = self.fbp_lpips
                wandb.run.summary["rec_lpips"] = self.rec_lpips
                if self.rec_lpips_avg is not None:
                    wandb.run.summary["rec_lpips_avg"] = self.rec_lpips_avg
            if self.log_dists and self.rec_dists is not None:
                wandb.run.summary["fbp_dists"] = self.fbp_dists
                wandb.run.summary["rec_dists"] = self.rec_dists
            if self.rec_nce is not None:
                wandb.run.summary["rec_nce"] = self.rec_nce

    def get_final_stats(self) -> Dict[str, Any]:
        """Return scalar metrics for this single sample."""
        stats: Dict[str, Any] = {}

        if self.time is not None:
            stats["time"] = self.time
        if self.reff_metrics:
            stats.update(self.reff_metrics)
        elif self.reff is not None:
            stats["Reff"] = self.reff
        if self.log_psnr and self.rec_psnr is not None:
            stats["fbp_psnr"] = self.fbp_psnr
            stats["rec_psnr"] = self.rec_psnr
        if self.log_ssim and self.rec_ssim is not None:
            stats["fbp_ssim"] = self.fbp_ssim
            stats["rec_ssim"] = self.rec_ssim
            if self.rec_ssim3d is not None:
                stats["rec_ssim3d"] = self.rec_ssim3d
            if self.rec_ssim3d_fg is not None:
                stats["rec_ssim3d_fg"] = self.rec_ssim3d_fg
            if self.rec_ssim2d_avg is not None:
                stats["rec_ssim2d_avg"] = self.rec_ssim2d_avg
        if self.log_vifp and self.rec_vifp is not None:
            stats["fbp_vifp"] = self.fbp_vifp
            stats["rec_vifp"] = self.rec_vifp
        if self.log_lpips and self.rec_lpips is not None:
            stats["fbp_lpips"] = self.fbp_lpips
            stats["rec_lpips"] = self.rec_lpips
            if self.rec_lpips_avg is not None:
                stats["rec_lpips_avg"] = self.rec_lpips_avg
        if self.log_dists and self.rec_dists is not None:
            stats["fbp_dists"] = self.fbp_dists
            stats["rec_dists"] = self.rec_dists
        if self.rec_nce is not None:
            stats["rec_nce"] = self.rec_nce
        if self._saved_final_sample_files:
            stats["saved_final_sample_npy_files"] = list(self._saved_final_sample_files)

        return stats