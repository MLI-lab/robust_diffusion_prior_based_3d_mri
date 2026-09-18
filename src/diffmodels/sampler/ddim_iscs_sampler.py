# %%
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Any, Dict, Optional, Tuple, List

import torch

from torch import Tensor
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo

from src.diffmodels.sde import SDE
from src.diffmodels.archs.std.unet import UNetModel
from src.reconstruction.posterior_sampling.conditioner_resolver import ConditioningMethod

from .base_sampler import BaseSampler
from .ddim_sampler import DDIM
import numpy as np

import torch

# Copied from https://github.com/duchenhe/ISCS/blob/main/algorithms/utils.py#L169
def slerp_path(
    z0: torch.Tensor,
    z1: torch.Tensor,
    n_mid: int = 30,
    include_endpoints: bool = False,
    eps: float = 1e-8,
):
    """Use SLERP to generate a smooth path between z1 and z2."""
    assert z0.shape == z1.shape, "z0 and z1 must have the same shape."
    # Calculate the angle after flattening
    z0_flat = z0.reshape(-1)
    z1_flat = z1.reshape(-1)
    # Compute the inner product and norm
    dot = torch.dot(z0_flat, z1_flat)
    norm_prod = z0_flat.norm() * z1_flat.norm() + eps
    cos_theta = torch.clamp(dot / norm_prod, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)

    if theta < 1e-6 or torch.abs(theta - np.pi) < 1e-6:
        print("Warning: z0 and z1 are too close or opposite, using linear interpolation.")
        num = n_mid + 2
        alphas = torch.linspace(0, 1, num, device=z0.device, dtype=z0.dtype)
        if not include_endpoints:
            alphas = alphas[1:-1]
        return torch.stack([(1 - a) * z0 + a * z1 for a in alphas]).squeeze(1)

    num = n_mid + 2
    alphas = torch.linspace(0, 1, num, device=z0.device, dtype=z0.dtype)
    if not include_endpoints:
        alphas = alphas[1:-1]

    sin_theta = torch.sin(theta)
    outs = []
    for a in alphas:
        w1 = torch.sin((1 - a) * theta) / (sin_theta + eps)
        w2 = torch.sin(a * theta) / (sin_theta + eps)
        z = w1 * z0 + w2 * z1
        outs.append(z)

    return torch.stack(outs).squeeze(1)

def take_from_center(t: torch.Tensor, n: int, step_left: int = 1, step_right: int = 1, dim: int = 0):
    """Samples `n` slices from dimension `dim` of tensor `t`, following a "center-to-both-sides" sampling pattern.
    """
    L = t.size(dim)
    if n > L:
        raise ValueError("n cannot exceed the length of that dimension.")

    center = L // 2
    indices = [center]

    k = 1
    while len(indices) < n and (center - k * step_left >= 0 or center + k * step_right < L):
        # left
        left = center - k * step_left
        if left >= 0:
            indices.append(left)
            if len(indices) == n:
                break
        # right
        right = center + k * step_right
        if right < L and len(indices) < n:
            indices.append(right)
        k += 1

    idx_sorted = sorted(indices)
    out = t.index_select(dim, torch.tensor(idx_sorted, device=t.device))
    return out, idx_sorted

# %%
def _schedule_jump(num_steps: int, travel_length: int = 1, travel_repeat: int = 1):
    jumps = {}
    for j in range(0, num_steps - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = num_steps
    time_steps = []
    while t >= 1:
        t = t - 1
        time_steps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                time_steps.append(t)
    time_steps.append(-1)

    return time_steps

def _ddim(
    s: Tensor,
    xhat: Tensor,
    ts: Tuple[Tensor, Tensor],
    sde: SDE,
    eta: float,
    noise_sample: Optional[Tensor] = None,
    ) -> Tensor:

    # here we assume that s : Z x C x H x W, i.e. the first dimension is the batch_dimension
    assert s.shape[0] > 1, "s should have a batch dimension for ddim sampling"
    z_0 = torch.randn_like(s[0].unsqueeze(0))
    z_1 = torch.randn_like(s[0].unsqueeze(0))
    noises = slerp_path(z_0, z_1, n_mid=s.shape[0] * 16, include_endpoints=False)
    noises, idx = take_from_center(noises, n=s.shape[0], step_left=16, step_right=16)

    current_time, previous_time = ts
    mean_prev_time = sde.marginal_prob_mean(
            t=previous_time)[:, None, None, None]
    mean_curr_time = sde.marginal_prob_mean(
            t=current_time)[:, None, None, None]
    
    sqrt_beta = ((1 - mean_prev_time.pow(2)) / (1 - mean_curr_time.pow(2))).sqrt() * \
                (1 - mean_curr_time.pow(2) / mean_prev_time.pow(2)).sqrt()
    if sqrt_beta.isnan().any():
        sqrt_beta = torch.zeros_like(sqrt_beta, device= s.device)
    scaled_noise = xhat * mean_prev_time
    deterministic_noise = torch.sqrt(1 - mean_prev_time.pow(2) - sqrt_beta.pow(2) * eta**2) *  s
    stochastic_noise = eta * sqrt_beta * noises

    return scaled_noise + deterministic_noise + stochastic_noise

class DDIM_ISCS(BaseSampler):
    def __init__(self,
            sampling_in_3d : bool,
            im_shape,
            score_mini_batch_size : int,
            cycling_skip_conditioning : bool,
            cycling : bool,
            num_steps : int,
            eta : float,
            score: UNetModel,
            sde: SDE,
            device: Optional[Any] = None,
            conditioning_method : Optional[ConditioningMethod] = None,
            fwd_trafo : Optional[BaseFwdTrafo] = None,
            prior_trafo : Optional[BasePriorTrafo] = None,
            sample_logger : Optional[Any] = None
        ):
        super().__init__(
            score=score,
            sde=sde,
            device=device,
            conditioning_method=conditioning_method,
            fwd_trafo=fwd_trafo,
            prior_trafo=prior_trafo,
            sample_logger=sample_logger,
            sampling_in_3d=sampling_in_3d,
            im_shape=im_shape,
            score_mini_batch_size=score_mini_batch_size,
            cycling_skip_conditioning=cycling_skip_conditioning,
            cycling=cycling,
        )
        self.num_steps = int(num_steps)
        self.eta = float(eta)

    def _init_timeschedule(self, start_timestep: Optional[int] = None) -> List[Tuple[int, int]]:
        if start_timestep is not None:
            timesteps = range(int(start_timestep), -2, -1)
            return list(zip(timesteps[:-1], timesteps[1:]))

        assert self.sde.num_steps >= self.num_steps
        skip = self.sde.num_steps // self.num_steps

        ts = _schedule_jump(self.num_steps)
        time_pairs = list(
            (i * skip , j * skip if j > 0 else -1)
            for i, j in zip(ts[:-1], ts[1:])
        )        
        return time_pairs

    def _predictor(self,
            score_xt: Tensor,
            x: Tensor,
            xhat0 : Optional[Tensor],
            t: Tuple[Tensor, Tensor],
            ) -> Tuple[Tensor, Tensor]:

        with torch.no_grad():
            s = score_xt.detach()
            if xhat0 is None:
                xhat0 = self.sde.tweedy(
                    x = x,
                    t = t[0],
                    score_xt = s)
            x = _ddim(
                s=s,
                xhat=xhat0,
                ts=t,
                sde=self.sde,
                eta=self.eta
                )

        return x.detach(), xhat0.detach()