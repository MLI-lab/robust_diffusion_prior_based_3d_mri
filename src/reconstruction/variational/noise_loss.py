from typing import Optional

import torch

from torch import Tensor

from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels.sde import SDE
import math
#import einops
from einops import rearrange

def tv_loss_last_two_dims(output: Tensor) -> Tensor:
    loss = output.new_zeros(())
    if output.size(-2) > 1:
        loss = loss + (output[..., 1:, :] - output[..., :-1, :]).abs().mean()
    if output.size(-1) > 1:
        loss = loss + (output[..., :, 1:] - output[..., :, :-1]).abs().mean()
    return loss

def _linear_descending_timestep(
    steps_scaler: float,
    num_steps: int,
    outer_iteration: int,
    outer_iterations_max: int,
) -> int:
    if outer_iterations_max is None or outer_iterations_max <= 0:
        raise ValueError(
            "outer_iterations_max must be a positive integer for descending time sampling"
        )

    progress_remaining = (outer_iterations_max - outer_iteration) / outer_iterations_max
    progress_remaining = min(max(progress_remaining, 0.0), 1.0)
    return min(max(math.floor(float(steps_scaler) * num_steps * progress_remaining), 0), num_steps - 1)

def noise_loss(
    output: Tensor,
    outer_iteration : int,
    outer_iterations_max : int,
    score: UNetModel,
    sde: SDE,
    repetition : int = 1,
    reg_strength: float = 1.,
    steps_scaler : float = 0.5,
    time_sampling_method : str = 'random',
    adapt_reg_strength: Optional[bool] = None,
    adapt_reg_strength_p: Optional[float] = None,
    subsampling_factor: Optional[float] = None, # for DMs which allow subsampling
    reg_strength_tv: Optional[float] = None, # for optionally adding a TV regularization term
    compensate_passthrough_alpha_t : bool = False,
    signal_domain_weighting: bool = False,
    ) -> Tensor:

    output = output.repeat(repetition, *[1]*(output.ndim -1))

    if time_sampling_method == 'random':
        t = torch.randint(1, 
            math.floor(steps_scaler * sde.num_steps),
            (output.shape[0],),
            device=output.device
        ) # random time-sampling (allows for batching and single time step reg.)
    elif time_sampling_method == 'linear_descending':
        t = torch.tensor(
            _linear_descending_timestep(
                steps_scaler=steps_scaler,
                num_steps=sde.num_steps,
                outer_iteration=outer_iteration,
                outer_iterations_max=outer_iterations_max,
            ),
            device=output.device
        ).repeat(output.shape[0])
    elif time_sampling_method in ['random_linear_descending', 'random_descending']:
        max_t = _linear_descending_timestep(
            steps_scaler=steps_scaler,
            num_steps=sde.num_steps,
            outer_iteration=outer_iteration,
            outer_iterations_max=outer_iterations_max,
        )
        if max_t <= 1:
            t = torch.ones(output.shape[0], dtype=torch.long, device=output.device)
        else:
            t = torch.randint(
                1,
                max_t + 1,
                (output.shape[0],),
                device=output.device
            )
    else:
        raise NotImplementedError(f'time_sampling {time_sampling_method} not implemented')
    
    z = torch.randn_like(output)
    mean, std = sde.marginal_prob(output, t)
    perturbed_x = mean + z * std[:, None, None, None]

    # subsampling
    if subsampling_factor is not None:
        batch_size, channels, height, width = output.shape
        import numpy as np
        size = math.ceil(height * width * subsampling_factor)
        sample_lst = torch.stack(
            [torch.from_numpy(
                    np.random.choice(height*width, size, replace=False)
                ) for _ in range(batch_size)
            ]).to(z.device) # sample lst has shape (B, size), e.g. 50 25600 -> (B, size, 1) -> (B, size, C)

        z = rearrange(z, 'b c h w -> b (h w) c') # x has shape (B, 256*320, 2)
        z = torch.gather(z, dim=1, index=sample_lst.unsqueeze(2).repeat(1,1,channels)).contiguous()

        perturbed_x = rearrange(perturbed_x, 'b c h w -> b (h w) c')
        perturbed_x = torch.gather(perturbed_x, dim=1, index=sample_lst.unsqueeze(2).repeat(1,1,channels)).contiguous()

        zhat = score(perturbed_x, (t, sample_lst, height, width))
    else:
        zhat = score(perturbed_x, t)

    if perturbed_x.size(1) == 1 and zhat.size(1) == 2:
        # this occurs when learn_sigma is enabled for the trained network
        zhat = zhat[:, :1]

    residual = (z - zhat).pow(2)
    per_sample_loss = residual.flatten(1).mean(dim=1)

    reg_strength_t = output.new_full((output.shape[0],), float(reg_strength))
    if adapt_reg_strength or signal_domain_weighting:
        alpha = sde.marginal_prob_mean(t).clamp_min(1e-8)
        sigma = std.clamp_min(1e-8)
        if signal_domain_weighting:
            # Tweedie gives x0_hat - x0 = (sigma / alpha) * (eps - eps_hat).
            # Thus signal-domain MSE corresponds to weighting epsilon MSE by (sigma / alpha)^2.
            reg_strength_t = (sigma / alpha).pow(2) * reg_strength
        else:
            exp = adapt_reg_strength_p if adapt_reg_strength_p is not None else 0.5
            reg_strength_t = (sigma / alpha).pow(exp) * reg_strength

        if compensate_passthrough_alpha_t:
            # Compensate the dx_t/dx = alpha passthrough in the gradient path.
            reg_strength_t = reg_strength_t / alpha

    loss = (reg_strength_t * per_sample_loss).mean()

    if reg_strength_tv:
        loss = loss + reg_strength_tv * tv_loss_last_two_dims(output)

    return loss
