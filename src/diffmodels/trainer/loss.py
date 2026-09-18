import torch

from torch import Tensor
from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels import SDE

from functools import partial


def epsilon_based_loss_fn(x: Tensor, model: UNetModel, sde: SDE):
    """The loss function for training epsilon-based generative models."""

    random_t = torch.randint(1, sde.num_steps, (x.shape[0],), device=x.device)
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, random_t)
    perturbed_x = mean + z * std[:, None, None, None]
    zhat = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((z - zhat).pow(2), dim=(1, 2, 3)))
    
    return loss


def loss_fn_resolver(name : str, **params):
    if name == "epsilon_based_loss_fn":
        return partial(epsilon_based_loss_fn, **params)
    else:
        raise ValueError(f"Unknown loss function name: {name}")