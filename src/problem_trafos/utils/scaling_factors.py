import torch

from src.problem_trafos import fwd_trafo
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
import math
import numpy as np
from tqdm import tqdm

def complex_abs(x):
    # x: (..., 2), last dim is [real, imag]
    return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)

def complex_inner_real(x, y):
    # real part of <x, y> = sum conj(x) * y
    return (x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1]).sum()

@torch.no_grad()
def cg_proxy_scale(
    fwd_trafo : BaseFwdTrafo,
    y : torch.Tensor,
    image_shape,
    show_tqdm : bool = False,
    max_iter=10,
    rtol=1e-2,
    lam=1e-6,
    eps=1e-12,
):
    """y and x use real complex format: (..., 2)
    Returns scaling_factor such that scaling_factor * y gives
    a CG proxy reconstruction with approx unit std.
    """
    
    A = lambda x: fwd_trafo.trafo(x)
    AH = lambda k: fwd_trafo.trafo_adjoint(k)
    normal_op = lambda x: AH(A(x)) + lam * x

    b = AH(y)

    x = torch.zeros(image_shape, device=y.device, dtype=y.dtype)

    r = b - normal_op(x)
    p = r.clone()

    b_norm = torch.sqrt(complex_inner_real(b, b)).clamp_min(eps)
    rsold = complex_inner_real(r, r).clamp_min(eps)

    bar = range(max_iter) if not show_tqdm else tqdm(range(max_iter), desc="CG for scaling factor", leave=False)

    for _ in bar:
        Ap = normal_op(p)
        alpha = rsold / complex_inner_real(p, Ap).clamp_min(eps)

        x = x + alpha * p
        r = r - alpha * Ap

        if torch.sqrt(complex_inner_real(r, r)) / b_norm < rtol:
            break

        rsnew = complex_inner_real(r, r).clamp_min(eps)
        p = r + (rsnew / rsold) * p
        rsold = rsnew

        if show_tqdm:
            bar.set_postfix({"residual": torch.sqrt(complex_inner_real(r, r)).item() / b_norm.item()})

    # std = complex_abs(x).std().clamp_min(eps)
    std = x.std().clamp_min(eps)
    scaling_factor = 1.0 / float(std)

    return scaling_factor, x

def get_scaling_factor(
    mode : str,
    fwd_trafo : BaseFwdTrafo,
    observation : torch.Tensor,
    rep_shape,
    **kwargs
):
    if mode == "default":
        return math.sqrt(float(np.prod(rep_shape).item())) / observation.detach().cpu().norm()
    elif mode == "iterapprox_pseudoinv":
        return cg_proxy_scale(
            fwd_trafo=fwd_trafo,
            y=observation,
            image_shape=rep_shape,
            **kwargs
        )[0]
    else:
        raise ValueError(f"Unknown scaling factor mode: {mode}")