from torch import Tensor
from typing import Tuple, Any, Dict, Optional
from src.diffmodels.sde import SDE
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
import logging
import functools
import torch

from src.reconstruction.posterior_sampling.impl_linear_cg import linear_cg

import numpy as np
from src.diffmodels.sampler.base_conditioning_method import ConditioningMethod
from src.reconstruction.posterior_sampling.dds_conditioner import _prepare_x_var_anchor

def _Dz(x):
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    y[:, -1] = x[:, 0]
    return y - x


def _DzT(x):
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    y[:, -1] = x[:, 0]

    tempt = -(y-x)
    difft = tempt[:, :-1]
    y[:, 1:] = difft
    y[:, 0] = x[:, -1] - x[:, 0]

    return y

def conj_grad_closure_tv(
    x: Tensor,
    im_shape: Tuple[int, int],
    fwd_trafo: BaseFwdTrafo,
    gamma: float = 1e-5,
    alpha: float = 1.0,
    rho: float = 0.5,
):
    x_img = x.T.reshape(1, *im_shape).contiguous()
    y_img = gamma * fwd_trafo.trafo_adjoint(fwd_trafo(x_img)) + alpha * x_img + rho * _DzT(_Dz(x_img))
    return y_img.view(1, int(np.prod(im_shape))).T

def shrink(src, lamb):
    return torch.sign(src) * torch.max(torch.abs(src)-lamb, torch.zeros_like(src))

class DecomposedDiffusionSamplingWithTV(ConditioningMethod):

    def __init__(
        self,
        fwd_trafo: BaseFwdTrafo,
        prior_trafo: BasePriorTrafo,
        observation: Tensor,
        sde: SDE,
        im_shape: Tuple[int, int],
        gamma: float = 0.95,
        rho: float = 0.5,
        lamb: float = 0.1,
        alpha: float = 1.0,
        admm_iters: int = 1,
        cg_max_iter: int = 2,
        cg_max_tridiag_iter: int = 2,
        x_var_anchor_weight: float = 0.0,
        x_var_anchor: Optional[Tensor] = None,
    ):
        super().__init__(fwd_trafo, observation, sde)
        self.prior_trafo = prior_trafo
        self.gamma = gamma
        self.alpha = alpha
        self.x_var_anchor_weight = float(x_var_anchor_weight)
        if self.x_var_anchor_weight < 0.0:
            raise ValueError(
                f"dds_tv requires x_var_anchor_weight >= 0, got {self.x_var_anchor_weight}"
            )
        self.x_var_anchor = x_var_anchor
        self.rho = rho
        if self.rho <= 0:
            raise ValueError(f"dds_tv requires rho > 0, got rho={self.rho}")
        # Keep CG system strictly positive definite for numerical stability.
        self.alpha_cg = float(self.alpha) if float(self.alpha) > 0.0 else 1e-6
        if float(self.alpha) <= 0.0:
            logging.warning(
                "dds_tv received alpha=%s; using alpha_cg=%s for CG stability.",
                self.alpha,
                self.alpha_cg,
            )
        self.lamb = lamb
        self.admm_iters = admm_iters
        self.cg_kwargs = {
            "max_iter": cg_max_iter,
            "max_tridiag_iter": cg_max_tridiag_iter,
        }  # first >! second
        self.im_shape = im_shape

        self.conj_grad_closure_partial = functools.partial(
            conj_grad_closure_tv,
            im_shape=im_shape,
            fwd_trafo=self.fwd_trafo,
            gamma=self.gamma,
            rho = self.rho,
            alpha=self.alpha_cg + self.x_var_anchor_weight,
        )

        self.z_t = None
        self.w_t = None

    def init_sampling(self, x : Tensor) -> None:
        """
            Called once at the beginning of the sampling process.
        """
        self.w_t = torch.zeros( (1, *self.im_shape), device=x.device)
        self.z_t = torch.zeros( (1, *self.im_shape), device=x.device)

    def pre_prediction_step(
        self, x: Tensor, t: Tensor, score_xt: Optional[Tensor], xhat0: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:

        assert self.w_t is not None and self.z_t is not None, "w_t and z_t must be initialized before calling pre_prediction_step"
        assert xhat0 is not None, "xhat0 must not be None for dds_tv conditioning"

        xhat = self.prior_trafo.trafo_inv(xhat0)
        x_var_anchor = _prepare_x_var_anchor(
            x_reference=xhat,
            x_var_anchor=self.x_var_anchor,
            x_var_anchor_weight=self.x_var_anchor_weight,
            conditioner_name="dds_tv",
        )
        
        for _ in range(self.admm_iters):
            # ADMM x-update solves:
            # (gamma A^T A + alpha I + beta I + rho D^T D)x
            # = gamma A^T y + alpha x0 + beta x_var + rho D^T(z-w)
            rhs = (
                self.gamma * self.fwd_trafo.trafo_adjoint(self.observation)
                + self.alpha_cg * xhat
                + self.rho * (_DzT(self.z_t) - _DzT(self.w_t))
            )
            if x_var_anchor is not None and self.x_var_anchor_weight > 0.0:
                rhs = rhs + self.x_var_anchor_weight * x_var_anchor

            xhat_shape = xhat.shape
            initial_guess = xhat.reshape(-1, 1)
            rhs_flat = rhs.reshape(-1, 1)
            xhat_flat, _ = linear_cg(
                matmul_closure=self.conj_grad_closure_partial,
                rhs=rhs_flat,
                initial_guess=initial_guess,
                **self.cg_kwargs,
            )
            xhat = xhat_flat.T.reshape(xhat_shape)

            # if not torch.isfinite(xhat).all():
                # raise RuntimeError("dds_tv produced non-finite xhat during ADMM-CG update")

            self.z_t = shrink(_Dz(xhat) + self.w_t, self.lamb / self.rho)
            self.w_t = _Dz(xhat) + self.w_t - self.z_t

        xhat = self.prior_trafo(xhat)

        return x, xhat

    def post_prediction_step(
        self, x_pre_cond: Tensor, x_pred: Tensor, t: Tensor, score_xt: Tensor
    ) -> Tensor:

        # don't do any update
        return x_pred
