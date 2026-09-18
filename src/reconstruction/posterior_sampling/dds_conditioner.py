from torch import Tensor
from typing import Tuple, Any, Dict, Optional
from src.diffmodels.sde import SDE
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
import logging
import functools

from src.reconstruction.posterior_sampling.impl_linear_cg import linear_cg

import numpy as np
from src.diffmodels.sampler.base_conditioning_method import ConditioningMethod

def decomposed_diffusion_sampling_sde_predictor(
    xhat0 : Tensor,
    rhs: Tensor,
    conj_grad_closure: callable,
    gamma: float,
    alpha: float,
    x_var_anchor_weight: float,
    x_var_anchor: Optional[Tensor],
    cg_kwargs: Dict,
) -> Tuple[Tensor, Tensor]:

    xhat_shape = xhat0.shape

    initial_guess = xhat0.reshape(-1, 1)
    rhs_flat = rhs.reshape(-1, 1)

    reg_rhs_flat = rhs_flat * gamma + alpha * initial_guess
    if x_var_anchor is not None and x_var_anchor_weight > 0.0:
        reg_rhs_flat = reg_rhs_flat + x_var_anchor_weight * x_var_anchor.reshape(-1, 1)

    res_before = (rhs_flat - conj_grad_closure(initial_guess)).square().mean()
    xhat, _ = linear_cg(
        matmul_closure=conj_grad_closure,
        rhs=reg_rhs_flat,
        initial_guess=initial_guess,
        **cg_kwargs
    )
    res_after = (rhs_flat - conj_grad_closure(xhat)).square().mean()
    xhat = xhat.T.reshape(xhat_shape)
    # if res_after > res_before + 1e-9:
        # logging.warning(
            # "CG did not converge, res_before: {}, res_after: {}".format(
                # res_before, res_after
            # )
        # )

    return xhat


def _prepare_x_var_anchor(
    x_reference: Tensor,
    x_var_anchor: Optional[Tensor],
    x_var_anchor_weight: float,
    conditioner_name: str,
) -> Optional[Tensor]:
    if x_var_anchor_weight <= 0.0:
        return None
    if x_var_anchor is None:
        raise ValueError(
            f"{conditioner_name} received x_var_anchor_weight={x_var_anchor_weight}, "
            "but x_var_anchor is None."
        )

    x_var_anchor = x_var_anchor.detach().to(
        device=x_reference.device,
        dtype=x_reference.dtype,
    )
    if tuple(x_var_anchor.shape) != tuple(x_reference.shape):
        raise ValueError(
            f"{conditioner_name} x_var_anchor shape {tuple(x_var_anchor.shape)} "
            f"does not match object-space xhat shape {tuple(x_reference.shape)}."
        )
    return x_var_anchor


def conj_grad_closure(
    x: Tensor,
    im_shape: Tuple[int, int],
    fwd_trafo: BaseFwdTrafo,
    gamma: float = 1e-5,
    alpha: float = 1.0,
):
    x = x.T.reshape(1, *im_shape).contiguous()
    return (
        (gamma * fwd_trafo.trafo_adjoint(fwd_trafo(x)) + alpha * x)
        .view(1, np.prod(im_shape))
        .T
    )

class DecomposedDiffusionSampling(ConditioningMethod):

    def __init__(
        self,
        fwd_trafo: BaseFwdTrafo,
        prior_trafo: BasePriorTrafo,
        observation: Tensor,
        sde: SDE,
        im_shape: Tuple[int, int],
        gamma: float = 0.95,
        alpha: float = 1.0,
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
                f"dds requires x_var_anchor_weight >= 0, got {self.x_var_anchor_weight}"
            )
        self.x_var_anchor = x_var_anchor
        self.cg_kwargs = {
            "max_iter": cg_max_iter,
            "max_tridiag_iter": cg_max_tridiag_iter,
        }  # first >! second

        self.conj_grad_closure_partial = functools.partial(
            conj_grad_closure,
            im_shape=im_shape,
            fwd_trafo=self.fwd_trafo,
            gamma=self.gamma,
            alpha=self.alpha + self.x_var_anchor_weight,
        )

    def pre_prediction_step(
        self, x: Tensor, t: Tensor, score_xt: Optional[Tensor], xhat0: Optional[Tensor]
    ) -> Tensor:

        rhs = self.fwd_trafo.trafo_adjoint(self.observation)

        xhat0 = self.prior_trafo.trafo_inv(xhat0)
        x_var_anchor = _prepare_x_var_anchor(
            x_reference=xhat0,
            x_var_anchor=self.x_var_anchor,
            x_var_anchor_weight=self.x_var_anchor_weight,
            conditioner_name="dds",
        )

        xhat = decomposed_diffusion_sampling_sde_predictor(
            rhs=rhs,
            xhat0=xhat0,
            conj_grad_closure=self.conj_grad_closure_partial,
            gamma=self.gamma,
            alpha=self.alpha,
            x_var_anchor_weight=self.x_var_anchor_weight,
            x_var_anchor=x_var_anchor,
            cg_kwargs=self.cg_kwargs
        )

        xhat = self.prior_trafo(xhat)

        return x, xhat

    def post_prediction_step(
        self, x_pre_cond: Tensor, x_pred: Tensor, t: Tensor, score_xt: Tensor
    ) -> Tensor:

        # don't do any update
        return x_pred
