# %%
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Any, Dict, Optional, Tuple, List

import torch

from torch import Tensor
from tqdm import tqdm
import numpy as np
from src.problem_trafos.prior_target_trafo.base_prior_trafo import BasePriorTrafo
from src.problem_trafos.fwd_trafo.base_fwd_trafo import BaseFwdTrafo

from src.diffmodels.sde import SDE
from src.diffmodels.archs.std.unet import UNetModel
from src.diffmodels.sampler.base_conditioning_method import ConditioningMethod

from src.sample_logger.base_sample_logger import BaseSampleLogger

class BaseSampler(ABC):

    def __init__(self,
            im_shape,
            sampling_in_3d : bool,
            score_mini_batch_size : int,
            cycling_skip_conditioning : bool,
            cycling : bool,
            score: UNetModel,
            sde: SDE,
            device: Optional[str] = None,
            prior_trafo : BasePriorTrafo = None,
            conditioning_method : Optional[ConditioningMethod] = None,
            fwd_trafo : Optional[BaseFwdTrafo] = None,
            sample_logger : Optional[BaseSampleLogger] = None,
        ):
        super().__init__()

        self.score = score
        self.sde = sde
        self.device = device
        self.condition_method = conditioning_method
        self.sample_logger = sample_logger
        self.fwd_trafo = fwd_trafo
        self.prior_trafo = prior_trafo

        self.sampling_in_3d = sampling_in_3d
        self.im_shape = im_shape
        self.score_mini_batch_size = score_mini_batch_size
        self.cycling_skip_conditioning = cycling_skip_conditioning
        self.cycling = cycling

    @abstractmethod
    def _init_timeschedule(self, start_timestep: Optional[int] = None) -> List[Tuple[int, int]]:
        """Called once at the beginning of the sampling process."""
        pass

    @abstractmethod
    def _predictor(self,
            score_xt: Tensor,
            x: Tensor,
            t: Tuple[Tensor, Tensor],
            xhat0 : Optional[Tensor],
        ) -> Tuple[Tensor, Tensor]:
        """Called at each time step to predict the next state in the (reverse) MC.
        Given the previous state x, the current time, and the score at time xt.
        """
        pass

    def _validate_warm_start(
        self,
        init_x: Optional[Tensor],
        start_timestep: Optional[int],
    ) -> Optional[int]:
        if (init_x is None) != (start_timestep is None):
            raise ValueError("init_x and start_timestep must either both be provided or both be None.")
        if start_timestep is None:
            return None

        start_timestep = int(start_timestep)
        if start_timestep < 0 or start_timestep >= int(self.sde.num_steps):
            raise ValueError(
                f"start_timestep must satisfy 0 <= start_timestep < {self.sde.num_steps}, "
                f"got {start_timestep}."
            )
        if tuple(init_x.shape) != tuple(self.im_shape):
            raise ValueError(f"init_x shape {tuple(init_x.shape)} does not match im_shape {tuple(self.im_shape)}.")
        return start_timestep

    def sample(
        self,
        init_x: Optional[Tensor] = None,
        start_timestep: Optional[int] = None,
    ) -> Tensor:

        start_timestep = self._validate_warm_start(init_x, start_timestep)
        self.time_pairs = self._init_timeschedule(start_timestep=start_timestep)

        if init_x is None:
            init_x = self.sde.prior_sampling(
                self.im_shape
            ).to(self.device)
        else:
            init_x = init_x.detach().to(self.device)

        ones_vec = torch.ones(
            (1),
            device=self.device
        )

        x = init_x
        i = 0
        pbar = tqdm(self.time_pairs)

        step_cntr = 0
        mini_batch_size = self.score_mini_batch_size

        if self.condition_method is not None:
            self.condition_method.init_sampling(x)

        for step in pbar:

            t = (ones_vec * step[0], ones_vec * step[1])

            # cycling application
            skip_conditioning = False
            if self.cycling and self.sampling_in_3d:
                if step_cntr % 3 in (1, 2):
                    skip_conditioning = self.cycling_skip_conditioning

            needs_score_grad = (
                self.condition_method is not None
                and not skip_conditioning
                and getattr(self.condition_method, "requires_score_grad", False)
            )
            if needs_score_grad:
                x = x.detach().requires_grad_(True)

            if self.cycling and self.sampling_in_3d:
                if step_cntr % 3 == 0:
                    x_into_score = x
                elif step_cntr % 3 == 1:
                    x_into_score = x.swapaxes(0, 2)
                elif step_cntr % 3 == 2:
                    x_into_score = x.swapaxes(0, 3)
            else:
                x_into_score = x

            score_grad_context = torch.enable_grad() if needs_score_grad else torch.no_grad()
            with score_grad_context:
                if self.sampling_in_3d:
                    score_xt = torch.vstack(
                        [self.score(x_into_score[j:j+mini_batch_size], t[0]) for j in range(0, x_into_score.shape[0], mini_batch_size)]
                    )
                else:
                    score_xt = self.score(x_into_score, t[0])

            if x.size(1) == 1 and score_xt.size(1) == 2:
                # this occurs when learn_sigma is enabled for the trained network
                score_xt = score_xt[:, :1]

            if self.cycling and self.sampling_in_3d:
                if step_cntr % 3 == 0:
                    score_xt = score_xt
                elif step_cntr % 3 == 1:
                    score_xt = score_xt.swapaxes(0, 2)
                elif step_cntr % 3 == 2:
                    score_xt = score_xt.swapaxes(0, 3) 

            step_cntr += 1

            # calc tweedy update
            xhat0 = self.sde.tweedy(x=x, t=t[0], score_xt=score_xt)

            # conditioning
            if self.condition_method is not None and not skip_conditioning:
                x, xhat_updated = self.condition_method.pre_prediction_step(
                    x=x, t=t[0], xhat0=xhat0, score_xt=score_xt
                )
            else:
                xhat_updated = xhat0

            # predicted x
            x_new, x_mean = self._predictor(
                score_xt=score_xt,
                x=x,
                t=t,
                xhat0=xhat_updated
            )

            # post_predictions tep for conditioning
            if self.condition_method is not None and not skip_conditioning:
                x_new = self.condition_method.post_prediction_step(
                    x_pre_cond=x, x_pred=x_new, t=t[0], score_xt=score_xt
                )

            # update
            x = x_new

            # logging
            if self.sample_logger is not None:
                from src.representations.fixed_grid_representation import FixedGridRepresentation
                x_mean_pi = self.prior_trafo.trafo_inv(x_mean)
                representation = FixedGridRepresentation(in_shape=x_mean_pi.shape[:-1], out_features=x_mean_pi.shape[-1], warm_start=x_mean_pi)
                self.sample_logger(representation=representation,
                    step=i, pbar=pbar)

            i += 1

        return self.prior_trafo.trafo_inv(x_mean)
