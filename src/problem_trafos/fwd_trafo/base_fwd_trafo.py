"""
Provides :class:`BaseRayTrafo`.
"""

from typing import Dict
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor
import numpy as np


class BaseFwdTrafo(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    def calibrate(self, y: Tensor, calib_params : Dict):
        """
        Used to calibrate the forward map, e.g. to calculate sensitivity maps for MRI.
        """
        pass

    @abstractmethod
    def trafo(self, x: Tensor) -> Tensor:
        """
        Apply the forward projection.
        """
        raise NotImplementedError

    @abstractmethod
    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        """
        """
        raise NotImplementedError

    def fbp(self, observation: Tensor) -> Tensor:
        """Apply a filtered back-projection."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """See :meth:`trafo`."""
        return self.trafo(x)
