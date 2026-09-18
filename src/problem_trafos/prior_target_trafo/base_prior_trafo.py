
from typing import Union, Tuple
from abc import ABC, abstractmethod
from torch import nn
from torch import Tensor

class BasePriorTrafo(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Apply the forward projection."""
        raise NotImplementedError

    @abstractmethod
    def trafo_inv(self, x: Tensor) -> Tensor:
        """Apply the forward projection."""
        raise NotImplementedError


