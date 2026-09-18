"""Lightweight wrapper that counts diffusion model function evaluations (NFEs)."""
import torch.nn as nn
from torch import Tensor
from typing import Any


class NfeCountingScoreWrapper(nn.Module):
    """Wraps an arbitrary score / diffusion model and records total NFEs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model
        self.nfe: int = 0

    # expose wrapped model parameters / buffers transparently
    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        self.nfe += int(x.shape[0])
        return self._model(x, *args, **kwargs)

    def reset(self) -> None:
        """Reset the counter to zero (call between reconstructions if needed)."""
        self.nfe = 0
