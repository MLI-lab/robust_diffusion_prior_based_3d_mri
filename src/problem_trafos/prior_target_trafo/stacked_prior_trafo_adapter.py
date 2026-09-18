import torch
from torch import Tensor


class StackedPriorTrafoAdapter:
    def __init__(
        self,
        base_prior_trafo,
        stack_num_slices: int,
        base_channels_per_slice: int,
        stack_padding_mode: str = "edge",
    ) -> None:
        self.base_prior_trafo = base_prior_trafo
        self.stack_num_slices = int(stack_num_slices)
        self.base_channels_per_slice = int(base_channels_per_slice)
        self.stack_padding_mode = stack_padding_mode

        if self.stack_num_slices < 1:
            raise ValueError(
                f"stack_num_slices must be a positive integer, got {self.stack_num_slices}."
            )
        if self.base_channels_per_slice < 1:
            raise ValueError(
                f"base_channels_per_slice must be >= 1, got {self.base_channels_per_slice}."
            )
        if self.stack_padding_mode not in ("edge",):
            raise ValueError(
                f"Unsupported stack_padding_mode '{self.stack_padding_mode}'. Currently supported: ['edge']."
            )

    @property
    def expected_stacked_channels(self) -> int:
        return self.base_channels_per_slice * self.stack_num_slices

    def _stack_along_depth(self, y: Tensor) -> Tensor:
        if y.ndim != 4:
            return y

        if y.shape[1] != self.base_channels_per_slice:
            return y

        if self.stack_num_slices == 1:
            return y

        depth = y.shape[0]
        half = self.stack_num_slices // 2
        if self.stack_num_slices % 2 == 0:
            offsets = torch.arange(-half + 1, half + 1, device=y.device)
        else:
            offsets = torch.arange(-half, half + 1, device=y.device)
        indices = torch.arange(depth, device=y.device).unsqueeze(1) + offsets.unsqueeze(0)
        if self.stack_padding_mode == "edge":
            indices = indices.clamp(0, depth - 1)

        y_stacked = y.index_select(0, indices.reshape(-1)).reshape(
            depth,
            self.stack_num_slices,
            self.base_channels_per_slice,
            y.shape[-2],
            y.shape[-1],
        )
        y_stacked = y_stacked.reshape(
            depth,
            self.expected_stacked_channels,
            y.shape[-2],
            y.shape[-1],
        )
        return y_stacked

    def __call__(self, x: Tensor) -> Tensor:
        y = self.base_prior_trafo(x)
        return self._stack_along_depth(y)

    def trafo_inv(self, y: Tensor) -> Tensor:
        if (
            self.stack_num_slices > 1
            and y.ndim == 4
            and y.shape[1] == self.expected_stacked_channels
        ):
            center = self.stack_num_slices // 2
            start = center * self.base_channels_per_slice
            end = (center + 1) * self.base_channels_per_slice
            y = y[:, start:end]
        return self.base_prior_trafo.trafo_inv(y)

    def __getattr__(self, name):
        return getattr(self.base_prior_trafo, name)
