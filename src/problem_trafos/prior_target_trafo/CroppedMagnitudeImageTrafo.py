from typing import Tuple, List, Optional
from torch import Tensor
import torch
from .base_prior_trafo import BasePriorTrafo
from .base_target_trafo import BaseTargetTrafo
from fastmri.data.transforms import (
    complex_center_crop,
    normalize_instance,
)
import fastmri
import torch.nn.functional as F
import torchvision.transforms as T


class CroppedMagnitudeImagePriorTrafo(BasePriorTrafo, BaseTargetTrafo):
    def __init__(
        self,
        crop_size: Tuple[int, int] = (320, 320),
        center_crop_enabled: bool = False,
        magnitude_enabled: bool = False,
        normalize_enabled: bool = False,
        scaling_factor: float = 1.0,
        swap_channels: bool = False,
        move_axis: Optional[Tuple[int, int]] = None,
        stack_channel_into_batchdim: bool = False,
        stack_z_into_batchdim: bool = False,
        squeeze_channels: Optional[List[int]] = None,
        unsqueeze_channels: Optional[List[int]] = None,
        conv_averaging_shape: Optional[Tuple[int, int, int]] = None,
        swap_spatial_channels: bool = False,
        rotate90: bool = False,
        rotate90_k : int = 1,
        rotate90_dims : Tuple[int, int] = (-2, -1),
        take_first_channel:  bool = False,
        merge_last_two_channels: bool = False,
        collapse_singleton_complex_channel: bool = False,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.center_crop_enabled = center_crop_enabled
        self.magnitude_enabled = magnitude_enabled
        self.normalize_enabled = normalize_enabled
        self.scaling_factor = scaling_factor
        self.swap_channels = swap_channels
        self.move_axis = move_axis
        self.squeeze_channels = squeeze_channels
        self.unsqueeze_channels = unsqueeze_channels

        self.pad_sizes = [0, 160]
        self.pad_trafo = T.Pad(self.pad_sizes)

        self.stack_channel_into_batchdim = stack_channel_into_batchdim
        self.stack_z_into_batchdim = stack_z_into_batchdim
        self.conv_averaging_shape = (
            tuple(conv_averaging_shape) if conv_averaging_shape is not None else None
        )

        self.swap_spatial_channels = swap_spatial_channels
        self.shape_before = None
        self.rotate90 = rotate90
        self.rotate90_k = rotate90_k
        self.rotate90_dims = rotate90_dims
        self.take_first_channel = take_first_channel
        self.merge_last_two_channels = merge_last_two_channels
        self.collapse_singleton_complex_channel = collapse_singleton_complex_channel
        self.shape_before_merge_last_two_channels = None

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the forward projection."""
        if (
            self.collapse_singleton_complex_channel
            and x.ndim == 5
            and x.shape[-2] == 1
            and x.shape[-1] in (1, 2)
        ):
            if x.shape[-1] == 2:
                x = fastmri.complex_abs(x)
            else:
                x = x.squeeze(-1)

        if x.ndim == 5:
            self.shape_before_merge_last_two_channels = x.shape  # (Batch, K, H, W, C)
            if self.center_crop_enabled:
                x = complex_center_crop(x, self.crop_size)
            if self.magnitude_enabled:
                if x.shape[-1] == 2:
                    x = fastmri.complex_abs(x)
                elif x.shape[-1] == 1:
                    x = x.squeeze(-1)
                else:
                    raise ValueError(f"Cannot convert 5D tensor with trailing dimension {x.shape[-1]} to magnitude.")
            if self.scaling_factor != 1.0:
                x = x * self.scaling_factor
            if self.normalize_enabled:
                x, _, _ = normalize_instance(x, eps=1e-11)
            if x.ndim == 4:
                if self.stack_channel_into_batchdim:
                    # (batch, Z, H, W) -> (batch*Z, 1, H, W)
                    self.shape_before = x.shape
                    x = x.reshape(-1, 1, *x.shape[2:])
                elif self.stack_z_into_batchdim:
                    # (batch, Z, H, W) -> (batch*Z, H, W)
                    self.shape_before = x.shape
                    x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
                if self.rotate90:
                    x = torch.rot90(x, k=self.rotate90_k, dims=self.rotate90_dims)
                return x

            # Reshape/transpose x to combine K (slice stack) and complex dimension C
            x = x.permute(0, 1, 4, 2, 3)
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
            if self.rotate90:
                x = torch.rot90(x, k=self.rotate90_k, dims=self.rotate90_dims)
            return x

        if self.center_crop_enabled:
            x = complex_center_crop(x, self.crop_size)
        if self.magnitude_enabled:
            if x.shape[-1] == 2:
                x = fastmri.complex_abs(x)
        if self.scaling_factor != 1.0:
            x = x * self.scaling_factor
        if self.normalize_enabled:
            x, _, _ = normalize_instance(x, eps=1e-11)
        if self.merge_last_two_channels and x.ndim >= 5 and x.shape[-1] == 2:
            self.shape_before_merge_last_two_channels = x.shape
            x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        else:
            self.shape_before_merge_last_two_channels = None
        if self.swap_channels:
            # (1, 256, 320, 320, 2) ->
            x = x.unsqueeze(-4).swapaxes(-4, -1).squeeze(-1)
        if self.move_axis is not None:
            x = x.moveaxis(self.move_axis[0], self.move_axis[1])
        if self.swap_spatial_channels:
            x = x.swapaxes(-2, -1)
        if self.take_first_channel:
            x = x[..., 0]
        if self.squeeze_channels is not None:
            for i in self.squeeze_channels:
                x = x.squeeze(i)
        if self.unsqueeze_channels is not None:
            for i in self.unsqueeze_channels:
                x = x.unsqueeze(i)
        if self.stack_channel_into_batchdim:
            # (batch, Z, ...) -> (batch*Z, 1, ...); for an unbatched
            # volume (Z, H, W), keep depth as the batch-like dimension.
            self.shape_before = x.shape
            if x.ndim == 3:
                x = x.unsqueeze(1)
            else:
                x = x.reshape(-1, 1, *x.shape[2:])
        if self.stack_z_into_batchdim:
            # (batch, Z, C, H, W) -> (batch*Z, C, H, W)  - keeps all dims from index 2 onward
            self.shape_before = x.shape
            x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        if self.conv_averaging_shape is not None:
            # (1, 320, 320, 320) -> (1, target_dim, 320, 320)
            kernel = torch.ones(
                (1, 1) + self.conv_averaging_shape, device=x.get_device()
            ) / torch.prod(torch.Tensor(self.conv_averaging_shape))
            x = F.conv3d(x, weight=kernel, stride=self.conv_averaging_shape)
        if self.rotate90:
            x = torch.rot90(x, k=self.rotate90_k, dims=self.rotate90_dims)
        return x

    def trafo_inv(self, x: Tensor) -> Tensor:
        if (
            self.shape_before_merge_last_two_channels is not None
            and len(self.shape_before_merge_last_two_channels) == 5
        ):
            if self.rotate90:
                x = torch.rot90(x, k=-self.rotate90_k, dims=self.rotate90_dims)
            
            orig_shape = self.shape_before_merge_last_two_channels
            self.shape_before_merge_last_two_channels = None
            
            K = orig_shape[1]
            C = orig_shape[4]
            x = x.reshape(orig_shape[0], K, C, x.shape[-2], x.shape[-1])
            x = x.permute(0, 1, 3, 4, 2)
            
            if self.normalize_enabled:
                raise NotImplementedError
            if self.scaling_factor != 1.0:
                x = x / self.scaling_factor
            if self.magnitude_enabled:
                x = F.pad(x, (0, 1), "constant", 0)
            if self.center_crop_enabled:
                pad_h = (orig_shape[2] - x.shape[2]) // 2
                pad_w = (orig_shape[3] - x.shape[3]) // 2
                x = F.pad(x, (0, 0, pad_w, pad_w, pad_h, pad_h), "constant", 0)
            return x

        if self.rotate90:
            x = torch.rot90(x, k=-self.rotate90_k, dims=self.rotate90_dims)
        if self.conv_averaging_shape is not None:
            for dim, rep in enumerate(self.conv_averaging_shape):
                x = x.repeat_interleave(repeats=rep, dim=dim)
        if self.stack_channel_into_batchdim or self.stack_z_into_batchdim:
            assert self.shape_before is not None
            x = x.reshape(*self.shape_before)
            self.shape_before = None
        if self.squeeze_channels is not None:
            for i in self.squeeze_channels:
                x = x.unsqueeze(i)
        if self.unsqueeze_channels is not None:
            for i in self.unsqueeze_channels:
                x = x.squeeze(i)

        if self.swap_spatial_channels:
            x = x.swapaxes(-2, -1)
        if self.move_axis is not None:
            x = x.moveaxis(self.move_axis[1], self.move_axis[0])
        if (
            self.merge_last_two_channels
            and self.shape_before_merge_last_two_channels is not None
            and x.ndim >= 3
        ):
            merge_last_shape = self.shape_before_merge_last_two_channels[-2:]
            merged_channels = merge_last_shape[0] * merge_last_shape[1]
            if x.shape[-1] == merged_channels:
                x = x.reshape(*x.shape[:-1], *merge_last_shape)
            self.shape_before_merge_last_two_channels = None
        if self.swap_channels:
            x = x.unsqueeze(-1).swapaxes(-4, -1).squeeze(-4)
        if self.normalize_enabled:
            raise NotImplementedError
        if self.scaling_factor != 1.0:
            x = x / self.scaling_factor
        if self.magnitude_enabled:
            x = F.pad(x, (0, 1, 0, 0), "constant", 0)
        if self.center_crop_enabled:
            bcwh = x.unsqueeze(-4).swapaxes(-4, -1).squeeze(-1)
            bcwh_p = self.pad_trafo(bcwh)
            x = bcwh_p.unsqueeze(-1).swapaxes(-1, -4).squeeze(-4)

        return x
