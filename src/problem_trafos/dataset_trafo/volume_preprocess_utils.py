"""Shared volume preprocessing utilities."""
from __future__ import annotations

import math

import numpy as np
import torch

from src.utils.fftn3d import fft3c, ifft3c


# Core interpolation helpers

def fourier_crop_3d(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Downsample a real-valued representation of a 3D complex tensor by cropping the centre of k-space (Fourier space).
    """
    kspace = fft3c(x)                              # (..., Z, Y, X, 2)
    *_, Z, Y, X, _ = kspace.shape
    nZ = int(round(Z * scale_factor))
    nY = int(round(Y * scale_factor))
    nX = int(round(X * scale_factor))
    sZ = (Z - nZ) // 2
    sY = (Y - nY) // 2
    sX = (X - nX) // 2
    kspace_cropped = kspace[..., sZ:sZ + nZ, sY:sY + nY, sX:sX + nX, :]
    return ifft3c(kspace_cropped).contiguous()


def spatial_interpolate_3d(
    x: torch.Tensor,
    scale_factor: float,
    mode: str,
) -> torch.Tensor:
    """Interpolate a 3D volume using ``torch.nn.functional.interpolate``."""
    # interpolate expects (N, C, D, H, W); move real/imag to channel dim
    return (
        torch.nn.functional.interpolate(
            x.movedim(-1, 0).unsqueeze(0),   # (1, 2, ..., Z, Y, X)
            scale_factor=scale_factor,
            mode=mode,
        )
        .squeeze(0)
        .movedim(0, -1)
        .contiguous()
    )


def interpolate_volume(
    x: torch.Tensor,
    scale_factor: float,
    method: str,
) -> torch.Tensor:
    """Dispatch to fourier_crop_3d or spatial_interpolate_3d."""
    if scale_factor == 1.0:
        return x
    if method == "fourier":
        return fourier_crop_3d(x, scale_factor)
    return spatial_interpolate_3d(x, scale_factor, method)


# Sensitivity-map interpolation

def interpolate_sensmaps(
    sens_maps_np: np.ndarray,
    scale_factor: float,
    method: str,
    device: str = "cpu",
) -> np.ndarray:
    """Interpolate a complex sensitivity-map array."""
    if scale_factor == 1.0:
        return sens_maps_np

    # To tensor: (Coils, Z, Y, X, 2)
    S_complex = sens_maps_np if np.iscomplexobj(sens_maps_np) else (
        sens_maps_np[..., 0] + 1j * sens_maps_np[..., 1]
    )
    S_torch = torch.view_as_real(
        torch.from_numpy(S_complex.astype(np.complex64)).movedim(-1, 0)
    ).to(device)                                             # (Coils, Z, Y, X, 2)

    if method == "fourier":
        S_interp = fourier_crop_3d(S_torch, scale_factor)   # (Coils, Z', Y', X', 2)
    else:
        # interpolate per-coil: (Coils, Z, Y, X, 2) -> need (Coils, 2, Z, Y, X)
        S_reform = S_torch.permute(0, 4, 1, 2, 3)           # (Coils, 2, Z, Y, X)
        S_interp = (
            torch.nn.functional.interpolate(
                S_reform,
                scale_factor=scale_factor,
                mode=method,
            )
            .permute(0, 2, 3, 4, 1)                         # (Coils, Z', Y', X', 2)
            .contiguous()
        )

    S_interp_complex = torch.view_as_complex(S_interp)      # (Coils, Z', Y', X')
    # -> (Z', Y', X', Coils)
    return S_interp_complex.movedim(0, -1).cpu().numpy().astype(np.complex64)


# Scaling helpers

def scale_by_kspace_norm(
    target: torch.Tensor,
    kspace_vol_norm: float,
) -> torch.Tensor:
    """Apply the k-space-norm-based global magnitude calibration:
    target *= sqrt(prod(shape)) / kspace_vol_norm
    """
    return target * math.sqrt(float(np.prod(target.shape).item())) / kspace_vol_norm


# Kspace shift (dataset-specific centering)
