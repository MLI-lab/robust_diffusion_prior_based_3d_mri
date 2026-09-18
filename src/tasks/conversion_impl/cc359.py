from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from src.tasks.conversion_impl.common import fft2_np, ifft2_np, iter_h5_files, rss_np, run_h5_directory, write_fastmri_h5


def convert_file(input_file: Path, output_dir: Path) -> int:
    with h5py.File(input_file, "r") as hf:
        kspace_hf = hf["kspace"][()]
    kspace_cpx = kspace_hf[..., ::2] + 1j * kspace_hf[..., 1::2]
    kspace_cpx = kspace_cpx.transpose(0, -1, 1, 2)
    kspace_cpx *= 1e-10

    img = ifft2_np(kspace_cpx)
    img_shifted = np.fft.ifftshift(img, axes=(-2, -1))
    # # ?
    # img_shifted = np.flip(img_shifted, axis=-1)
    kspace = fft2_np(img_shifted)
    target = rss_np(img_shifted, axis=1)

    write_fastmri_h5(output_dir / input_file.name, kspace, target)
    return 1


def run(input_dir: str | Path, output_dir: str | Path, cfg=None, **_: object) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_h5_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"CC359 conversion found no .h5 files in {input_dir}. "
            "Check conversion.<fold>.input_relpath against the extracted dataset layout."
        )

    max_files = getattr(cfg, "max_files_per_fold", None) if cfg is not None else None
    if max_files is None and cfg is not None:
        max_files = getattr(cfg, "max_files", None)
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]

    num_workers = getattr(cfg, "num_workers", 1) if cfg is not None else 1
    return run_h5_directory(
        input_dir,
        output_dir,
        convert_file,
        files=files,
        num_workers=int(num_workers),
        desc=f"Converting CC359 {input_dir}",
    )
