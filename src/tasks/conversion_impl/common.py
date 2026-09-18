from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable
import multiprocessing as mp

import h5py
import numpy as np
from tqdm.autonotebook import tqdm


def ifft2_np(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def fft2_np(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )


def rss_np(x: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.sum(np.square(np.abs(x)), axis=axis))


def center_crop(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Invalid crop shape {shape}.")
    w_from = max((data.shape[-2] - shape[0]) // 2, 0)
    h_from = max((data.shape[-1] - shape[1]) // 2, 0)
    return data[..., w_from : w_from + shape[0], h_from : h_from + shape[1]]


def iter_h5_files(input_dir: str | Path) -> list[Path]:
    return sorted(Path(input_dir).glob("*.h5"))


def write_fastmri_h5(
    output_file: str | Path,
    kspace: np.ndarray,
    reconstruction_rss: np.ndarray,
    attrs: dict | None = None,
    extra_datasets: dict | None = None,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_name(output_file.stem + f".tmp{output_file.suffix}")
    try:
        with h5py.File(tmp, "w") as hf:
            if extra_datasets:
                for key, value in extra_datasets.items():
                    hf.create_dataset(key, data=value)
            hf.create_dataset("kspace", data=kspace.astype(np.complex64, copy=False))
            hf.create_dataset("reconstruction_rss", data=reconstruction_rss.astype(np.float32, copy=False))
            hf.attrs["max"] = float(np.max(reconstruction_rss))
            for key, value in (attrs or {}).items():
                hf.attrs[key] = value
        tmp.replace(output_file)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _count_produced(produced: int | None) -> int:
    return int(produced if produced is not None else 1)


def run_h5_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    converter: Callable[[Path, Path], int | None],
    files: list[Path] | None = None,
    num_workers: int = 1,
    desc: str | None = None,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = list(files if files is not None else iter_h5_files(input_dir))
    if not selected:
        return 0

    worker_count = max(1, int(num_workers))
    if worker_count > 1 and len(selected) > 1:
        worker_count = min(worker_count, len(selected))
        count = 0
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(converter, input_file, output_dir) for input_file in selected]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                count += _count_produced(fut.result())
        return count

    count = 0
    for input_file in tqdm(selected, total=len(selected), desc=desc):
        count += _count_produced(converter(input_file, output_dir))
    return count
