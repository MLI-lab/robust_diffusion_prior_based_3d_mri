from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm.autonotebook import tqdm

from src.tasks.conversion_impl.common import fft2_np, iter_h5_files, rss_np, write_fastmri_h5


def ifft3_np(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(x, axes=(-3, -2, -1)),
            axes=(-3, -2, -1),
            norm="ortho",
        ),
        axes=(-3, -2, -1),
    )


def _read_ismrmrd_volume(input_file: Path) -> tuple[np.ndarray, str]:
    try:
        import ismrmrd
    except ImportError as exc:
        raise RuntimeError("AHEAD conversion requires the ismrmrd package.") from exc

    dataset = ismrmrd.Dataset(str(input_file), "dataset", create_if_needed=False)
    xml_header = dataset.read_xml_header()
    header = ismrmrd.xsd.CreateFromDocument(xml_header)

    num_x = header.encoding[0].encodedSpace.matrixSize.x
    num_y = header.encoding[0].encodedSpace.matrixSize.y
    num_z = header.encoding[0].encodedSpace.matrixSize.z
    num_coils = header.acquisitionSystemInformation.receiverChannels

    kspace = np.zeros((num_coils, num_x, num_y, num_z), dtype=np.complex64)
    try:
        for acq_idx in range(dataset.number_of_acquisitions()):
            acq = dataset.read_acquisition(acq_idx)
            ky = acq.idx.kspace_encode_step_1
            kz = acq.idx.kspace_encode_step_2
            if ky < num_y and kz < num_z:
                kspace[:, :, ky, kz] = acq.data
    finally:
        dataset.close()

    return kspace, xml_header


def _project_perspective(volume: np.ndarray, perspective: str) -> np.ndarray:
    if perspective == "sag":
        return volume.transpose(1, 0, 3, 2)
    if perspective == "cor":
        return volume.transpose(2, 0, 3, 1)
    if perspective == "ax":
        return volume.transpose(3, 0, 2, 1)
    raise ValueError(f"Unknown AHEAD perspective {perspective!r}; expected one of sag, cor, ax.")


def convert_file(input_file: Path, output_dir: Path, perspective: str, scale: float) -> int:
    kspace_3d, xml_header = _read_ismrmrd_volume(input_file)
    volume = ifft3_np(kspace_3d / scale)

    target_complex = _project_perspective(volume, perspective)
    kspace = fft2_np(target_complex).astype(np.complex64, copy=False)
    target = rss_np(target_complex, axis=1).astype(np.float32, copy=False)

    write_fastmri_h5(
        output_dir / input_file.name,
        kspace,
        target,
        attrs={"perspective": perspective},
        extra_datasets={"ismrmrd_header": xml_header},
    )
    return 1


def run(input_dir: str | Path, output_dir: str | Path, cfg: Any | None = None, **_: object) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perspective = str(getattr(cfg, "perspective", "sag"))
    scale = float(getattr(cfg, "scale", 1e14))
    count = 0
    files = iter_h5_files(input_dir)
    for input_file in tqdm(files, total=len(files), desc=f"Converting AHEAD {input_dir} [{perspective}]"):
        count += convert_file(input_file, output_dir, perspective=perspective, scale=scale)
    return count
