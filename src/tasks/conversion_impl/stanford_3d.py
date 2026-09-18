from __future__ import annotations

from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as etree

import h5py
import numpy as np
from src.tasks.conversion_impl.common import center_crop, ifft2_np, iter_h5_files, rss_np, run_h5_directory, write_fastmri_h5


def et_query(root: etree.Element, qlist: Sequence[str], namespace: str = "http://www.ismrm.org/ISMRMRD") -> str:
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}
    query = "." + "".join(f"//{prefix}:{el}" for el in qlist)
    value = root.find(query, ns)
    if value is None:
        raise RuntimeError(f"Element not found for query {qlist}.")
    return str(value.text)


def _ismrmrd_user_param_value(entry):
    if hasattr(entry, "value"):
        return entry.value
    if hasattr(entry, "value_"):
        return entry.value_
    raise AttributeError(
        f"ISMRMRD user parameter {getattr(entry, 'name', '<unnamed>')!r} "
        f"has no value/value_ attribute; available attributes: {sorted(dir(entry))}"
    )


def ismrmrd_user_param_to_dict(header) -> dict:
    user_dict = {}
    for section in (
        header.userParameters.userParameterLong,
        header.userParameters.userParameterDouble,
        header.userParameters.userParameterString,
        header.userParameters.userParameterBase64,
    ):
        for entry in list(section):
            user_dict[entry.name] = _ismrmrd_user_param_value(entry)
    return user_dict


def load_ismrmrd_to_np(file_name: str | Path, verbose: bool = False):
    import ismrmrd

    dataset = ismrmrd.Dataset(str(file_name), create_if_needed=False)
    xml_header = dataset.read_xml_header()
    header = ismrmrd.xsd.CreateFromDocument(xml_header)
    param_dict = ismrmrd_user_param_to_dict(header)

    num_kx = header.encoding[0].encodedSpace.matrixSize.x
    num_ky = header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum
    num_kz = header.encoding[0].encodingLimits.kspace_encoding_step_2.maximum
    num_channels = header.acquisitionSystemInformation.receiverChannels
    num_slices = header.encoding[0].encodingLimits.slice.maximum + 1
    num_echoes = header.encoding[0].encodingLimits.contrast.maximum + 1
    num_phases = header.encoding[0].encodingLimits.phase.maximum + 1

    chop_y = 1 - int(param_dict.get("ChopY", 1))
    chop_z = 1 - int(param_dict.get("ChopZ", 1))

    try:
        rec_std = dataset.read_array("rec_std", 0)
        rec_weight = 1.0 / (rec_std ** 2)
        rec_weight = np.sqrt(rec_weight / np.sum(rec_weight))
    except Exception:
        rec_weight = np.ones(num_channels)
    opt_mat = np.diag(rec_weight)

    if verbose:
        print(f"Data dims: ({num_kx}, {num_ky}, {num_kz}, {num_channels}, {num_slices}, {num_echoes}, {num_phases})")

    kspace = np.zeros([num_phases, num_echoes, num_slices, num_channels, num_kz, num_ky, num_kx], dtype=np.complex64)
    max_slice = 0
    num_acq = dataset.number_of_acquisitions()
    for idx in range(num_acq):
        acq = dataset.read_acquisition(idx)
        i_ky = acq.idx.kspace_encode_step_1
        i_kz = acq.idx.kspace_encode_step_2
        i_echo = acq.idx.contrast
        i_phase = acq.idx.phase
        i_slice = acq.idx.slice
        max_slice = max(max_slice, i_slice)
        sign = (-1) ** (i_ky * chop_y + i_kz * chop_z)
        data = np.matmul(opt_mat.T, acq.data) * sign
        if i_kz < num_kz and i_ky < num_ky:
            kspace[i_phase, i_echo, i_slice, :, i_kz, i_ky, :] = data
    dataset.close()

    max_slice += 1
    if num_slices != max_slice:
        kspace = kspace[:, :, :max_slice, :, :, :, :]
    return kspace, header, xml_header


def kspace_to_target(kspace: np.ndarray) -> np.ndarray:
    return rss_np(ifft2_np(kspace), axis=-3)


def convert_file(input_file: Path, output_dir: Path) -> int:
    kspace, _header, xml_header = load_ismrmrd_to_np(input_file, verbose=False)
    kspace = kspace[0, 0, :, :, 0, :, :] / 1e7
    kspace = kspace.transpose(0, 1, -1, -2)

    et_root = etree.fromstring(xml_header)
    enc = ["encoding", "encodedSpace", "matrixSize"]
    enc_size = (
        int(et_query(et_root, enc + ["x"])),
        int(et_query(et_root, enc + ["y"])),
        int(et_query(et_root, enc + ["z"])),
    )
    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
    enc_limits_center = int(et_query(et_root, lims + ["center"]))
    enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

    padding_left = enc_size[1] // 2 - enc_limits_center
    padding_right = padding_left + enc_limits_max
    padded = np.zeros((kspace.shape[0], kspace.shape[1], enc_size[0], enc_size[1]), dtype=np.complex64)
    padded[..., padding_left:padding_right] = kspace
    kspace = padded

    rec = ["encoding", "reconSpace", "matrixSize"]
    recon_size = (int(et_query(et_root, rec + ["x"])), int(et_query(et_root, rec + ["y"])))
    target = center_crop(kspace_to_target(kspace), recon_size)

    write_fastmri_h5(
        output_dir / input_file.name,
        kspace,
        target,
        extra_datasets={"ismrmrd_header": xml_header},
    )
    return 1


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    max_files_per_fold: int | None = None,
    volume_limit: int | None = None,
    num_workers: int | None = None,
    cfg=None,
    **_: object,
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_h5_files(input_dir)
    if max_files_per_fold is None and cfg is not None:
        max_files_per_fold = getattr(cfg, "max_files_per_fold", None)
    if volume_limit is None and cfg is not None:
        volume_limit = getattr(cfg, "volume_limit", None)
    if num_workers is None and cfg is not None:
        num_workers = getattr(cfg, "num_workers", 1)
    limit = max_files_per_fold if max_files_per_fold is not None else volume_limit
    if limit is not None and int(limit) >= 0:
        files = files[: int(limit)]
    return run_h5_directory(
        input_dir,
        output_dir,
        convert_file,
        files=files,
        num_workers=int(num_workers or 1),
        desc=f"Converting Stanford 3D {input_dir}",
    )
