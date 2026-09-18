from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Union


GOLF_SPARKLING_TRAJECTORY_URL = (
    "https://raw.githubusercontent.com/Shamachrist7/wcrr-noncartesian-3d-mri/main/gs.bin"
)
GOLF_SPARKLING_TRAJECTORY_SHA256 = (
    "c7c90a6b64133c5bc3d6604f28e086b576ddc1708b7922cc9db6778fbc194603"
)
GOLF_SPARKLING_TRAJECTORY_SIZE = 22463056
GOLF_SPARKLING_TRAJECTORY_RELATIVE_PATH = "assets/trajectories/golf_sparkling/gs.bin"


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_trajectory_asset(
    path: Union[str, Path],
    *,
    sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
) -> None:
    path = Path(path)
    if expected_size is not None:
        size = path.stat().st_size
        if size != int(expected_size):
            raise ValueError(
                f"Trajectory asset at {path} has size {size}, expected {expected_size}."
            )
    if sha256 is not None:
        actual = sha256_file(path)
        if actual.lower() != str(sha256).lower():
            raise ValueError(
                f"Trajectory asset at {path} has sha256 {actual}, expected {sha256}."
            )


def ensure_trajectory_asset(
    path: Union[str, Path],
    *,
    source_url: Optional[str] = None,
    sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    timeout_s: float = 60.0,
) -> str:
    """Ensure a trajectory file exists locally and matches provenance checks."""
    path = Path(path)
    if path.exists():
        validate_trajectory_asset(path, sha256=sha256, expected_size=expected_size)
        return str(path)

    if source_url is None:
        raise FileNotFoundError(
            f"Trajectory asset is missing at {path}. Provide the file or set "
            "preprocess.trajectory_source_url so it can be fetched."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with urllib.request.urlopen(str(source_url), timeout=timeout_s) as response:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(response, f)
        validate_trajectory_asset(tmp, sha256=sha256, expected_size=expected_size)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return str(path)
