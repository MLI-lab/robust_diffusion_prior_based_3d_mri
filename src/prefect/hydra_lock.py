"""Serializes Hydra's GlobalHydra usage across concurrent threads."""

import os
import threading
from contextlib import contextmanager
from pathlib import Path

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

hydra_lock = threading.Lock()

_HYDRA_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "hydra"


@contextmanager
def locked_hydra_initialize(*, config_path: str, version_base: str = "1.2"):
    config_dir = config_path if os.path.isabs(config_path) else str(_HYDRA_CONFIG_ROOT / config_path)
    with hydra_lock:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=version_base):
            yield
