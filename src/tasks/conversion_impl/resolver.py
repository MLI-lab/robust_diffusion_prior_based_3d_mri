from __future__ import annotations

from importlib import import_module
from typing import Callable


_ALIASES = {
    "cc359": "cc359",
    "cc359_fastmri_h5": "cc359",
    "stanford_3d": "stanford_3d",
    "stanford3d": "stanford_3d",
    "stanford3d_fastmri_h5": "stanford_3d",
    "luesebrink": "luesebrink",
    "luesebrink_nifti": "luesebrink",
    "luesebrink_nifti_h5": "luesebrink",
    "ahead": "ahead",
    "ahead_fastmri_h5": "ahead",
}


def get_conversion_impl(task_name: str) -> Callable:
    key = _ALIASES.get(str(task_name), str(task_name))
    module = import_module(f"src.tasks.conversion_impl.{key}")
    if not hasattr(module, "run"):
        raise AttributeError(f"Conversion implementation {key!r} does not expose run(input_dir, output_dir, **kwargs).")
    return module.run
