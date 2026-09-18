"""Serializes wandb's init()/log()/finish() usage across concurrent threads."""

import os
import threading
from contextlib import contextmanager

wandb_lock = threading.Lock()

_INIT_TIMEOUT_S = float(os.environ.get("RESROB_WANDB_INIT_TIMEOUT", "300"))


@contextmanager
def locked_wandb_init(**kwargs):
    import wandb

    if "settings" not in kwargs:
        kwargs = {**kwargs, "settings": wandb.Settings(init_timeout=_INIT_TIMEOUT_S)}

    with wandb_lock:
        with wandb.init(**kwargs) as run:
            yield run
