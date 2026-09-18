
from .ddim_sampler import DDIM
from .ddim_iscs_sampler import DDIM_ISCS
from typing import Any

def get_sampler(name : str, **cfg_kwargs) -> Any:
    if name == "ddim":
        return DDIM(**cfg_kwargs)
    elif name == "ddim_iscs":
        return DDIM_ISCS(**cfg_kwargs)
    else:
        raise NotImplementedError(f"Sampler {name} not implemented")