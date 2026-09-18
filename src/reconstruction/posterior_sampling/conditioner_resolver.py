
from src.diffmodels.sampler.base_conditioning_method import ConditioningMethod
from .dds_conditioner import DecomposedDiffusionSampling
from .dds_tv_conditioner import DecomposedDiffusionSamplingWithTV

def get_conditioning_method(name : str, **kwargs) -> ConditioningMethod:
    if name == 'dds':
        return DecomposedDiffusionSampling(**kwargs)
    elif name == 'dds_tv':
        return DecomposedDiffusionSamplingWithTV(**kwargs)
    else:
        raise NotImplementedError(f'Conditioning method {name} not implemented')