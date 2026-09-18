from functools import partial

def resolve_loss(name : str, **params):
    if name == 'noise_loss':
        from .noise_loss import noise_loss
        return partial(noise_loss, **params)
    else:
        raise NotImplementedError(f'Loss {name} not implemented')
