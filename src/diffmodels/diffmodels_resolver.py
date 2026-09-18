
import torch
import logging

from .archs.std.unet import UNetModel
from .ema import ExponentialMovingAverage

# from src.utils.path_utils import get_path_by_cluster_name



def create_dense_model(
    num_channels: int,
    in_channels: int,
    out_channels: int,
    num_res_blocks: int,
    channel_mult: str = '',
    use_checkpoint: bool = False,
    attention_resolutions: str = '16',
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0.,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    resamp_with_conv : bool = True,
    learn_sigma : bool = False,
    **kwargs
):

    logging.info(f"Unused kwargs: {kwargs}")
    
    attention_ds = []
    for res in attention_resolutions.split(","):
        #attention_ds.append(image_size // int(res))
        attention_ds.append(int(res)) # this is different now

    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else in_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=tuple(channel_mult),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        conv_resample=resamp_with_conv
    )

def create_model(name: str, params, arch_cfg = None) -> UNetModel:

    if name == "dense":
        return create_dense_model(**params)
    elif name == "depthwise_dm" or name == "sparse_fno":
        from .archs.infdm.model_sparse import SparseUNet
        if arch_cfg is not None:
            # there are certain parameters we overwrite
            if not "knn_neighbours" in params or arch_cfg.params.knn_neighbours != params["knn_neighbours"]:
                logging.warning(f"Overwriting knn_neighbours from {params['knn_neighbours'] if 'knn_neighbours' in params else 'unknown'} to {arch_cfg.params.knn_neighbours}")
                params["knn_neighbours"] = arch_cfg.params.knn_neighbours
            
            if not "kernel_interpolation_method" in params or arch_cfg.params.kernel_interpolation_method != params["kernel_interpolation_method"]:
                logging.warning(f"Overwriting kernel_interpolation_method from {params['kernel_interpolation_method'] if 'kernel_interpolation_method' in params else 'unknown'} to {arch_cfg.params.kernel_interpolation_method}")
                params["kernel_interpolation_method"] = arch_cfg.params.kernel_interpolation_method

        return SparseUNet(**params)
    else:
        raise ValueError(f"Unknown model name {name}")

def load_score_model(score_model, model_use_ema : bool):
    model_path = "model.pt" if not model_use_ema else "ema_model.pt" # see utils_save
    if model_use_ema:
        ema = ExponentialMovingAverage(score_model.parameters(), decay=0.999)
        ema.load_state_dict(torch.load(model_path, map_location='cpu'))
        ema.copy_to(score_model.parameters())
        logging.info(f'model ema ckpt loaded from: {model_path}')
    else:
        score_model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        logging.info(f'model ckpt loaded from: {model_path}')
