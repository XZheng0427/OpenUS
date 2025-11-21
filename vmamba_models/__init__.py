import os
from functools import partial
import torch

from .vmamba import VSSM, vanilla_vmamba_tiny, vmamba_tiny_s2l5, vmamba_tiny_m2, vmamba_small_s2l15,CSSFVSSLayer_v2, CSSFVSSLayer_v3
from .fusion_vmamba import CSSFVSSLayer_v4, single_channel_feature_extract_mamba
from .dino_vmamba import dinov2_vmamba_small


def build_vssm_model(config, **kwargs):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            posembed=config.MODEL.VSSM.POSEMBED,
            imgsize=config.DATA.IMG_SIZE,
        )
        return model
    elif model_type in ["dinovssm"]:
        # For DINOv2 compatible VSSM model
        model = DinoVSSM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            embed_dim=config.MODEL.VSSM.EMBED_DIM,
            depth=config.MODEL.VSSM.DEPTHS,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            drop_path_uniform=False,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            num_register_tokens=config.MODEL.NUM_REGISTER_TOKENS if hasattr(config.MODEL, 'NUM_REGISTER_TOKENS') else 0,
            block_chunks=config.MODEL.BLOCK_CHUNKS if hasattr(config.MODEL, 'BLOCK_CHUNKS') else 1,
        )
        return model

    return None


# def build_model(config, is_pretrain=False):
#     model = None
#     if model is None:
#         model = build_vssm_model(config)
#     if model is None:
#         from .simvmamba import simple_build
#         model = simple_build(config.MODEL.TYPE)
#     return model

def build_model(num_classes):
    model = vmamba_small_s2l15(num_classes=num_classes)

    return model


def build_dino_vssm_model(variant='small', patch_size=16, num_register_tokens=0, **kwargs):
    """Helper function to build DINOv2-compatible VSSM models"""
    if variant == 'small':
        return vssm_small(patch_size=patch_size, num_register_tokens=num_register_tokens, **kwargs)
    elif variant == 'base':
        return vssm_base(patch_size=patch_size, num_register_tokens=num_register_tokens, **kwargs)
    elif variant == 'large':
        return vssm_large(patch_size=patch_size, num_register_tokens=num_register_tokens, **kwargs)
    else:
        raise ValueError(f"Unknown DinoVSSM variant: {variant}")


def build_fusionmambav2_model(hidden_dim, depth, dpr, norm_layer, attn_drop_rate, d_state, attention_downsampling):
    model = CSSFVSSLayer_v2(
        hidden_dim=hidden_dim,
        depth=depth,
        drop_path=dpr,
        norm_layer=norm_layer,
        attn_drop_rate=attn_drop_rate,
        d_state= d_state,
        downsampling = attention_downsampling
    )

    return model


def build_fusionmambav3_model(hidden_dim, depth, dpr, norm_layer, attn_drop_rate, d_state, attention_downsampling):
    model = CSSFVSSLayer_v3(
        hidden_dim=hidden_dim,
        depth=depth,
        drop_path=dpr,
        norm_layer=norm_layer,
        attn_drop_rate=attn_drop_rate,
        d_state= d_state,
        downsampling = attention_downsampling
    )

    return model


def build_fusionmambav4_model(hidden_dim, depth, dpr, norm_layer, attn_drop_rate, d_state, attention_downsampling):
    model = CSSFVSSLayer_v4(
        hidden_dim=hidden_dim,
        depth=depth,
        drop_path=dpr,
        norm_layer=norm_layer,
        attn_drop_rate=attn_drop_rate,
        d_state= d_state,
        downsampling = attention_downsampling
    )

    return model


def build_single_channel_feature_extract_mamba():
    model = single_channel_feature_extract_mamba(channel_first=True)
    
    return model