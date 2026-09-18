# Converted from projects/CAT-Seg/configs/cat_seg/catseg_vitl-swin-b_4xb1-warmcoslr2e-4_adamw-80k_coco-stuff164k_384x384.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .catseg_vitb_r101_4xb2_warmcoslr2e_4_adamw_80k_coco_stuff164k_384x384 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit
from mmengine.optim import CosineAnnealingLR, LinearLR, OptimWrapper
from mmengine.runner import IterBasedTrainLoop
from torch.optim import AdamW

from mmseg.engine import SegVisualizationHook
from mmseg.models import SwinTransformer

  # noqa

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_20220317-55b0104a.pth'  # noqa
crop_size = (384, 384)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    backbone=dict(
        type='CLIPOVCATSeg',
        feature_extractor=dict(
            _delete_=True,
            type=SwinTransformer,
            pretrain_img_size=384,
            embed_dims=128,
            depths=[2, 2, 18],
            num_heads=[4, 8, 16],
            window_size=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2),
            init_cfg=dict(type=PretrainedInit, checkpoint=pretrained)),
        clip_pretrained='ViT-L/14@336px',
    ),
    neck=dict(
        text_guidance_dim=768,
        appearance_guidance_dim=512,
    ),
    decode_head=dict(
        embed_dims=128,
        decoder_guidance_dims=(256, 128),
    )))

# dataset settings
train_dataloader.merge(dict(
    batch_size=1,
    num_workers=2,
))

# training schedule for 80k
train_cfg.merge(dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000))

default_hooks.merge(dict(
    visualization=dict(type=SegVisualizationHook, draw=True, interval=4000)))

# optimizer
optim_wrapper.merge(dict(
    _delete_=True,
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.0002, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.feature_extractor': dict(lr_mult=0.01),
            'backbone.clip_model.visual': dict(lr_mult=0.01)
        })))

# learning policy
param_scheduler = [
    # Use a linear warm-up at [0, 100) iterations
    dict(type=LinearLR, start_factor=0.01, by_epoch=False, begin=0, end=500),
    # Use a cosine learning rate at [100, 900) iterations
    dict(
        type=CosineAnnealingLR,
        T_max=79500,
        by_epoch=False,
        begin=500,
        end=80000),
]
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
