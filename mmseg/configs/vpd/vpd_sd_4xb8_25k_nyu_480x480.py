# Converted from configs/vpd/vpd_sd_4xb8-25k_nyu-480x480.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.vpd_sd import *  # noqa: F401,F403
    from .._base_.datasets.nyu import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_25k import *  # noqa: F401,F403

crop_size = (480, 480)

model.merge(dict(
    type='DepthEstimator',
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        class_embed_path='https://download.openmmlab.com/mmsegmentation/'
        'v0.5/vpd/nyu_class_embeddings.pth',
        class_embed_select=True,
        pad_shape=512,
        unet_cfg=dict(use_attn=False),
    ),
    decode_head=dict(
        type='VPDDepthHead',
        in_channels=[320, 640, 1280, 1280],
        max_depth=10,
        fmap_border=(1, 1),
    ),
    test_cfg=dict(mode='slide_flip', crop_size=crop_size, stride=(160, 160))))

default_hooks.merge(dict(
    checkpoint=dict(save_best='rmse', rule='less', max_keep_ckpts=1)))

# custom optimizer
optim_wrapper.merge(dict(
    constructor='ForceDefaultOptimWrapperConstructor',
    paramwise_cfg=dict(
        bias_decay_mult=0,
        force_default_settings=True,
        custom_keys={
            'backbone.encoder_vq': dict(lr_mult=0),
            'backbone.unet': dict(lr_mult=0.01),
        })))
