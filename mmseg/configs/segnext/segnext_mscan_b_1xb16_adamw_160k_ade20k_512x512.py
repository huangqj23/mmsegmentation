# Converted from configs/segnext/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .segnext_mscan_t_1xb16_adamw_160k_ade20k_512x512 import *  # noqa: F401,F403

# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth'  # noqa
ham_norm_cfg.merge(dict(type='GN', num_groups=32, requires_grad=True))
model.merge(dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        drop_path_rate=0.1,
        norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')))
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
