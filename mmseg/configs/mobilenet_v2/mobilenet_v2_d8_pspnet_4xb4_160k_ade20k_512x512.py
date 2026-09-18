# Converted from configs/mobilenet_v2/mobilenet-v2-d8_pspnet_4xb4-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..pspnet.pspnet_r101_d8_4xb4_160k_ade20k_512x512 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(in_channels=320),
    auxiliary_head=dict(in_channels=96)))
