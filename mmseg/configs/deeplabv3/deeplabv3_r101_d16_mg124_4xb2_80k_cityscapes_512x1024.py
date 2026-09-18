# Converted from configs/deeplabv3/deeplabv3_r101-d16-mg124_4xb2-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .deeplabv3_r50_d8_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        depth=101,
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        multi_grid=(1, 2, 4)),
    decode_head=dict(
        dilations=(1, 6, 12, 18),
        sampler=dict(type='OHEMPixelSampler', min_kept=100000))))
