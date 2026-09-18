# Converted from configs/resnest/resnest_s101-d8_deeplabv3plus_4xb2-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..deeplabv3plus.deeplabv3plus_r101_d8_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403
  # noqa
model.merge(dict(
    pretrained='open-mmlab://resnest101',
    backbone=dict(
        type='ResNeSt',
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True)))
