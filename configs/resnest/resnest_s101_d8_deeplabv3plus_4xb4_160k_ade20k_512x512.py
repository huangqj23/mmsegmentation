# Converted from configs/resnest/resnest_s101-d8_deeplabv3plus_4xb4-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from ..deeplabv3plus.deeplabv3plus_r101_d8_4xb4_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmseg.models import ResNeSt

model.merge(dict(
    pretrained='open-mmlab://resnest101',
    backbone=dict(
        type=ResNeSt,
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True)))
