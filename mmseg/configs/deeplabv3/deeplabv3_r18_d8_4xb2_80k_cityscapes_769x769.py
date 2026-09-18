# Converted from configs/deeplabv3/deeplabv3_r18-d8_4xb2-80k_cityscapes-769x769.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .deeplabv3_r50_d8_4xb2_80k_cityscapes_769x769 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)))
