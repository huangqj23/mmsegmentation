# Converted from configs/fcn/fcn_r18b-d8_4xb2-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .fcn_r50_d8_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)))
