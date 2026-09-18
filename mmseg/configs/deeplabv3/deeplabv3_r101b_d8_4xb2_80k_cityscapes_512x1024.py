# Converted from configs/deeplabv3/deeplabv3_r101b-d8_4xb2-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .deeplabv3_r50_d8_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101)))
