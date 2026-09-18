# Converted from configs/fcn/fcn-d6_r101b-d16_4xb2-80k_cityscapes-769x769.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .fcn_d6_r50b_d16_4xb2_80k_cityscapes_769x769 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101)))
