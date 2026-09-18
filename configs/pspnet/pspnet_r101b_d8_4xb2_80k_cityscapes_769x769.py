# Converted from configs/pspnet/pspnet_r101b-d8_4xb2-80k_cityscapes-769x769.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pspnet_r50_d8_4xb2_80k_cityscapes_769x769 import *  # noqa: F401,F403

from mmseg.models import ResNet

model.merge(dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type=ResNet, depth=101)))
