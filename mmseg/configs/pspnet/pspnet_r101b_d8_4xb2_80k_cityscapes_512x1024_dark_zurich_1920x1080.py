# Converted from configs/pspnet/pspnet_r101b-d8_4xb2-80k_cityscapes-512x1024_dark-zurich-1920x1080.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pspnet_r50_d8_4xb2_80k_cityscapes_512x1024_dark_zurich_1920x1080 import *  # noqa: F401,F403

from mmseg.models import ResNet

  # noqa
model.merge(dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type=ResNet, depth=101)))
