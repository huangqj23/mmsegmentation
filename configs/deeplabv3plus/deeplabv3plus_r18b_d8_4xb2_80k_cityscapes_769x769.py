# Converted from configs/deeplabv3plus/deeplabv3plus_r18b-d8_4xb2-80k_cityscapes-769x769.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .deeplabv3plus_r50_d8_4xb2_40k_cityscapes_769x769 import *  # noqa: F401,F403

from mmseg.models import ResNet

model.merge(dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type=ResNet, depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)))
