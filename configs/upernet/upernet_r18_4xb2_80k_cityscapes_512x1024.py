# Converted from configs/upernet/upernet_r18_4xb2-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .upernet_r50_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=[64, 128, 256, 512]),
    auxiliary_head=dict(in_channels=256)))
