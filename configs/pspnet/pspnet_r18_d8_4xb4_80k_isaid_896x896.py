# Converted from configs/pspnet/pspnet_r18-d8_4xb4-80k_isaid-896x896.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pspnet_r50_d8_4xb4_80k_isaid_896x896 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)))
