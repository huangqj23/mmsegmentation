# Converted from configs/pspnet/pspnet_r101-d8_4xb4-80k_pascal-context-59-480x480.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pspnet_r50_d8_4xb4_80k_pascal_context_59_480x480 import *  # noqa: F401,F403
model.merge(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))
