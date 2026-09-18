# Converted from configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-40k_pascal-context-59-480x480.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .deeplabv3plus_r50_d8_4xb4_40k_pascal_context_59_480x480 import *  # noqa: F401,F403
model.merge(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))
