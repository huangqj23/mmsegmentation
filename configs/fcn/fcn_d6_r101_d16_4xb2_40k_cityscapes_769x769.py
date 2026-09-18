# Converted from configs/fcn/fcn-d6_r101-d16_4xb2-40k_cityscapes-769x769.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .fcn_d6_r50_d16_4xb2_40k_cityscapes_769x769 import *  # noqa: F401,F403
model.merge(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))
