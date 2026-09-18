# Converted from configs/pspnet/pspnet_r101-d8_4xb2-40k_cityscapes-769x769.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pspnet_r50_d8_4xb2_40k_cityscapes_769x769 import *  # noqa: F401,F403
model.merge(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))
