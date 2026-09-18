# Converted from configs/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .mask2former_r50_8xb2_90k_cityscapes_512x1024 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

model.merge(dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type=PretrainedInit,
                      checkpoint='torchvision://resnet101'))))
