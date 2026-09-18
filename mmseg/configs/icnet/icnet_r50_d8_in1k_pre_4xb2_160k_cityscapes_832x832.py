# Converted from configs/icnet/icnet_r50-d8-in1k-pre_4xb2-160k_cityscapes-832x832.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .icnet_r50_d8_4xb2_160k_cityscapes_832x832 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

model.merge(dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type=PretrainedInit, checkpoint='open-mmlab://resnet50_v1c')))))
