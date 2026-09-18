# Converted from configs/bisenetv1/bisenetv1_r101-d32-in1k-pre_4xb4-160k_coco-stuff164k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .bisenetv1_r101_d32_4xb4_160k_coco_stuff164k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

model.merge(dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type=PretrainedInit, checkpoint='open-mmlab://resnet101_v1c')))))
