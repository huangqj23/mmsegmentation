# Converted from configs/bisenetv1/bisenetv1_r50-d32-in1k-pre_4xb4-160k_cityscapes-1024x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .bisenetv1_r50_d32_4xb4_160k_cityscapes_1024x1024 import *  # noqa: F401,F403
model.merge(dict(
    type='EncoderDecoder',
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')))))
