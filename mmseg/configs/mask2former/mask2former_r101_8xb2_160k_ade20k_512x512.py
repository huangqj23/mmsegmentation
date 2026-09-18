# Converted from configs/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .mask2former_r50_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

model.merge(dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101'))))
