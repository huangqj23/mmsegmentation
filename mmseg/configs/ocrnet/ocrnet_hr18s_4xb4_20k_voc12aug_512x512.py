# Converted from configs/ocrnet/ocrnet_hr18s_4xb4-20k_voc12aug-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .ocrnet_hr18_4xb4_20k_voc12aug_512x512 import *  # noqa: F401,F403
model.merge(dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2))))))
