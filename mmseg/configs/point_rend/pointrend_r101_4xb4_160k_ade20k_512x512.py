# Converted from configs/point_rend/pointrend_r101_4xb4-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pointrend_r50_4xb4_160k_ade20k_512x512 import *  # noqa: F401,F403
model.merge(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))
