# Converted from configs/upernet/upernet_r101_4xb4-20k_voc12aug-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .upernet_r50_4xb4_20k_voc12aug_512x512 import *  # noqa: F401,F403
model.merge(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))
