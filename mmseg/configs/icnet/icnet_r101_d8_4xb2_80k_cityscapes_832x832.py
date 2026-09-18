# Converted from configs/icnet/icnet_r101-d8_4xb2-80k_cityscapes-832x832.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .icnet_r50_d8_4xb2_80k_cityscapes_832x832 import *  # noqa: F401,F403
model.merge(dict(backbone=dict(backbone_cfg=dict(depth=101))))
