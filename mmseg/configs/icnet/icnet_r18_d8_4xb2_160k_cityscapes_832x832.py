# Converted from configs/icnet/icnet_r18-d8_4xb2-160k_cityscapes-832x832.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .icnet_r50_d8_4xb2_160k_cityscapes_832x832 import *  # noqa: F401,F403
model.merge(dict(
    backbone=dict(layer_channels=(128, 512), backbone_cfg=dict(depth=18))))
