# Converted from configs/stdc/stdc2_4xb12-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .stdc1_4xb12_80k_cityscapes_512x1024 import *  # noqa: F401,F403
model.merge(dict(backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2'))))
