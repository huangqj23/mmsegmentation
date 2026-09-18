# Converted from configs/stdc/stdc1_in1k-pre_4xb12-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc1_20220308-5368626c.pth'  # noqa
from mmengine.config import read_base

with read_base():
    from .stdc1_4xb12_80k_cityscapes_512x1024 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

model.merge(dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint)))))
