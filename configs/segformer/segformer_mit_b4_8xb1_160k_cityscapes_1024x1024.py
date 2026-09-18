# Converted from configs/segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .segformer_mit_b0_8xb1_160k_cityscapes_1024x1024 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa

model.merge(dict(
    backbone=dict(
        init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512])))
if isinstance(checkpoint, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint.pop('_delete_', None)
