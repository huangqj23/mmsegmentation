# Converted from configs/twins/twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .twins_svt_s_fpn_fpnhead_8xb4_80k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/alt_gvt_base_20220308-1b7eb711.pth'  # noqa

model.merge(dict(
    backbone=dict(
        init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint),
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 18, 2]),
    neck=dict(in_channels=[96, 192, 384, 768]),
))
if isinstance(checkpoint, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint.pop('_delete_', None)
