# Converted from projects/van/configs/van/van-b0_fpn_8xb4-40k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .van_b2_fpn_8xb4_40k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b0_3rdparty_20230522-956f5e0d.pth'  # noqa
model.merge(dict(
    backbone=dict(
        embed_dims=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],
        init_cfg=dict(type=PretrainedInit, checkpoint=ckpt_path)),
    neck=dict(in_channels=[32, 64, 160, 256])))
if isinstance(ckpt_path, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    ckpt_path.pop('_delete_', None)
