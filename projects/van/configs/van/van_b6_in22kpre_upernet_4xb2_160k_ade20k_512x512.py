# Converted from projects/van/configs/van/van-b6-in22kpre_upernet_4xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .van_b2_upernet_4xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b6-in22k_3rdparty_20230522-5e5172a3.pth'  # noqa
model.merge(dict(
    backbone=dict(
        embed_dims=[96, 192, 384, 768],
        depths=[6, 6, 90, 6],
        init_cfg=dict(type=PretrainedInit, checkpoint=ckpt_path),
        drop_path_rate=0.5),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150)))
if isinstance(ckpt_path, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    ckpt_path.pop('_delete_', None)
