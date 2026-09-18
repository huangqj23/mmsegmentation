# Converted from projects/van/configs/van/van-b3_fpn_8xb4-40k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .van_b2_fpn_8xb4_40k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

from mmseg.models import EncoderDecoder

ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b3_3rdparty_20230522-a184e051.pth'  # noqa
model.merge(dict(
    type=EncoderDecoder,
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 5, 27, 3],
        init_cfg=dict(type=PretrainedInit, checkpoint=ckpt_path),
        drop_path_rate=0.3),
    neck=dict(in_channels=[64, 128, 320, 512])))
train_dataloader.merge(dict(batch_size=4))
if isinstance(ckpt_path, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    ckpt_path.pop('_delete_', None)
