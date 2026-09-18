# Converted from projects/van/configs/van/van-b4_upernet_4xb4-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .van_b2_upernet_4xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b4_3rdparty_20230522-1d71c077.pth'  # noqa
model.merge(dict(
    backbone=dict(
        depths=[3, 6, 40, 3],
        init_cfg=dict(type=PretrainedInit, checkpoint=ckpt_path),
        drop_path_rate=0.4)))

# By default, models are trained on 4 GPUs with 4 images per GPU
train_dataloader.merge(dict(batch_size=4))
if isinstance(ckpt_path, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    ckpt_path.pop('_delete_', None)
