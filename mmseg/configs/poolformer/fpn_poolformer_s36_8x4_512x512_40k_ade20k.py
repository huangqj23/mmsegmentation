# Converted from configs/poolformer/fpn_poolformer_s36_8x4_512x512_40k_ade20k.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .fpn_poolformer_s12_8xb4_40k_ade20k_512x512 import *  # noqa: F401,F403
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth'  # noqa

# model settings
model.merge(dict(
    backbone=dict(
        arch='s36',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.'))))
if isinstance(checkpoint_file, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint_file.pop('_delete_', None)
