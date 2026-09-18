# Converted from configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .mask2former_swin_b_in1k_384x384_pre_8xb2_160k_ade20k_640x640 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa
model.merge(dict(
    backbone=dict(init_cfg=dict(type=PretrainedInit, checkpoint=pretrained))))
if isinstance(pretrained, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    pretrained.pop('_delete_', None)
