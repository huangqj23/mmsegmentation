# Converted from configs/bisenetv1/bisenetv1_r18-d32-in1k-pre_4xb4-160k_coco-stuff164k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .bisenetv1_r18_d32_4xb4_160k_coco_stuff164k_512x512 import *  # noqa: F401,F403
crop_size = (512, 512)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))),
))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
