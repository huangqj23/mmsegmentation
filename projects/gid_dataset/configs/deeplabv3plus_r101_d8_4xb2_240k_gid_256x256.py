# Converted from projects/gid_dataset/configs/deeplabv3plus_r101-d8_4xb2-240k_gid-256x256.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from mmseg.configs._base_.models.deeplabv3plus_r50_d8 import *  # noqa: F401,F403
    from ._base_.datasets.gid import *  # noqa: F401,F403
    from mmseg.configs._base_.default_runtime import *  # noqa: F401,F403
    from mmseg.configs._base_.schedules.schedule_240k import *  # noqa: F401,F403
custom_imports = dict(imports=['projects.gid_dataset.mmseg.datasets.gid'])

crop_size = (256, 256)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6)))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
