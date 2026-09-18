# Converted from configs/ocrnet/ocrnet_r101-d8_8xb2-40k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.ocrnet_r50_d8 import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_40k import *  # noqa: F401,F403
crop_size = (512, 1024)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101)))
optimizer.merge(dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005))
optim_wrapper.merge(dict(type='OptimWrapper', optimizer=optimizer))
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=2e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
