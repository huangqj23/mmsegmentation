# Converted from configs/pspnet/pspnet_r50-d32_rsb_4xb2-adamw-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.pspnet_r50_d8 import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403

from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, MultiStepLR, OptimWrapper
from torch.optim import AdamW

from mmseg.models import ResNet

crop_size = (512, 1024)
data_preprocessor.merge(dict(size=crop_size))
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
model.merge(dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=ResNet,
        init_cfg=dict(
            type=PretrainedInit, prefix='backbone.', checkpoint=checkpoint),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 2))))

optim_wrapper.merge(dict(
    _delete_=True,
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=0.0005, weight_decay=0.05),
    clip_grad=dict(max_norm=1, norm_type=2)))
# learning policy
param_scheduler = [
    dict(
        type=LinearLR, start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type=MultiStepLR,
        begin=1000,
        end=80000,
        by_epoch=False,
        milestones=[60000, 72000],
    )
]
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
