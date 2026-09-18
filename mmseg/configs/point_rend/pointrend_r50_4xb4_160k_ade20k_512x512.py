# Converted from configs/point_rend/pointrend_r50_4xb4-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.pointrend_r50 import *  # noqa: F401,F403
    from .._base_.datasets.ade20k import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_160k import *  # noqa: F401,F403

from mmengine.optim import LinearLR, PolyLR
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmseg.models import CrossEntropyLoss, FPNHead, PointHead

crop_size = (512, 512)
data_preprocessor.merge(dict(size=crop_size))
norm_cfg.merge(dict(type=SyncBatchNorm, requires_grad=True))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    decode_head=[
        dict(
            type=FPNHead,
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=-1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
        dict(
            type=PointHead,
            in_channels=[256],
            in_index=[0],
            channels=256,
            num_fcs=3,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=150,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0))
    ]))
param_scheduler = [
    dict(type=LinearLR, by_epoch=False, start_factor=0.1, begin=0, end=200),
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=200,
        end=160000,
        by_epoch=False,
    )
]
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
