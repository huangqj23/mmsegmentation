# Converted from configs/ocrnet/ocrnet_hr18_4xb4-20k_voc12aug-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.ocrnet_hr18 import *  # noqa: F401,F403
    from .._base_.datasets.pascal_voc12_aug import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_20k import *  # noqa: F401,F403
crop_size = (512, 512)
data_preprocessor.merge(dict(size=crop_size))
norm_cfg.merge(dict(type='SyncBN', requires_grad=True))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[18, 36, 72, 144],
            channels=sum([18, 36, 72, 144]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=21,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            channels=512,
            ocr_channels=256,
            dropout_ratio=-1,
            num_classes=21,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ]))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
