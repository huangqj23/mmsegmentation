# Converted from configs/ocrnet/ocrnet_hr48_4xb4-40k_voc12aug-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .ocrnet_hr18_4xb4_40k_voc12aug_512x512 import *  # noqa: F401,F403

from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmseg.models import CrossEntropyLoss, FCNHead, OCRHead

norm_cfg.merge(dict(type=SyncBatchNorm, requires_grad=True))
model.merge(dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type=FCNHead,
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=21,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
        dict(
            type=OCRHead,
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=21,
            align_corners=False,
            loss_decode=dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0))
    ]))
