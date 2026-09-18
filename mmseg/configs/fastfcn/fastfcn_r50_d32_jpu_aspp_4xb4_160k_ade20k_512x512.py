# Converted from configs/fastfcn/fastfcn_r50-d32_jpu_aspp_4xb4-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
# model settings
from mmengine.config import read_base

with read_base():
    from .fastfcn_r50_d32_jpu_psp_4xb4_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmseg.models import ASPPHead, CrossEntropyLoss

norm_cfg.merge(dict(type=SyncBatchNorm, requires_grad=True))
model.merge(dict(
    decode_head=dict(
        _delete_=True,
        type=ASPPHead,
        in_channels=2048,
        in_index=2,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')))
