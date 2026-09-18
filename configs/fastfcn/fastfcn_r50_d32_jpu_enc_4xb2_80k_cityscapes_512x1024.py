# Converted from configs/fastfcn/fastfcn_r50-d32_jpu_enc_4xb2-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
# model settings
from mmengine.config import read_base

with read_base():
    from .fastfcn_r50_d32_jpu_psp_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403

from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmseg.models import CrossEntropyLoss, EncHead

norm_cfg.merge(dict(type=SyncBatchNorm, requires_grad=True))
model.merge(dict(
    decode_head=dict(
        _delete_=True,
        type=EncHead,
        in_channels=[512, 1024, 2048],
        in_index=(0, 1, 2),
        channels=512,
        num_codes=32,
        use_se_loss=True,
        add_lateral=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
        loss_se_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=True, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')))
