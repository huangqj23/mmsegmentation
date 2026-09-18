# Converted from projects/pp_mobileseg/configs/_base_/models/pp_mobile.py by industrial-vision tools/convert_configs.py
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm
from projects.pp_mobileseg.backbones import StrideFormer
from projects.pp_mobileseg.decode_head import PPMobileSegHead
from torch.nn import ReLU, ReLU6

from mmseg.models import EncoderDecoder, SegDataPreProcessor

# model settings
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type=StrideFormer,
        mobileV3_cfg=[
            # k t c, s
            [[3, 16, 16, True, 'ReLU', 1], [3, 64, 32, False, 'ReLU', 2],
             [3, 96, 32, False, 'ReLU', 1]],  # cfg1
            [[5, 128, 64, True, 'HSwish', 2], [5, 240, 64, True, 'HSwish',
                                               1]],  # cfg2
            [[5, 384, 128, True, 'HSwish', 2],
             [5, 384, 128, True, 'HSwish', 1]],  # cfg3
            [[5, 768, 192, True, 'HSwish', 2],
             [5, 768, 192, True, 'HSwish', 1]],  # cfg4
        ],
        channels=[16, 32, 64, 128, 192],
        depths=[3, 3],
        embed_dims=[128, 192],
        num_heads=8,
        inj_type='AAMSx8',
        out_feat_chs=[64, 128, 192],
        act_cfg=dict(type=ReLU6),
    ),
    decode_head=dict(
        type=PPMobileSegHead,
        num_classes=150,
        in_channels=256,
        dropout_ratio=0.1,
        use_dw=True,
        act_cfg=dict(type=ReLU),
        align_corners=False),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
