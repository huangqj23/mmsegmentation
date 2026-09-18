# Converted from projects/isnet/configs/isnet_r50-d8_8xb2-160k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from mmseg.configs._base_.datasets.cityscapes import *  # noqa: F401,F403
    from mmseg.configs._base_.default_runtime import *  # noqa: F401,F403
    from mmseg.configs._base_.schedules.schedule_80k import *  # noqa: F401,F403

from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm
from projects.isnet.decode_heads import ISNetHead

from mmseg.models import (CrossEntropyLoss, EncoderDecoder, FCNHead, ResNetV1c,
                          SegDataPreProcessor)

data_root = '../../data/cityscapes/'
train_dataloader.merge(dict(dataset=dict(data_root=data_root)))
val_dataloader.merge(dict(dataset=dict(data_root=data_root)))
test_dataloader.merge(dict(dataset=dict(data_root=data_root)))

custom_imports = dict(imports=['projects.isnet.decode_heads'])

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
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type=ResNetV1c,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type=ISNetHead,
        in_channels=(256, 512, 1024, 2048),
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        channels=512,
        dropout_ratio=0.1,
        transform_channels=256,
        concat_input=True,
        with_shortcut=False,
        shortcut_in_channels=256,
        shortcut_feat_channels=48,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss,
                use_sigmoid=False,
                loss_weight=1.0,
                loss_name='loss_o'),
            dict(
                type=CrossEntropyLoss,
                use_sigmoid=False,
                loss_weight=0.4,
                loss_name='loss_d'),
        ]),
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=1024,
        in_index=2,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    # test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
    test_cfg=dict(mode='whole'))
if isinstance(data_root, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    data_root.pop('_delete_', None)
