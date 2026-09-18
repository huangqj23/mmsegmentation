# Converted from configs/segmenter/segmenter_vit-l_mask_8xb1-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.segmenter_vit_b16_mask import *  # noqa: F401,F403
    from .._base_.datasets.ade20k_640x640 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_160k import *  # noqa: F401,F403
crop_size = (640, 640)
data_preprocessor.merge(dict(size=crop_size))
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'  # noqa

model.merge(dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type='VisionTransformer',
        img_size=(640, 640),
        embed_dims=1024,
        num_layers=24,
        num_heads=16),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        num_heads=16,
        embed_dims=1024),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608))))

optimizer.merge(dict(lr=0.001, weight_decay=0.0))
optim_wrapper.merge(dict(type='OptimWrapper', optimizer=optimizer))
train_dataloader.merge(dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=1))
val_dataloader.merge(dict(batch_size=1))
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(checkpoint, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    checkpoint.pop('_delete_', None)
