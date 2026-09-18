# Converted from projects/van/configs/van/van-b2_upernet_4xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.van_upernet import *  # noqa: F401,F403
    from .._base_.datasets.ade20k import *  # noqa: F401,F403
    from mmseg.configs._base_.default_runtime import *  # noqa: F401,F403
    from mmseg.configs._base_.schedules.schedule_160k import *  # noqa: F401,F403

from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, OptimWrapper, PolyLR
from torch.optim import AdamW

from mmseg.models import EncoderDecoder

custom_imports = dict(imports=['projects.van.backbones'])
ckpt_path = 'https://download.openmmlab.com/mmsegmentation/v0.5/van_3rdparty/van-b2_3rdparty_20230522-636fac93.pth'  # noqa
model.merge(dict(
    type=EncoderDecoder,
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[3, 3, 12, 3],
        init_cfg=dict(type=PretrainedInit, checkpoint=ckpt_path)),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=150),
    auxiliary_head=dict(in_channels=320, num_classes=150)))

# AdamW optimizer
# no weight decay for position embedding & layer norm in backbone
optim_wrapper.merge(dict(
    _delete_=True,
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=None,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        })))
# learning policy
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=train_cfg.max_iters,
        eta_min=0.0,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader.merge(dict(batch_size=2))
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
