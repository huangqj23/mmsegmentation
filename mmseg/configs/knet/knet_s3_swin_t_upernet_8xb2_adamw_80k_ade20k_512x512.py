# Converted from configs/knet/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .knet_s3_r50_d8_upernet_8xb2_adamw_80k_ade20k_512x512 import *  # noqa: F401,F403


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220308-f41b89d3.pth'  # noqa

# model settings
norm_cfg.merge(dict(type='SyncBN', requires_grad=True))
num_stages = 3
conv_kernel_size = 1

model.merge(dict(
    type='EncoderDecoder',
    pretrained=checkpoint_file,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        kernel_generate_head=dict(in_channels=[96, 192, 384, 768])),
    auxiliary_head=dict(in_channels=384)))

optim_wrapper.merge(dict(
    _delete_=True,
    type='OptimWrapper',
    # modify learning rate following the official implementation of Swin Transformer # noqa
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=dict(max_norm=1, norm_type=2)))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=1000,
        end=80000,
        milestones=[60000, 72000],
        by_epoch=False,
    )
]
# In K-Net implementation we use batch size 2 per GPU as default
train_dataloader.merge(dict(batch_size=2, num_workers=2))
val_dataloader.merge(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(num_stages, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    num_stages.pop('_delete_', None)
if isinstance(conv_kernel_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    conv_kernel_size.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
