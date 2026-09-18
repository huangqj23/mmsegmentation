# Converted from configs/maskformer/maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .maskformer_r50_d32_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit
from mmengine.optim import LinearLR, OptimWrapper, PolyLR
from torch.nn import GELU, LayerNorm
from torch.optim import AdamW

from mmseg.models import MaskFormerHead, SwinTransformer


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_model = model
__base_optim_wrapper = optim_wrapper
__base_optimizer = optimizer
backbone_norm_cfg = dict(type=LayerNorm, requires_grad=True)
depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        _delete_=True,
        type=SwinTransformer,
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type=GELU),
        norm_cfg=backbone_norm_cfg,
        init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint_file)),
    decode_head=dict(
        type=MaskFormerHead,
        in_channels=[96, 192, 384,
                     768],  # input channels of pixel_decoder modules
    ))

# optimizer
optimizer = dict(
    type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
# set all layers in backbone to lr_mult=1.0
# set all norm layers, position_embeding,
# query_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=1.0, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
embed_multi = dict(decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    _delete_=True,
    type=OptimWrapper,
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys))

# learning policy
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
model = __iv_merge(__base_model, model)
optimizer = __iv_merge(__base_optimizer, optimizer)
optim_wrapper = __iv_merge(__base_optim_wrapper, optim_wrapper)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
