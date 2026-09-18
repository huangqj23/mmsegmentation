# Converted from configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .mask2former_swin_t_8xb2_160k_ade20k_512x512 import *  # noqa: F401,F403

from mmengine.model import PretrainedInit

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa

depths = [2, 2, 18, 2]
model.merge(dict(
    backbone=dict(
        depths=depths, init_cfg=dict(type=PretrainedInit,
                                     checkpoint=pretrained))))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi.merge(dict(lr_mult=0.1, decay_mult=0.0))
backbone_embed_multi.merge(dict(lr_mult=0.1, decay_mult=0.0))
embed_multi.merge(dict(lr_mult=1.0, decay_mult=0.0))
custom_keys.merge({
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
})
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
optim_wrapper.merge(dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)))
if isinstance(pretrained, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    pretrained.pop('_delete_', None)
if isinstance(depths, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    depths.pop('_delete_', None)
