# Copyright (c) OpenMMLab. All rights reserved.
"""``type`` written as a class must behave exactly like the registered name."""
import os.path as osp
import tempfile

import pytest
import torch
import torch.nn as nn
from mmcv.transforms import LoadImageFromFile
from mmengine.config import Config
from mmengine.model.weight_init import PretrainedInit
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

from mmseg.apis import init_model
from mmseg.apis.mmseg_inferencer import MMSegInferencer
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.models import EncoderDecoder
from mmseg.models.backbones import CGNet, ResNetV1c, VisionTransformer
from mmseg.models.decode_heads import EncHead, FCNHead
from mmseg.utils import register_all_modules

register_all_modules()


def _built_same(make, a_arg, b_arg, seed=0):
    torch.manual_seed(seed)
    a = make(a_arg)
    torch.manual_seed(seed)
    b = make(b_arg)
    assert repr(a) == repr(b)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    return a, b


@pytest.mark.parametrize('s,c', [('BN', nn.BatchNorm2d), ('BN2d', nn.BatchNorm2d),
                                 ('SyncBN', SyncBatchNorm), ('GN', nn.GroupNorm)])
def test_enc_head_norm_to_1d(s, c):
    def make(t):
        extra = dict(num_groups=4) if t in ('GN', nn.GroupNorm) else {}
        return EncHead(in_channels=[8, 8, 8, 16], channels=16, num_classes=3,
                       in_index=[0, 1, 2, 3], num_codes=4,
                       norm_cfg=dict(type=t, **extra))
    _built_same(make, s, c)


def test_cgnet_prelu():
    def make(t):
        return CGNet(num_channels=(8, 16, 32), num_blocks=(1, 1),
                     act_cfg=dict(type=t))
    _built_same(make, 'PReLU', nn.PReLU)


def test_inferencer_transform_idx():
    idx = MMSegInferencer._get_transform_idx
    assert idx(None, [dict(type='LoadImageFromFile')], 'LoadImageFromFile') == 0
    assert idx(None, [dict(type=LoadImageFromFile)], 'LoadImageFromFile') == 0
    assert idx(None, [dict(type=LoadAnnotations)], 'LoadImageFromFile') == -1


def _small_cfg(t_model, t_backbone, t_head, t_init):
    return Config(dict(model=dict(
        type=t_model,
        data_preprocessor=dict(type='SegDataPreProcessor'),
        backbone=dict(type=t_backbone, depth=18, init_cfg=dict(
            type=t_init, checkpoint='does_not_exist.pth')),
        decode_head=dict(type=t_head, in_channels=512, channels=16,
                         num_classes=2, num_convs=1),
        test_cfg=dict(mode='whole'))))


@pytest.mark.parametrize('types', [
    ('EncoderDecoder', 'ResNetV1c', 'FCNHead', 'Pretrained'),
    (EncoderDecoder, ResNetV1c, FCNHead, PretrainedInit),
])
def test_init_model_clears_backbone_init_cfg(types):
    cfg = _small_cfg(*types)
    init_model(cfg, device='cpu')
    assert cfg.model.backbone.init_cfg is None


@pytest.mark.parametrize('t', ['Pretrained', PretrainedInit])
def test_vit_pretrained_branch(t):
    kw = dict(img_size=(32, 32), patch_size=16, embed_dims=32, num_layers=1,
              num_heads=2, out_indices=(0, ))
    src = VisionTransformer(**kw)
    with tempfile.TemporaryDirectory() as d:
        ckpt = osp.join(d, 'vit.pth')
        torch.save(src.state_dict(), ckpt)
        dst = VisionTransformer(**kw, init_cfg=dict(type=t, checkpoint=ckpt))
        dst.init_weights()
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k
