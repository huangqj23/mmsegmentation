# Converted from configs/bisenetv1/bisenetv1_r50-d32_4xb4-160k_cityscapes-1024x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base
from copy import deepcopy as __deepcopy

with read_base():
    from .._base_.models.bisenetv1_r18_d32 import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes_1024x1024 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_160k import *  # noqa: F401,F403

from mmengine.optim import LinearLR, OptimWrapper, PolyLR
from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm
from torch.optim import SGD

from mmseg.models import BiSeNetV1, EncoderDecoder, FCNHead, ResNet


def __iv_merge(base, child):
    """旧式继承的递归合并（支持 _delete_）。生成新对象、不修改 base 原对象 ——
    它可能正被 x 形式的引用持有（旧式合并同样不会改动 base 原对象）。"""
    out = __deepcopy(base)
    out.merge(child)
    return out

# 旧式继承中，子配置里的名字在整个文件内都指向子配置自己的值，
# 与 base 的合并在最后发生。先保存 base 值以保持完全等价。
__base_test_dataloader = test_dataloader
norm_cfg.merge(dict(type=SyncBatchNorm, requires_grad=True))
crop_size = (1024, 1024)
data_preprocessor.merge(dict(size=crop_size))
model.merge(dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=BiSeNetV1,
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_channels=1024,
        backbone_cfg=dict(type=ResNet, depth=50)),
    decode_head=dict(
        type=FCNHead, in_channels=1024, in_index=0, channels=1024),
    auxiliary_head=[
        dict(
            type=FCNHead,
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False),
        dict(
            type=FCNHead,
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False),
    ]))
param_scheduler = [
    dict(type=LinearLR, by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=160000,
        by_epoch=False,
    )
]
optimizer.merge(dict(type=SGD, lr=0.05, momentum=0.9, weight_decay=0.0005))
optim_wrapper.merge(dict(type=OptimWrapper, optimizer=optimizer))
train_dataloader.merge(dict(batch_size=4, num_workers=4))
val_dataloader.merge(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader

# ---- 旧式继承语义：子配置的值最后才与 base 递归合并 ----
if isinstance(test_dataloader, dict):
    test_dataloader = __iv_merge(__base_test_dataloader, test_dataloader)
if isinstance(crop_size, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    crop_size.pop('_delete_', None)
if isinstance(param_scheduler, dict):  # base 中是非 dict，旧式替换前弹掉顶层 _delete_
    param_scheduler.pop('_delete_', None)
