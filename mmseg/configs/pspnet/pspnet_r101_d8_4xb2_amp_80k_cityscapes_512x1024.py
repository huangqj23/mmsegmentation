# Converted from configs/pspnet/pspnet_r101-d8_4xb2-amp-80k_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .pspnet_r101_d8_4xb2_80k_cityscapes_512x1024 import *  # noqa: F401,F403

from mmengine.optim import AmpOptimWrapper
from torch.optim import SGD

optim_wrapper.merge(dict(
    _delete_=True,
    type=AmpOptimWrapper,
    optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005),
    loss_scale=512.))
