# Converted from configs/bisenetv2/bisenetv2_fcn_4xb4-amp-160k_cityscapes-1024x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .bisenetv2_fcn_4xb4_160k_cityscapes_1024x1024 import *  # noqa: F401,F403
optim_wrapper.merge(dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005),
    loss_scale=512.))
