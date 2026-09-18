# Converted from projects/Adabins/configs/adabins/adabins_efficient_b5_4x16_25e_kitti_352x704.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.Adabins import *  # noqa: F401,F403

from mmengine.utils.dl_utils.parrots_wrapper import SyncBatchNorm

custom_imports = dict(
    imports=['projects.Adabins.backbones', 'projects.Adabins.decode_head'],
    allow_failed_imports=False)
crop_size = (352, 704)
data_preprocessor.merge(dict(size=crop_size))
norm_cfg.merge(dict(type=SyncBatchNorm, requires_grad=True))
model.merge(dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(),
    decode_head=dict(min_val=0.001, max_val=80),
))
