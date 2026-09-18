# Converted from projects/hssn/configs/hssn/hieraseg_deeplabv3plus_r101-d8_4xb2-80l_cityscapes-512x1024.py by industrial-vision tools/convert_configs.py
from mmengine.config import read_base

with read_base():
    from .._base_.models.deeplabv3plus_r50_d8_vd_contrast import *  # noqa: F401,F403
    from .._base_.datasets.cityscapes import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.schedule_80k import *  # noqa: F401,F403

from projects.hssn.losses import HieraTripletLossCityscape

custom_imports = dict(imports=[
    'projects.hssn.decode_head.sep_aspp_contrast_head',
    'projects.hssn.losses.hiera_triplet_loss_cityscape'
])

model.merge(dict(
    pretrained=None,
    backbone=dict(depth=101),
    decode_head=dict(
        num_classes=26,
        loss_decode=dict(
            type=HieraTripletLossCityscape, num_classes=19,
            loss_weight=1.0)),
    auxiliary_head=dict(num_classes=19),
    test_cfg=dict(mode='whole', is_hiera=True, hiera_num_classes=7)))
