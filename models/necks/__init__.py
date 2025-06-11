# # from .vovnet import VoVNet
# # from .eva02 import EVA02

# # __all__ = ['VoVNet', 'EVA02']

# from mmdet.models.necks.fpn import FPN

from .second_fpn import SECONDFPN_Custom

__all__ = ['SECONDFPN_Custom']      #'SECONDFPN'