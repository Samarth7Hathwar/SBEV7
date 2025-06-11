import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import DETECTORS

@DETECTORS.register_module()
class PerViewFuser(nn.Module):
    def __init__(self,
                type = 'PerViewFuser',
                in_channels=512,  # cam (256) + radar (256)
                out_channels=256,
                num_cams=6,
                bev_shapes=[(64, 176), (32, 88), (16, 44), (8, 22)],
                conv_cfg=dict(
                    type='Conv2d',
                    kernel_size=1,
                    norm_cfg=dict(type='BN2d'),
                    activation='ReLU'
                )):
        super(PerViewFuser, self).__init__()

        self.num_cams = num_cams
        self.bev_shapes = bev_shapes

        # Create a conv layer for each resolution
        self.fusion_convs = nn.ModuleList()
        for (h, w) in bev_shapes:
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=conv_cfg['kernel_size']),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.fusion_convs.append(conv)

    def forward(self, cam_feats_list, radar_feats_list):
        """
        Args:
            cam_feats_list: list of tensors, each [B, N, C, H, W] for each resolution
            radar_feats_list: list of tensors, same shape as cam_feats_list

        Returns:
            fused_feats_list: list of tensors [B, N, C, H, W] (C=out_channels)
        """
        fused_feats_list = []

        for i, (cam_feat, radar_feat) in enumerate(zip(cam_feats_list, radar_feats_list)):
            B, N, C, H, W = cam_feat.shape

            # Concatenate features along channel dim: [B, N, 2C, H, W]
            fused = torch.cat([cam_feat, radar_feat], dim=2)

            # Merge cam dim into batch for fusion: [B*N, 2C, H, W]
            fused = fused.view(B * N, -1, H, W)
            fused = self.fusion_convs[i](fused)

            # Reshape back to [B, N, C, H, W]
            fused = fused.view(B, N, -1, H, W)
            fused_feats_list.append(fused)

        return fused_feats_list
