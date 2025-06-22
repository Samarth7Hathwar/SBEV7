import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion

# import vis_features
from models.backbones.pts_backbone import PtsBackbone           
# from models.fuser.multimodal_feature_aggregation import MFAFuser 
from models.fuser.per_view_fuse import PerViewFuser

@DETECTORS.register_module()
class SparseBEV(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,     
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SparseBEV, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True

        self.memory = {}
        self.queue = queue.Queue()

        self.backbone_pts = PtsBackbone(pts_voxel_layer, pts_voxel_encoder,     
                            pts_middle_encoder, pts_backbone, pts_neck,)
        # self.fuser = MFAFuser(**pts_fusion_layer)
        self.fuser = PerViewFuser(**pts_fusion_layer)

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, 6, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, 6, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          ptss_context=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      augmented_radar=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        # vis_features.main(feature=img_feats[0][0], name = 'cam_feat_64x176_0', out_dir='images/cam_rad_sync_check/')
        '''
import matplotlib.pyplot as plt

cam_feat_vis = img_feats[0][0][0].mean(dim=0).detach().cpu().numpy()        #img_feats[scale][0][CAM]
radar_feat_vis = ptss_context[0][0][0].mean(dim=0).detach().cpu().numpy()   #ptss_context[scale][CAM][0]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Camera Features (32x88)")
plt.imshow(cam_feat_vis, cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Radar Features (32x88)")
plt.imshow(radar_feat_vis, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.savefig("images/cam_rad_sync_check/cam_rad_feat_64_0.png")
plt.close()
        '''

        # radar
        if len(augmented_radar[0].shape) == 4:
            augmented_radar[0] = augmented_radar[0].unsqueeze(1)
        # ptss_context, ptss_occupancy, _ = self.backbone_pts(augmented_radar[0])   #DARC
        ptss_context, _ = self.backbone_pts(augmented_radar[0])     #take output of neck
        img_feats = self.extract_feat(img, img_metas)   #sbev

        # camera features + radar features
        # for feat_index in range(0,len(img_feats)):
        #     radar_feats = ptss_context[feat_index].permute(1, 0, 2, 3, 4)
        #     radar_feats = self.reshape_samples(radar_feats)
        #     curr_cam_frame = (img_feats[feat_index][:,:6,:,:,:])

        #     '''
        #     ###temp vis code start
        #     import torch
        #     from torchvision.transforms.functional import to_pil_image

        #     img1 = img[0][0].cpu().detach()  # move to CPU if it's on GPU
        #     img_pil = to_pil_image(img1)  # Converts tensor to PIL image
        #     img_pil.save(f'images/cam_rad_sync_check/rgb_image_{feat_index}.png')

        #     vis_features.main(feature=radar_feats[0][0], name = f'radar_feat_{feat_index}', out_dir='images/cam_rad_sync_check/')
        #     vis_features.main(feature=curr_cam_frame[0][0], name = f'cam_feat_{feat_index}', out_dir='images/cam_rad_sync_check/')
        #     ###temp vis code end
        #     '''

        #     curr_frame_feats = torch.cat((curr_cam_frame,radar_feats),dim=2)
        #     B, N, C, H, W = curr_frame_feats.shape
        #     curr_frame_feats = curr_frame_feats.view(B*N,C,H,W).unsqueeze(1)
        #     fused_curr_feats = (self.fuser(curr_frame_feats, feat_index))
        #     C = fused_curr_feats[0].shape[1]
        #     fused_curr_feats = list(fused_curr_feats)
        #     fused_curr_feats[0] = fused_curr_feats[0].view(B,N,C,H,W)
        #     img_feats[feat_index][:,:6,:,:,:] = fused_curr_feats[0]
        radar_feats=[]
        curr_cam_feats=[]
        for feat_index in range(0,len(ptss_context)):
            radar_feat = ptss_context[feat_index].permute(1, 0, 2, 3, 4)
            #radar_feats.append(self.reshape_samples(radar_feat))

            # Sam : I am adding here
            radar_feat_reshaped = self.reshape_samples(radar_feat)
            # Flipping Vertically
            #radar_feat_reshaped = torch.flip(radar_feat_reshaped, dims=[-2])
            radar_feats.append(radar_feat_reshaped)
            cam_feat = img_feats[feat_index][:, :6, :, :, :]
            curr_cam_feats.append(cam_feat)
            #print(f"Level {feat_index} - Radar Feat Shape: {radar_feat_reshaped.shape}")  # Should be [B, 6, C, H, W]
            #print(f"Level {feat_index} - Camera Feat Shape: {cam_feat.shape}")            # Should be [B, 6, C, H, W]
            # Sam : Ending here

            #radar_feats.append(cam_feat)
            #curr_cam_feats.append(img_feats[feat_index][:,:6,:,:,:])

        fused_feats = self.fuser(curr_cam_feats, radar_feats)

        for feat_index in range(0,len(fused_feats)):
            img_feats[feat_index][:,:6,:,:,:] = fused_feats[feat_index]

        for i in range(len(img_metas)):     #sbev
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i]
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]

        losses = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        return losses

    def reshape_samples(self, radar_samples, samples_per_batch=6):
        if radar_samples.shape[1] != samples_per_batch:
            _, N, C, H, W = radar_samples.shape 
            batch_size = N // samples_per_batch
            return radar_samples.view(batch_size, samples_per_batch, C, H, W)
        else:
            return radar_samples

    def forward_test(self, img_metas, img=None, augmented_radar=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], augmented_radar, **kwargs)    #modded

    def simple_test_pts(self, x, img_metas, rescale=False):
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas[0], rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
    
    def simple_test(self, img_metas, img=None, augmented_radar=None, rescale=False):
        world_size = get_dist_info()[1]
        # if world_size == 1:  # online
        return self.simple_test_online(img_metas, img, augmented_radar, rescale)  
        # else:  # offline
        # return self.simple_test_offline(img_metas, img, augmented_radar, rescale)

    def simple_test_offline(self, img_metas, img=None, augmented_radar=None, rescale=False):
        # radar
        if len(augmented_radar[0][0].shape) == 4:
            augmented_radar[0][0] = augmented_radar[0][0].unsqueeze(1)
        ptss_context, ptss_occupancy, _ = self.backbone_pts(augmented_radar[0][0])

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # camera features + radar features
        radar_feats=[]
        curr_cam_feats=[]
        for feat_index in range(0,len(ptss_context)):
            radar_feat = ptss_context[feat_index].permute(1, 0, 2, 3, 4)
            radar_feats.append(self.reshape_samples(radar_feat))
            curr_cam_feats.append(img_feats[feat_index][:,:6,:,:,:])

        fused_feats = self.fuser(curr_cam_feats, radar_feats)

        for feat_index in range(0,len(fused_feats)):
            img_feats[feat_index][:,:6,:,:,:] = fused_feats[feat_index]
        

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def simple_test_online(self, img_metas, img=None, augmented_radar=None, rescale=False):

        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        img = img.reshape(B, N//6, 6, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // 6
        # assert num_frames == img.shape[1]

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        # extract feature frame by frame
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if img_filenames[img_indices[0]] in self.memory:
                # found in memory
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                # extract feature and put into memory
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() >= 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        # reorganize
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        # radar
        if len(augmented_radar[0][0].shape) == 4:
            augmented_radar[0][0] = augmented_radar[0][0].unsqueeze(1)
        # ptss_context, ptss_occupancy, _ = self.backbone_pts(augmented_radar[0][0])    #DARC
        ptss_context, _ = self.backbone_pts(augmented_radar[0][0])         #take output of neck

        radar_feats=[]
        curr_cam_feats=[]
        for feat_index in range(0,len(ptss_context)):
            radar_feat = ptss_context[feat_index].permute(1, 0, 2, 3, 4)
            radar_feats.append(self.reshape_samples(radar_feat))
            curr_cam_feats.append(img_feats[feat_index][:,:6,:,:,:])

        fused_feats = self.fuser(curr_cam_feats, radar_feats)


        # SAM add here start
        # ===== Debugging: Visualize Camera vs Radar Features After Fusion =====
        import matplotlib.pyplot as plt
        import os

        # Create output directory if not exists
        os.makedirs("/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/images/experiment1", exist_ok=True)
        # Choose which level (scale) and view (camera index) to visualize
        level_to_plot = 1  # You can change this to 0,1,2,3
        view_idx = 4       # Change between 0–5 depending on which camera

        # Take camera feature (before fusion)
        cam_feat = curr_cam_feats[level_to_plot][0][view_idx].mean(dim=0).cpu().detach().numpy()

        # Take radar feature (reshaped, before fusion)
        radar_feat = radar_feats[level_to_plot][0][view_idx].mean(dim=0).cpu().detach().numpy()

        # Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Camera View {view_idx} - Level {level_to_plot}')
        plt.imshow(cam_feat, cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title(f'Radar View {view_idx} - Level {level_to_plot}')
        plt.imshow(radar_feat, cmap='viridis')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/images/experiment1/cam_radar_feat_L{level_to_plot}_V{view_idx}.png')
        plt.close()

        # SAM Add ends here


        for feat_index in range(0,len(fused_feats)):
            img_feats[feat_index][:,:6,:,:,:] = fused_feats[feat_index]

        # run detector
        bbox_list = [dict() for _ in range(1)]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list
