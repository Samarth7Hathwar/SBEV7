dataset_type = 'CustomNuScenesDataset'
dataset_root = 'data/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,    #False originally
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

# arch config
embed_dims = 256
num_layers = 6
num_query = 900
num_frames = 8
num_levels = 4
num_points = 4

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=False)   #true sbev
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

#DARC radar config
pts_voxel_layer = dict(         #op coors, etc
    max_num_points=8,
    voxel_size=[8, 0.4, 2],     #DARC[8, 0.4, 2],    #SBEV[8, 0.875, 2]
    point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
    max_voxels=(768, 1024)
)
pts_voxel_encoder = dict(           #op voxel feats
    type='PillarFeatureNet',
    in_channels=5,
    feat_channels=[32, 64],
    with_distance=False,
    with_cluster_center=False,
    with_voxel_center=True,
    voxel_size=[8, 0.4, 2],        #DARC[8, 0.4, 2],    #SBEV[8, 0.875, 2]
    point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
    norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
    legacy=True
)
pts_middle_encoder = dict(
    type='PointPillarsScatter',
    in_channels=64,
    output_shape=(140,88)      #DARC(140, 88)  #desired(64, 176)
)
pts_backbone = dict(
    type='SECOND',
    in_channels=64,
    out_channels=[64, 128, 256 ,512],
    layer_nums=[3, 5, 5, 5],   
    layer_strides=[1, 2, 2, 2],
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect')
)
pts_neck = dict(
    type='SECONDFPN_Custom',
    in_channels=[64, 128, 256, 512],
    out_channels=[256, 256, 256, 256],
    upsample_strides=[1, 1, 1, 1],
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
    upsample_cfg=dict(type='deconv', bias=False),
    use_conv_for_no_stride=True,
    target_sizes = [(64, 176),(32,  88),(16,  44),(8,  22)]

)
occupancy_init = 0.01
out_channels_pts = 80

#DARC fuser conf
# fuser_conf = dict(
#         type = 'MFAFuser',
#         num_sweeps=1,
#         img_dims = 256,
#         pts_dims = 256,
#         embed_dims = 256,
#         num_layers = 6,
#         num_heads = 4,
#         # bev_shape = (128, 128)    #CRN
#         bev_shape = [(64, 176), (32, 88), (16, 44), (8, 22)]
#         )

# per view fuser
fuser_conf = dict(
    type='PerViewFuser',
    in_channels=512,     # camera (256) + radar (256)
    out_channels=256,
    num_cams=6,
    bev_shapes=[(64, 176), (32, 88), (16, 44), (8, 22)],
    conv_cfg=dict(
        type='Conv2d',
        kernel_size=1,
        norm_cfg=dict(type='BN2d'),
        activation='ReLU'
        )
    )
            

model = dict(
    type='SparseBEV',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    stop_prev_grad=1,
    pts_voxel_layer=pts_voxel_layer,        #DARC
    pts_voxel_encoder=pts_voxel_encoder,    #DARC
    pts_middle_encoder=pts_middle_encoder,  #DARC
    pts_fusion_layer=fuser_conf,              #DARC
    img_backbone=img_backbone,
    pts_backbone=pts_backbone,              #DARC
    img_neck=img_neck,
    pts_neck=pts_neck,
    pts_bbox_head=dict(
        type='SparseBEVHead',
        num_classes=10,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=True,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        transformer=dict(
            type='SparseBEVTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    ))
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = {      #added for radar from DARC
            'rot_ratio': 1.0,
            'rot_lim': (-22.5, 22.5),
            'scale_lim': (0.9, 1.1),
            'flip_dx_ratio': 0.5,
            'flip_dy_ratio': 0.5
        }

rda_aug_conf = {                     #added for radar from DARC
    'N_sweeps': 6,                   #DARC 6
    'N_use': 5,                      #DARC 5
    'drop_ratio': 0.1,               #randomly drop radar points
    'max_distance_pv': 58.0,         #d_bound[1] of DARC added
    'max_radar_points_pv': 1536,     #was hardcoded in DARC added
    'remove_z_axis': True            #from DARC added
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, bda_aug_conf=bda_aug_conf, rda_aug_conf=rda_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'augmented_radar'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames - 1, test_mode=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, bda_aug_conf=bda_aug_conf, rda_aug_conf=rda_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'augmented_radar'], meta_keys=(
                'filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape',
                'lidar2img', 'img_timestamp'))
        ])
]

data = dict(
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep_downloaded.pkl',   #PKL of sparsebev
        # ann_file=dataset_root + 'nuscenes_infos_train_sweep_dict.pkl',    #PKL generated by KP
        # ann_file=dataset_root + 'nuscenes_infos_train_dict_3.pkl',    #PKL generated by KP with metadata
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep_downloaded.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)
total_epochs = 24   #24
batch_size = 1

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=total_epochs)

# other flags
debug = True
