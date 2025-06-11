import mmcv
import torch
import numpy as np
from PIL import Image
from numpy import random
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        if self.size_divisor is not None:
            pad_h = int(np.ceil(img.shape[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(img.shape[1] / self.size_divisor)) * self.size_divisor
        else:
            pad_h, pad_w = self.size

        pad_width = ((0, pad_h - img.shape[0]), (0, pad_w - img.shape[1]), (0, 0))
        img = np.pad(img, pad_width, constant_values=self.pad_val)
        return img

    def _pad_imgs(self, results):
        padded_img = [self._pad_img(img) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_imgs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1)
        self.std = 1 / np.array(std, dtype=np.float32).reshape(-1)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        normalized_imgs = []

        for img in results['img']:
            img = img.astype(np.float32)
            if self.to_rgb:
                img = img[..., ::-1]
            img = img - self.mean
            img = img * self.std
            normalized_imgs.append(img)

        results['img'] = normalized_imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean,
            std=self.std,
            to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            ori_dtype = img.dtype
            img = img.astype(np.float32)

            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]

            new_imgs.append(img.astype(ori_dtype))

        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class RandomTransformImage(object):
    def __init__(self, ida_aug_conf=None, bda_aug_conf=None, rda_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.rda_aug_conf = rda_aug_conf
        self.max_distance_pv = rda_aug_conf['max_distance_pv']
        self.max_radar_points_pv = rda_aug_conf['max_radar_points_pv']
        self.remove_z_axis = rda_aug_conf['remove_z_axis']
        self.training = training

    def __call__(self, results):
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        radar_idx = self.sample_radar_augmentation()    #from DARC
        
        sweep_radar_points = list()
        radar_points = list()
        for radar_point in results['raw_radar']:    #augmenting raw radar points
            radar_point_augmented = self.transform_radar_pv(
                        radar_point, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate, radar_idx)  #assuming rotate is same as rotate_ida(DARC) - todo check
            radar_points.append(radar_point_augmented)
        sweep_radar_points.append(torch.stack(radar_points))
        results['augmented_radar'] = sweep_radar_points

        if len(results['lidar2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        elif len(results['img']) == 6:
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)

            for i in range(len(results['lidar2img'])):
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        else:
            raise ValueError()

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in results['img']]

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """
        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat.numpy()

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def sample_radar_augmentation(self):    #taken from DARC 
        """Generate bda augmentation values based on bda_config."""
        if self.training:
            radar_idx = np.random.choice(self.rda_aug_conf['N_sweeps'],
                                         self.rda_aug_conf['N_use'],
                                         replace=False)
        else:
            radar_idx = np.arange(self.rda_aug_conf['N_sweeps'])
        return radar_idx

    def transform_radar_pv(self, points, resize, resize_dims, crop, flip, rotate, radar_idx):   #from DARC
        points = points[points[:, 2] < self.max_distance_pv, :]

        H, W = resize_dims
        points[:, :2] = points[:, :2] * resize
        points[:, 0] -= crop[0]
        points[:, 1] -= crop[1]
        if flip:
            points[:, 0] = resize_dims[1] - points[:, 0]

        points[:, 0] -= W / 2.0
        points[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        points[:, :2] = np.matmul(rot_matrix, points[:, :2].T).T

        points[:, 0] += W / 2.0
        points[:, 1] += H / 2.0

        depth_coords = points[:, :2].astype(np.int16)

        valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                      & (depth_coords[:, 0] < resize_dims[1])
                      & (depth_coords[:, 1] >= 0)
                      & (depth_coords[:, 0] >= 0))

        points = torch.Tensor(points[valid_mask])

        if self.remove_z_axis:
            points[:, 1] = 1.  # dummy height value

        points_save = []
        for i in radar_idx:
            points_save.append(points[points[:, 6] == i])
        points = torch.cat(points_save, dim=0)

        # mean, std of rcs and speed are from train set
        points[:, 3] = (points[:, 3] - 4.783) / 7.576
        points[:, 4] = (torch.norm(points[:, 4:6], dim=1) - 0.677) / 1.976

        if self.training:
            drop_idx = np.random.uniform(size=points.shape[0])  # randomly drop points
            points = points[drop_idx > self.rda_aug_conf['drop_ratio']]

        num_points, num_feat = points.shape
        if num_points > self.max_radar_points_pv:
            choices = np.random.choice(num_points, self.max_radar_points_pv, replace=False)
            points = points[choices]
        else:
            num_append = self.max_radar_points_pv - num_points
            points = torch.cat([points, -999*torch.ones(num_append, num_feat)], dim=0)

        if num_points == 0:
            points[0, :] = points.new_tensor([0.1, 0.1, self.max_distance_pv-1, 0, 0, 0, 0])

        points[..., [0, 1, 2]] = points[..., [0, 2, 1]]  # convert [w, h, d] to [w, d, h]

        return points[..., :5]


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    def __init__(self,
                 rot_range=[-0.3925, 0.3925],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0]):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def __call__(self, results):
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)
        self.rotate_z(results, rot_angle)
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_z(self, results, rot_angle):
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_sin = torch.sin(torch.tensor(rot_angle))

        rot_mat = torch.tensor([
            [rot_cos, -rot_sin, 0, 0],
            [rot_sin, rot_cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        rot_mat_inv = torch.inverse(rot_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ rot_mat_inv).numpy()

    def scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])
        scale_mat_inv = torch.inverse(scale_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ scale_mat_inv).numpy()
