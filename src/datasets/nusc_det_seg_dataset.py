import os

import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

__all__ = ['NuscDetSegDataset']

map_name_from_general_to_det_seg = {
    'animal': 'ignore',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.wheelchair': 'ignore',
    'movable_object.barrier': 'barrier',
    'movable_object.debris': 'ignore',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.trafficcone': 'traffic_cone',
    'static_object.bicycle_rack': 'ignore',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.other': 'ignore',
    'static.vegetation': 'vegetation',
    'vehicle.ego': 'ignore',
}

map_idx_from_general_to_seg = {
    0: 0,
    1: 0,
    2: 7,
    3: 7,
    4: 7,
    5: 0,
    6: 7,
    7: 0,
    8: 0,
    9: 1,
    10: 0,
    11: 0,
    12: 8,
    13: 0,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    19: 0,
    20: 0,
    21: 6,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 0,
    30: 16,
    31: 0,
}

label_16_names = ['noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                  'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
                  'manmade', 'vegetation']

# def plot_image_for_debug(ret_list):
#     imgs = ret_list[0][0].permute(0, 2, 3, 1).numpy()
#     img_depths = ret_list[10][0].numpy()
#     img_segs = ret_list[11][0].numpy()
#     bev_seg = imgviz.label2rgb(ret_list[12][0, 0].numpy(), label_names=label_16_names)
#     fig, axes = plt.subplots(3, 6, figsize=(24, 16))
#     for idx, ax in enumerate(axes.flatten()[:6]):
#         ax.imshow(imgs[idx])
#     for idx, ax in enumerate(axes.flatten()[6:12]):
#         ax.imshow(img_depths[idx])
#     for idx, ax in enumerate(axes.flatten()[12:]):
#         ax.imshow(imgviz.label2rgb(img_segs[idx],label_names=label_16_names))
#     axes.flatten()[-1].axis('off')
#     plt.tight_layout()
#     fig.subplots_adjust(wspace=0, hspace=0)
#     plt.show()
#     plt.close()
#     plt.imshow(bev_seg)
#     # plt.imshow(ret_list[12].numpy())
#     plt.show()
#     return None

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, resize, resize_dims, crop, flip, rotate):
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
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_transform(gt_boxes, inrange_ego_pts, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    if inrange_ego_pts.shape[0] > 0:
        inrange_ego_pts = (rot_mat @ inrange_ego_pts.unsqueeze(-1)).squeeze(-1)
    return gt_boxes, inrange_ego_pts, rot_mat


def depth_transform(cam_depth, cam_label, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2]#.astype(np.int16)

    depth_map = np.zeros(resize_dims)
    label_map = np.zeros(resize_dims, dtype=np.uint8)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_coords = depth_coords[valid_mask, :]
    cam_depth = cam_depth[valid_mask, 2][:, None]
    cam_label = cam_label[valid_mask, None]
    depth_coords_label_pair = np.concatenate([cam_depth, depth_coords, cam_label], axis=1)
    depth_coords_label_pair = depth_coords_label_pair[np.argsort(-cam_depth[:, 0]), ...]
    cam_depth = depth_coords_label_pair[:, 0]
    depth_coords = depth_coords_label_pair[:, 1:3].astype(np.int16)
    cam_label = depth_coords_label_pair[:, -1]
    depth_map[depth_coords[:, 1], depth_coords[:, 0]] = cam_depth
    label_map[depth_coords[:, 1], depth_coords[:, 0]] = cam_label
    return torch.Tensor(depth_map), torch.tensor(label_map)

def get_bev_seg_map(
    bev_points,
    bev_labels,
    x_bound=(-51.2, 51.2),
    y_bound=(-51.2, 51.2),
    z_bound=(-5., 3.),
    size=0.8,
):
    bev_map = np.zeros((int((x_bound[1]-x_bound[0]) / size), int((y_bound[1]-y_bound[0]) / size)), dtype=np.uint8)
    bev_height = np.zeros((int((x_bound[1]-x_bound[0]) / size), int((y_bound[1]-y_bound[0]) / size)), dtype=np.float32)
    bev_mask = np.zeros((int((x_bound[1]-x_bound[0]) / size), int((y_bound[1]-y_bound[0]) / size)), dtype=np.uint8)
    voxel_coord = (x_bound[0] - size / 2.0, y_bound[0] - size / 2.0)
    coords = ((bev_points[:, :2].numpy() - voxel_coord) / size)#.astype(np.int16)
    heights = bev_points[:, 2].numpy()
    mask = np.ones(coords.shape[0], dtype=bool)
    mask = np.logical_and(mask, coords[:, 0] > 1)
    mask = np.logical_and(mask, coords[:, 0] < bev_map.shape[0] - 1)
    mask = np.logical_and(mask, coords[:, 1] > 1)
    mask = np.logical_and(mask, coords[:, 1] < bev_map.shape[1] - 1)
    mask = np.logical_and(mask, heights > z_bound[0])
    mask = np.logical_and(mask, heights < z_bound[1])
    coords = coords[mask, :]
    heights = heights[mask, None]
    bev_labels = bev_labels[mask, None].numpy()
    height_coords_label_pair = np.concatenate([heights, coords, bev_labels], axis=1)
    height_coords_label_pair = height_coords_label_pair[np.argsort(heights[:, 0]), ...]
    heights = height_coords_label_pair[:, 0]
    coords = height_coords_label_pair[:, 1:3].astype(np.int16)
    bev_labels = height_coords_label_pair[:, -1]
    bev_map[coords[:, 1], coords[:, 0]] = bev_labels
    bev_height[coords[:, 1], coords[:, 0]] = heights
    bev_mask[coords[:, 1], coords[:, 0]] = 1.
    return torch.tensor(bev_map), torch.tensor(bev_height), torch.BoolTensor(bev_mask)

def map_pointcloud_to_bev(
    lidar_points,
    points_label,
    lidar_info,
    x_bound=(-51.2, 51.2),
    y_bound=(-51.2, 51.2),
    z_bound=(-5., 3.),
    flat=False,
):
    pc = LidarPointCloud(lidar_points.T)
    cs_record = lidar_info['LIDAR_TOP']['calibrated_sensor']
    pose_record = lidar_info['LIDAR_TOP']['ego_pose']
    ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                  rotation=Quaternion(cs_record["rotation"]))

    if flat:
        # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
        ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        rotation_vehicle_flat_from_vehicle = np.dot(
            Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
            Quaternion(pose_record['rotation']).inverse.rotation_matrix)
        vehicle_flat_from_vehicle = np.eye(4)
        vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
        viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
    else:
        viewpoint = ref_to_ego

    points = view_points(pc.points[:3, :], viewpoint, normalize=False)
    ref_index = np.arange(points.shape[1])
    ref_labels = points_label.copy()
    # x = points[0, :]
    # y = points[1, :]
    # z = points[2, :]
    # mask = np.ones(x.shape[0], dtype=bool)
    # mask = np.logical_and(mask, x > x_bound[0])
    # mask = np.logical_and(mask, x < x_bound[1])
    # mask = np.logical_and(mask, y > y_bound[0])
    # mask = np.logical_and(mask, y < y_bound[1])
    # mask = np.logical_and(mask, z > z_bound[0])
    # mask = np.logical_and(mask, z < z_bound[1])
    # points = points[:, mask]
    # points_label = points_label[mask]
    # ref_index = ref_index[mask]
    return points, points_label, ref_labels, ref_index

def map_pointcloud_to_image(
    lidar_points,
    points_label,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    depth = depths[mask]
    label = points_label[mask]
    return points, depth, label


class NuscDetSegDataset(Dataset):

    def __init__(self,
                 ida_aug_conf,
                 bda_aug_conf,
                 classes,
                 data_root,
                 info_paths,
                 mode='train',
                 use_dense=False,
                 use_cbgs=False,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 sweep_idxes=list(),
                 key_idxes=list()):
        """Dataset used for bevdetection task.
        Args:
            ida_aug_conf (dict): Config for ida augmentation.
            bda_aug_conf (dict): Config for bda augmentation.
            classes (list): Class names.
            mode (str): train, val or test.
            use_cbgs (bool): Whether to use cbgs strategy,
                Default: False.
            num_sweeps (int): Number of sweeps to be used for each sample.
                default: 1.
            img_conf (dict): Config for image.
            sweep_idxes (list): List of sweep idxes to be used.
                default: list().
            key_idxes (list): List of key idxes to be used.
                default: list().
        """
        super().__init__()
        if isinstance(info_paths, list):
            self.infos = list()
            for info_path in info_paths:
                self.infos.extend(mmcv.load(info_path))
        else:
            self.infos = mmcv.load(info_paths)
        self.mode = mode
        self.use_dense = use_dense
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.data_root = data_root
        self.classes = classes
        self.use_cbgs = use_cbgs
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.num_sweeps = num_sweeps
        self.img_mean = np.array(img_conf['img_mean'], np.float32)
        self.img_std = np.array(img_conf['img_std'], np.float32)
        self.to_rgb = img_conf['to_rgb']
        assert sum([sweep_idx >= 0 for sweep_idx in sweep_idxes]) \
            == len(sweep_idxes), 'All `sweep_idxes` must greater \
                than or equal to 0.'

        self.sweeps_idx = sweep_idxes
        assert sum([key_idx < 0 for key_idx in key_idxes]) == len(key_idxes),\
            'All `key_idxes` must less than 0.'
        self.key_idxes = [0] + key_idxes

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set(
                [ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_det_seg[gt_name]
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.mode == 'train':
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.mode == 'train':
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def get_lidar_depth_label(self, lidar_points, points_label, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
        lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']

        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        pts_img, depth, label = map_pointcloud_to_image(
            lidar_points.copy(), points_label.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        return np.concatenate([pts_img[:2, :].T, depth[:, None]], axis=1).astype(np.float32), label

    def get_lidar_bev_label(self, lidar_points, points_label, lidar_info):
        pts_bev, bev_label, ref_labels, ref_index = map_pointcloud_to_bev(lidar_points, points_label, lidar_info)
        return pts_bev[:3, :].T, bev_label, ref_labels, ref_index

    def get_lidar(self, lidar_infos, dense_lidar=False):
        sweep_lidar_points = list()
        for lidar_info in lidar_infos:
            lidar_path = lidar_info['LIDAR_TOP']['filename']
            lidarseg_path = lidar_info['LIDAR_TOP']['lidarseg_labels_filename']
            if dense_lidar:
                # dense lidar path
                dense_path = os.path.join(self.data_root, lidar_path).replace('samples/LIDAR_TOP', 'occupancy')
                lidar_points_label = np.fromfile(dense_path, dtype=np.float16, count=-1).reshape(-1, 5)
                lidar_points = lidar_points_label[..., :4].astype(np.float32)
                points_label = lidar_points_label[..., 4].astype(np.int8)
                # lidar_points = lidar_points.asytype(np.float32)
                points_label = np.vectorize(map_idx_from_general_to_seg.__getitem__)(points_label)
                points_label = points_label.astype(np.int8)
            else:
                lidar_points = np.fromfile(os.path.join(
                    self.data_root, lidar_path),
                                           dtype=np.float32,
                                           count=-1).reshape(-1, 5)[..., :4]
                if lidarseg_path is not None:
                    points_label = np.fromfile(os.path.join(
                        self.data_root, lidarseg_path), dtype=np.uint8)
                    # map to label-16
                    points_label = np.vectorize(map_idx_from_general_to_seg.__getitem__)(points_label)
                    points_label = points_label.astype(np.int8)
                else:
                    points_label = np.zeros_like(lidar_points[:, 0], dtype=np.uint8)
            sweep_lidar_points.append((lidar_points, points_label))
        return sweep_lidar_points

    def get_image(self, cam_infos, cams, lidar_infos=None, sweep_lidar_points=None):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_depth = list()
        sweep_lidar_label = list()
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            lidar_label = list()
            key_info = cam_infos[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
            for sweep_idx, cam_info in enumerate(cam_infos):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
                # if self.return_depth and (self.use_fusion or sweep_idx == 0):
                if self.mode == 'train':
                    point_depth, point_label = self.get_lidar_depth_label(
                        sweep_lidar_points[sweep_idx][0], sweep_lidar_points[sweep_idx][1], img,
                        lidar_infos[sweep_idx], cam_info[cam])
                    point_depth_augmented, point_label_augmented = depth_transform(
                        point_depth, point_label, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)
                    lidar_label.append(point_label_augmented)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.mode == 'train':
                sweep_lidar_depth.append(torch.stack(lidar_depth))
                sweep_lidar_label.append(torch.stack(lidar_label))
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        # point_bev, bev_label, ref_labels, ref_index = self.get_lidar_bev_label(sweep_lidar_points[0][0],
        #                                                                        sweep_lidar_points[0][1],
        #                                                                        lidar_infos[0])
        # if self.return_depth:
        # get bev lidar segmentation label
        if self.mode == 'train':
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
            ret_list.append(torch.stack(sweep_lidar_label).permute(1, 0, 2, 3))
            # ret_list.append(torch.Tensor(point_bev))
            # ret_list.append(torch.tensor(bev_label))
        # ret_list.append(torch.Tensor(ref_pts_bev))
        # ret_list.append(torch.tensor(ref_labels_bev))
        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_det_seg[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_det_seg[
                    ann_info['category_name']]))
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.mode == 'train' and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        # idx = 1234
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        lidar_infos = list()
        occ_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            lidar_infos.append(info['lidar_infos'])
            if self.mode != 'test':
                occ_infos.append(info['occ_infos'])
            lidar_sweep_timestamps = [
                lidar_sweep['LIDAR_TOP']['timestamp']
                for lidar_sweep in info['lidar_sweeps']
            ]
            for sweep_idx in self.sweeps_idx:
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                    lidar_infos.append(info['lidar_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx),
                                   -1, -1):
                        if sum([cam in info['cam_sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            cam_timestamp = np.mean([
                                val['timestamp']
                                for val in info['cam_sweeps'][i].values()
                            ])
                            # Find the closest lidar frame to the cam frame.
                            lidar_idx = np.abs(lidar_sweep_timestamps -
                                               cam_timestamp).argmin()
                            lidar_infos.append(info['lidar_sweeps'][lidar_idx])
                            break
        # lidar is needed for all mode (train-supervise, val-seg iou, test-seg iou)
        if self.use_dense and self.mode == 'train':
            # training stage can use dense point clouds for supervision
            try:
                sweep_lidar_points = self.get_lidar(lidar_infos, dense_lidar=True)
            except:
                sweep_lidar_points = self.get_lidar(lidar_infos, dense_lidar=False)
        else:
            # validation/test stages use sparse point clouds for query position of lidar segmentation task
            sweep_lidar_points = self.get_lidar(lidar_infos, dense_lidar=False)
        image_data_list = self.get_image(cam_infos, cams, lidar_infos, sweep_lidar_points)
        inrange_ego_pts, inrange_ego_labels, ref_labels, ref_index = self.get_lidar_bev_label(sweep_lidar_points[0][0],
                                                                                              sweep_lidar_points[0][1],
                                                                                              lidar_infos[0])
        inrange_ego_pts, inrange_ego_labels = torch.Tensor(inrange_ego_pts), torch.tensor(inrange_ego_labels)
        ref_labels, ref_index = torch.tensor(ref_labels), torch.tensor(ref_index)
        # ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_data_list[:7]
        img_metas['token'] = self.infos[idx]['sample_token']
        if self.mode == 'train':
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
            # bev_points, bev_labels = image_data_list[-4:-2]
        # Temporary solution for test.
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )
            # bev_points = sweep_imgs.new_zeros(0, 3)
            # bev_labels = sweep_imgs.new_zeros(0, )
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        # rotate_bda = 0.
        # scale_bda = 1.
        # flip_dx = True
        # flip_dy = True
        # _, axes = plt.subplots(1, 2, figsize=(18, 9))
        # axes[0].scatter(inrange_ego_pts[:, 0].numpy(), inrange_ego_pts[:, 1].numpy(), c='orange', s=0.2)
        gt_boxes, inrange_ego_pts, bda_rot = bev_transform(gt_boxes, inrange_ego_pts, rotate_bda, scale_bda, flip_dx,
                                                           flip_dy)
        # axes[1].scatter(inrange_ego_pts[:, 0].numpy(), inrange_ego_pts[:, 1].numpy(), c='red', s=0.2)
        # plt.show()
        # plt.close()
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ]
        if self.mode == 'train':
            # get bev
            bev_seg_map, bev_height, bev_mask = get_bev_seg_map(inrange_ego_pts, inrange_ego_labels, size=0.4)
            ret_list.append(image_data_list[7])
            ret_list.append(image_data_list[8].long())
            ret_list.append(bev_seg_map[None, None, ...].long())
            ret_list.append(bev_height[None, None, ...])
            ret_list.append(bev_mask[None, None, ...])
        ret_list.append(inrange_ego_pts)
        ret_list.append(inrange_ego_labels.long())
        ret_list.append(ref_labels.long())
        ret_list.append(ref_index)
        # lidar token for submit lidar seg results
        ret_list.append(lidar_infos[0]['LIDAR_TOP']['lidar_token'])
        # load occupancy challenge data
        if self.mode != 'test':
            occ_gt_path = os.path.join(self.data_root, occ_infos[0]['occ_gt_path'])
            occ_labels = np.load(occ_gt_path)
            occ_semantics = torch.tensor(occ_labels['semantics'])
            mask_lidar = torch.tensor(occ_labels['mask_lidar'])
            mask_camera = torch.tensor(occ_labels['mask_camera'])
            occ_density_labels = (occ_semantics != 17).float()
        else:
            occ_semantics = sweep_imgs.new_zeros(0, )
            occ_density_labels = sweep_imgs.new_zeros(0, )
            mask_lidar = sweep_imgs.new_zeros(0, )
            mask_camera = sweep_imgs.new_zeros(0, )
        ret_list.append(occ_semantics.long())
        ret_list.append(occ_density_labels.float())
        ret_list.append(mask_lidar.bool())
        ret_list.append(mask_camera.bool())
        # debug vis
        # plot_image_for_debug(ret_list)
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {self.mode}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)


def collate_fn(data, mode='train'):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    seg_labels_batch = list()
    bev_seg_batch = list()
    bev_height_batch = list()
    bev_mask_batch = list()
    inrange_pts_batch = list()
    inrange_labels_batch = list()
    ref_labels_batch = list()
    ref_index_batch = list()
    lidar_token_batch = list()
    occ_semantics_batch = list()
    occ_density_labels_batch = list()
    mask_lidar_batch = list()
    mask_camera_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        if mode == 'train':
            gt_depth, gt_seg, gt_bev_seg, gt_bev_height, gt_bev_mask = iter_data[10:15]
            depth_labels_batch.append(gt_depth)
            seg_labels_batch.append(gt_seg)
            bev_seg_batch.append(gt_bev_seg)
            bev_height_batch.append(gt_bev_height)
            bev_mask_batch.append(gt_bev_mask)
        inrange_pts, inrange_labels, ref_labels, ref_index, lidar_token, occ_semantics, occ_density_labels, mask_lidar, mask_camera = iter_data[-9:]
        inrange_pts_batch.append(inrange_pts)
        inrange_labels_batch.append(inrange_labels)
        ref_labels_batch.append(ref_labels)
        ref_index_batch.append(ref_index)
        lidar_token_batch.append(lidar_token)
        occ_semantics_batch.append(occ_semantics)
        occ_density_labels_batch.append(occ_density_labels)
        mask_lidar_batch.append(mask_lidar)
        mask_camera_batch.append(mask_camera)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    ret_list = [
        torch.stack(imgs_batch),
        mats_dict,
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
    ]
    if mode == 'train':
        ret_list.append(torch.stack(depth_labels_batch))
        ret_list.append(torch.stack(seg_labels_batch))
        ret_list.append(torch.stack(bev_seg_batch))
        ret_list.append(torch.stack(bev_height_batch))
        ret_list.append(torch.stack(bev_mask_batch))
    ret_list.append(inrange_pts_batch)
    ret_list.append(inrange_labels_batch)
    ret_list.append(ref_labels_batch)
    ret_list.append(ref_index_batch)
    ret_list.append(lidar_token_batch)
    ret_list.append(torch.stack(occ_semantics_batch))
    ret_list.append(torch.stack(occ_density_labels_batch))
    ret_list.append(torch.stack(mask_lidar_batch))
    ret_list.append(torch.stack(mask_camera_batch))
    return ret_list