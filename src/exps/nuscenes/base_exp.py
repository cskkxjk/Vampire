import os
from functools import partial
import matplotlib.pyplot as plt
import json
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.utils

from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from src.datasets.nusc_det_seg_dataset import NuscDetSegDataset as NuscDetDataset
from src.datasets.nusc_det_seg_dataset import collate_fn
from src.evaluators.det_evaluators import DetNuscEvaluator
from src.models.vampire2 import VAMPIRE2 as BaseModel
from src.utils.torch_dist import all_gather_object, get_rank, synchronize
from src.utils.vis_utils import visualize_depth, visualize_semantic
from src.utils.lovasz_losses import lovasz_softmax
from torchmetrics import JaccardIndex, MultiScaleStructuralSimilarityIndexMeasure
import math
import pickle
H = 900
W = 1600
final_dim = (256, 704)
resize_lim = (0.386, 0.55)
sample_factor = 4
# final_dim = (900, 1600)
# resize_lim = (1.0, 1.0)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

backbone_conf = {
    'x_bound_seg': [-51.2, 51.2, 0.4],
    'y_bound_seg': [-51.2, 51.2, 0.4],
    'z_bound_seg': [-5., 3., 0.4],
    'x_bound_det': [-51.2, 51.2, 0.4],
    'y_bound_det': [-51.2, 51.2, 0.4],
    'z_bound_det': [-1., 3., 0.4],
    'd_bound': [2.0, 70.4, 0.8],
    # 'resolution': 0.4,
    'final_dim':
        final_dim,
    'density_mode': 'sdf',
    'sdf_bias': -1.0,
    'cat_pos': True,
    'cat_seg': False,
    'mid_channels':
        16,
    'output_channels':
        # 32,
        80,
    'downsample_factor':
        sample_factor,
    'upsample_factor':
        sample_factor,
    'img_backbone_conf':
        dict(
            type='ResNet',
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            # out_indices=[0, 1, 2],
            norm_eval=False,
            # with_cp=True,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        ),
    'img_neck_conf':
        dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.5, 1, 2, 4],
            out_channels=[128, 128, 128, 128],

            # in_channels=[256, 512, 1024],
            # upsample_strides=[0.5, 1, 2],
            # out_channels=[32, 32, 32],

            # in_channels=[256, 512, 1024, 2048],
            # upsample_strides=[0.5, 1, 2, 4],
            # out_channels=[20, 20, 20, 20],
        ),

    'num_classes': 18,
}
ida_aug_conf = {
    'resize_lim': resize_lim,
    'final_dim':
        final_dim,
    'rot_lim': (0., 0.),  # (-5.4, 5.4),
    'H':
        H,
    'W':
        W,
    'rand_flip':
        False,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    # 'rot_lim': (-22.5, 22.5),
    'rot_lim': (0.0, 0.0),
    # 'scale_lim': (0.95, 1.05),
    'scale_lim': (1., 1.),
    'flip_dx_ratio': 0.,
    'flip_dy_ratio': 0.
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}
unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
label_16_names = ['noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                  'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
                  'manmade', 'vegetation']
label_17_names = ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                  'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
                  'manmade', 'vegetation', 'free']

class VAMPIRELightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 trainval: bool = False,
                 vis: bool = False,
                 debug: bool = False,
                 task_weights=[1., 1., 1.],
                 loss_weights=[1., 1., 1., 1., 1.],
                 data_root='data/nuScenes',
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 bda_aug_conf=bda_aug_conf,
                 unique_label = unique_label,
                 label_names = label_17_names,
                 default_root_dir='./outputs/',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.trainval = trainval
        self.vis = vis
        self.debug = debug
        self.task_weights = task_weights   # occ, lidarseg, detection
        self.loss_weights = loss_weights   # camera, bev, sdf
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 8  # 2e-4 / 64
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.num_seg_classes = backbone_conf['num_classes']
        self.sdf_bias = backbone_conf['sdf_bias']
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.detection_submit_dir = os.path.join(default_root_dir, 'detection_submit')
        mmcv.mkdir_or_exist(self.detection_submit_dir)
        self.lidar_seg_submit_dir = os.path.join(default_root_dir, 'lidarseg_submit')
        mmcv.mkdir_or_exist(self.lidar_seg_submit_dir)
        self.vis_dir = os.path.join(default_root_dir, 'visualization')
        mmcv.mkdir_or_exist(self.vis_dir)
        self.evaluator = DetNuscEvaluator(class_names=self.class_names,
                                          output_dir=self.detection_submit_dir)
        self.model = BaseModel(self.backbone_conf,
                               self.head_conf)
        # unique_label = np.asarray(unique_label)
        self.ignore_label = 0
        self.unique_label_str = [label_names[x] for x in unique_label][1:][:-1]
        self.occ_label_str = [label_names[x] for x in unique_label]
        # self.val_iou = IoU(unique_label, 0, unique_label_str)
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.train_iou = JaccardIndex(task='multiclass', num_classes=len(unique_label)-1, average='none', ignore_index=self.ignore_label)
        self.occ_train_iou = JaccardIndex(task='multiclass', num_classes=len(unique_label), average='none')
        self.val_iou = JaccardIndex(task='multiclass', num_classes=len(unique_label)-1, average='none', ignore_index=self.ignore_label)
        self.occ_val_iou = JaccardIndex(task='multiclass', num_classes=len(unique_label), average='none')
        self.best_miou = 0
        self.best_occ_miou = 0
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.upsample_factor = self.backbone_conf['upsample_factor']
        self.dbound = self.backbone_conf['d_bound']

        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.train_info_paths = os.path.join(self.data_root,
                                             'nuscenes_occ_infos_train.pkl')
        self.val_info_paths = os.path.join(self.data_root,
                                           'nuscenes_occ_infos_val.pkl')
        self.vis_info_paths = os.path.join(self.data_root,
                                           'nuscenes_occ_infos_scene_0012_0018.pkl')
        self.trainval_info_paths = os.path.join(self.data_root,
                                                'nuscenes_occ_infos_trainval.pkl')
        self.predict_info_paths = os.path.join(self.data_root,
                                               'nuscenes_infos_test.pkl')

    def forward(self, sweep_imgs, mats, inrange_pts=None, lidar_seg=False):
        return self.model(sweep_imgs, mats, inrange_pts, lidar_seg=lidar_seg)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels, seg_labels, bev_seg, bev_height, bev_mask,
         inrange_pts, inrange_labels, _, _, _, occ_semantics, occ_density_labels, mask_lidar, mask_camera) = batch
        if torch.cuda.is_available() and not self.debug:
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            depth_labels = depth_labels.cuda()
            seg_labels = seg_labels.cuda()
            bev_seg = bev_seg.cuda()
            bev_height = bev_height.cuda()
            bev_mask = bev_mask.cuda()
            inrange_pts = [inrange_pt.cuda() for inrange_pt in inrange_pts]
            inrange_labels = [inrange_label.cuda() for inrange_label in inrange_labels]
        preds, \
        rgb_preds, seg_logits_preds, depth_preds, \
        bev_rgb_preds, bev_seg_logits_preds, bev_height_preds, bev_density, \
        pts_logits_batch, pts_sdf_batch, occ_logits_batch, occ_density_batch = self(sweep_imgs, mats, inrange_pts=inrange_pts, lidar_seg=False)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)

        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            sweep_imgs = sweep_imgs[:, 0, ...]
            depth_labels = depth_labels[:, 0, ...]
            seg_labels = seg_labels[:, 0, ...]
        depth_preds = depth_preds[:, :, 0, ...]
        rgb_labels, depth_labels, seg_labels, fg_mask = self.get_downsampled_gt(sweep_imgs, depth_labels, seg_labels)
        # camera loss
        camera_depth_loss = self.get_depth_loss(depth_labels, depth_preds, fg_mask)
        rgb_loss = self.get_rgb_loss(rgb_labels, rgb_preds)
        camera_seg_loss = self.get_camera_seg_loss(seg_labels, seg_logits_preds, fg_mask)
        # camera_loss = rgb_loss + depth_loss + camera_seg_loss
        # bev loss
        bev_height_loss = self.get_height_loss_bev(bev_height, bev_height_preds, bev_mask)
        bev_seg_loss = self.get_bev_seg_loss(bev_seg, bev_seg_logits_preds, bev_mask)
        # bev_loss = bev_height_loss + bev_seg_loss
        depth_loss = camera_depth_loss + bev_height_loss
        seg_loss = camera_seg_loss + bev_seg_loss
        sdf_loss = 0
        density_loss = 0
        lidarseg_loss = 0
        if len(pts_logits_batch) != 0:
            pts_seg_loss = self.get_pts_seg_loss(inrange_labels, pts_logits_batch)
            self.log('pts_seg_loss', pts_seg_loss)
            lidarseg_loss += pts_seg_loss
            for pts_logits, inrange_label in zip(pts_logits_batch, inrange_labels):
                seg_result = pts_logits[..., 1:-1].argmax(1) + 1
                self.train_iou(seg_result, inrange_label)
        if len(pts_sdf_batch) != 0:
            sdf_loss += self.get_sdf_loss(pts_sdf_batch)
            self.log('sdf_loss', sdf_loss)
        # if use camera mask
        visible_occ_semantics = occ_semantics[mask_camera]
        visible_occ_logits_batch = occ_logits_batch[mask_camera]
        visible_occ_result = visible_occ_logits_batch.argmax(1)
        self.occ_train_iou(visible_occ_result, visible_occ_semantics)
        visible_occ_density_labels = occ_density_labels[mask_camera]
        visible_occ_density_batch = occ_density_batch[mask_camera]
        # invisible_occ_semantics = occ_semantics[~mask_camera]
        # invisible_occ_logits_batch = occ_logits_batch[~mask_camera]
        invisible_occ_density_labels = occ_density_labels[~mask_camera]
        invisible_occ_density_batch = occ_density_batch[~mask_camera]

        visible_occ_seg_loss = self.get_occ_seg_loss(visible_occ_semantics, visible_occ_logits_batch)
        self.log('visible_occ_seg_loss', visible_occ_seg_loss)
        # visible mask only
        occ_loss = visible_occ_seg_loss
        # invisible_occ_seg_loss = self.get_occ_seg_loss(invisible_occ_semantics, invisible_occ_logits_batch)
        # self.log('invisible_occ_seg_loss', invisible_occ_seg_loss)
        # occ_loss = visible_occ_seg_loss + invisible_occ_seg_loss

        visible_occ_density_loss = self.get_occ_density_loss(visible_occ_density_labels, visible_occ_density_batch)
        self.log('visible_occ_density_loss', visible_occ_density_loss)
        invisible_occ_density_loss = self.get_occ_density_loss(invisible_occ_density_labels, invisible_occ_density_batch)
        density_loss += visible_occ_density_loss + invisible_occ_density_loss
        self.log('invisible_occ_density_loss', invisible_occ_density_loss)

        self.log('detection_loss', detection_loss)
        self.log('depth_loss', depth_loss)
        self.log('rgb_loss', rgb_loss)
        self.log('camera_seg_loss', camera_seg_loss)
        self.log('bev_height_loss', bev_height_loss)
        self.log('bev_seg_loss', bev_seg_loss)
        total_loss = self.task_weights[0] * occ_loss + \
                     self.task_weights[1] * lidarseg_loss + \
                     self.task_weights[2] * detection_loss + \
                     self.loss_weights[0] * depth_loss + \
                     self.loss_weights[1] * seg_loss + \
                     self.loss_weights[2] * rgb_loss + \
                     self.loss_weights[3] * sdf_loss + \
                     self.loss_weights[4] * density_loss

        # if self.global_step % 250 == 0:
        if self.global_step % 500 == 0:
            seg_preds = torch.max(seg_logits_preds.permute(0, 1, 3, 4, 2).softmax(-1), dim=-1)[1]
            bev_seg_preds = torch.max(bev_seg_logits_preds.permute(0, 2, 3, 1).softmax(-1), dim=-1)[1]
            self.log_images(rgb_labels[0].cpu().detach(),
                            seg_labels[0].cpu().detach(),
                            rgb_preds[0].cpu().detach(),
                            depth_preds[0].cpu().detach(),
                            seg_preds[0].cpu().detach(),
                            bev_seg[0].cpu().detach(),
                            bev_height[0].cpu().detach(),
                            bev_rgb_preds[0].cpu().detach(),
                            bev_height_preds[0].cpu().detach(),
                            bev_seg_preds[0].cpu().detach(),
                            bev_density[0].cpu().detach())
        # return depth_loss + rgb_loss + seg_loss + bev_seg_loss
        return total_loss

    def log_images(self, gt_rgbs, gt_segs, rgb_preds, depth_preds, seg_preds, bev_gt_seg, bev_gt_height, bev_rgb_preds,
                   bev_height_preds, bev_seg_preds, bev_density) -> None:
        num_cams, _, h, w = gt_rgbs.shape
        gt_rgb_cat = torch.cat([torch.cat([gt_rgb for gt_rgb in gt_rgbs[:3]], dim=2),
                                torch.cat([gt_rgb for gt_rgb in torch.flip(gt_rgbs[3:], dims=[0])], dim=2)], dim=1)
        gt_rgb_cat = torch.flip(gt_rgb_cat, dims=[0])
        gt_rgb_grid = torchvision.utils.make_grid(gt_rgb_cat[None, ...])
        self.logger.experiment.add_image('image/rgb_gts', gt_rgb_grid, self.global_step)
        # plt.imshow(torch.flip(gt_rgb_cat.permute(1, 2, 0), dims=[2]).numpy())
        # plt.show()
        # plt.close()
        rgb_preds_cat = torch.cat([torch.cat([rgb_pred for rgb_pred in rgb_preds[:3]], dim=2),
                                   torch.cat([rgb_pred for rgb_pred in torch.flip(rgb_preds[3:], dims=[0])], dim=2)],
                                  dim=1)
        rgb_preds_cat = torch.flip(rgb_preds_cat, dims=[0])
        # plt.imshow(rgb_preds_cat.sum(1).sum(1)[0].detach().cpu().numpy())
        # plt.show()
        # plt.close()
        rgb_preds_grid = torchvision.utils.make_grid(rgb_preds_cat[None, ...])
        self.logger.experiment.add_image('image/rgb_preds', rgb_preds_grid, self.global_step)
        # plt.imshow(rgb_preds_cat.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()
        depth_preds_cat = torch.cat(
            [torch.cat([visualize_depth(depth_pred) for depth_pred in depth_preds[:3]], dim=2),
             torch.cat([visualize_depth(depth_pred) for depth_pred in torch.flip(depth_preds[3:], dims=[0])], dim=2)],
            dim=1)
        depth_preds_grid = torchvision.utils.make_grid(depth_preds_cat[None, ...])
        self.logger.experiment.add_image('image/depth_preds', depth_preds_grid, self.global_step)
        # plt.imshow(depth_preds_cat.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()
        gt_seg_cat = torch.cat([torch.cat([visualize_semantic(gt_seg) for gt_seg in gt_segs[:3]], dim=2),
                                torch.cat([visualize_semantic(gt_seg) for gt_seg in torch.flip(gt_segs[3:], dims=[0])], dim=2)], dim=1)
        gt_seg_grid = torchvision.utils.make_grid(gt_seg_cat[None, ...])
        self.logger.experiment.add_image('image/seg_gts', gt_seg_grid, self.global_step)

        seg_preds_cat = torch.cat(
            [torch.cat([visualize_semantic(seg_pred) for seg_pred in seg_preds[:3]], dim=2),
             torch.cat([visualize_semantic(seg_pred) for seg_pred in torch.flip(seg_preds[3:], dims=[0])], dim=2)],
            dim=1)
        seg_preds_grid = torchvision.utils.make_grid(seg_preds_cat[None, ...])
        self.logger.experiment.add_image('image/seg_preds', seg_preds_grid, self.global_step)
        # plt.imshow(seg_preds_cat.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()
        bev_gt_seg = visualize_semantic(torch.flip(bev_gt_seg[0, 0], dims=[0, 1]).permute(1, 0))
        bev_gt_seg_grid = torchvision.utils.make_grid(bev_gt_seg[None, ...])
        self.logger.experiment.add_image('image/bev_gt_seg', bev_gt_seg_grid, self.global_step)


        bev_rgb_preds = torch.flip(bev_rgb_preds, dims=[0, 1, 2]).permute(0, 2, 1)
        bev_rgb_grid = torchvision.utils.make_grid(bev_rgb_preds[None, ...])
        self.logger.experiment.add_image('image/bev_rgb', bev_rgb_grid, self.global_step)


        bev_gt_height = visualize_depth(torch.flip(bev_gt_height[0, 0], dims=[0, 1]).permute(1, 0), min=-5., max=3.)
        bev_gt_height_grid = torchvision.utils.make_grid(bev_gt_height[None, ...])
        self.logger.experiment.add_image('image/bev_gt_height', bev_gt_height_grid, self.global_step)
        bev_height_preds = visualize_depth(torch.flip(bev_height_preds[0], dims=[0, 1]).permute(1, 0), min=-5., max=3.)
        bev_height_grid = torchvision.utils.make_grid(bev_height_preds[None, ...])
        self.logger.experiment.add_image('image/bev_height', bev_height_grid, self.global_step)

        bev_seg_preds = visualize_semantic(torch.flip(bev_seg_preds, dims=[0, 1]).permute(1, 0))
        bev_seg_grid = torchvision.utils.make_grid(bev_seg_preds[None, ...])
        self.logger.experiment.add_image('image/bev_seg', bev_seg_grid, self.global_step)

        # plt.imshow(bev_seg_grid.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

        bev_density = visualize_depth(torch.flip(bev_density[0].sum(0), dims=[0, 1]).permute(1, 0))
        bev_density_grid = torchvision.utils.make_grid(bev_density[None, ...])
        self.logger.experiment.add_image('image/bev_density', bev_density_grid, self.global_step)
        # plt.imshow(bev_density.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

    def get_occ_seg_loss(self, occ_semantics, occ_logits):
        all_occ_labels = occ_semantics#.reshape(b, 1)
        all_occ_logits = occ_logits#.reshape(b, -1)
        with autocast(enabled=False):
            occ_seg_loss = F.cross_entropy(all_occ_logits, all_occ_labels) + lovasz_softmax(
                F.softmax(all_occ_logits, dim=1), all_occ_labels)
        return occ_seg_loss

    def get_occ_density_loss(self, occ_density_labels, occ_density):
        all_occ_labels = occ_density_labels.reshape(-1)
        all_occ_density = occ_density.reshape(-1)
        # occ_mask = (all_occ_labels != 17)
        # free_mask = (all_occ_labels == 17)
        # label_density = (all_occ_labels != 17).float()
        with autocast(enabled=False):
            occ_density_loss = ((all_occ_labels - all_occ_density) ** 2).mean()
        return occ_density_loss

    def get_sdf_loss(self, pts_sdf_batch):
        all_pts_sdf = torch.cat(pts_sdf_batch, dim=0)
        with autocast(enabled=False):
            sdf_loss = ((all_pts_sdf - self.sdf_bias) ** 2).mean()
        return sdf_loss

    def get_rgb_loss(self, sweep_imgs, rgb_preds):
        b, _, _, h, w = sweep_imgs.shape
        # fg_mask = fg_mask.unsqueeze(2).expand(fg_mask.shape[0], fg_mask.shape[1], 3, fg_mask.shape[2], fg_mask.shape[3])
        # rgb_labels = sweep_imgs.permute(0, 1, 3, 4, 2)[fg_mask]
        # rgb_preds = rgb_preds.permute(0, 1, 3, 4, 2)[fg_mask]
        rgb_labels = sweep_imgs.reshape(-1, 3, h, w)
        rgb_preds = rgb_preds.reshape(-1, 3, h, w)
        with autocast(enabled=False):
            rgb_loss = F.smooth_l1_loss(rgb_preds, rgb_labels, reduction='none') + 1 - self.ms_ssim(rgb_preds, rgb_labels)
            rgb_loss = rgb_loss.mean()
        return rgb_loss

    def get_camera_seg_loss(self, seg_labels, seg_logits_preds, fg_mask):
        seg_labels = seg_labels[fg_mask]
        seg_logits_preds = seg_logits_preds.permute(0, 1, 3, 4, 2)[fg_mask]
        with autocast(enabled=False):
            camera_seg_loss = F.cross_entropy(seg_logits_preds, seg_labels, ) + lovasz_softmax(
                F.softmax(seg_logits_preds, dim=1), seg_labels)
            # camera_seg_loss = F.cross_entropy(seg_logits_preds, seg_labels, ignore_index=self.ignore_label) + lovasz_softmax(
            #     F.softmax(seg_logits_preds, dim=1), seg_labels, ignore=self.ignore_label)
        return camera_seg_loss

    def get_pts_seg_loss(self, pts_labels, pts_logits_batch):
        all_pts_labels = torch.cat(pts_labels, dim=0)
        all_pts_logits = torch.cat(pts_logits_batch, dim=0)
        with autocast(enabled=False):
            pts_seg_loss = F.cross_entropy(all_pts_logits, all_pts_labels) + lovasz_softmax(
                F.softmax(all_pts_logits, dim=1), all_pts_labels)
            # pts_seg_loss = F.cross_entropy(all_pts_logits, all_pts_labels, ignore_index=self.ignore_label) + lovasz_softmax(
            #     F.softmax(all_pts_logits, dim=1), all_pts_labels, ignore=self.ignore_label)
        return pts_seg_loss

    def get_bev_seg_loss(self, bev_seg, bev_seg_logits_preds, bev_mask):
        bev_seg = bev_seg[bev_mask]
        bev_seg_logits_preds = bev_seg_logits_preds[:, None, None, ...].permute(0, 1, 2, 4, 5, 3)[bev_mask]
        with autocast(enabled=False):
            bev_seg_loss = F.cross_entropy(bev_seg_logits_preds, bev_seg) + lovasz_softmax(
                F.softmax(bev_seg_logits_preds, dim=1), bev_seg)
            # bev_seg_loss = F.cross_entropy(bev_seg_logits_preds, bev_seg, ignore_index=self.ignore_label) + lovasz_softmax(
            #     F.softmax(bev_seg_logits_preds, dim=1), bev_seg, ignore=self.ignore_label)
        return bev_seg_loss

    def get_height_loss_bev(self, bev_height, bev_height_preds, bev_mask):
        bev_height = bev_height[bev_mask]
        bev_height_preds = bev_height_preds.unsqueeze(1)[bev_mask]
        with autocast(enabled=False):
            bev_height_loss = F.smooth_l1_loss(bev_height, bev_height_preds)
        return bev_height_loss

    def get_depth_loss(self, depth_labels, depth_preds, fg_mask):
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.smooth_l1_loss(depth_preds, depth_labels)
            # depth_loss = depth_loss.mean()
        return depth_loss

    def get_downsampled_gt(self, gt_imgs, gt_depths, gt_segs):
        """
        Input:
            gt_imgs: [B, N, 3, H, W]
            gt_depths: [B, N, H, W]
            gt_segs: [B, N, H, W]
        Output:
            gt_imgs: [B, N, 3, h, w]
            gt_depths: [B, N, h, w]
            gt_segs: [B, N, h, w]
        """
        B, N, _, H, W = gt_imgs.shape
        gt_imgs = gt_imgs.view(B, N, 3,
                               H * self.upsample_factor // self.downsample_factor,
                               self.downsample_factor // self.upsample_factor,
                               W * self.upsample_factor // self.downsample_factor,
                               self.downsample_factor // self.upsample_factor)
        gt_imgs = gt_imgs.permute(0, 1, 2, 3, 5, 4, 6)[..., 0, 0].contiguous()
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=gt_imgs.device)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=gt_imgs.device)
        gt_imgs = gt_imgs * rgb_std[None, None, :, None, None] + rgb_mean[None, None, :, None, None]
        gt_depths = gt_depths.view(B, N,
                                   H * self.upsample_factor // self.downsample_factor,
                                   self.downsample_factor // self.upsample_factor,
                                   W * self.upsample_factor // self.downsample_factor,
                                   self.downsample_factor // self.upsample_factor)
        gt_depths = gt_depths.permute(0, 1, 2, 4, 3, 5)[..., 0, 0].contiguous()
        gt_segs = gt_segs.view(B, N,
                               H * self.upsample_factor // self.downsample_factor,
                               self.downsample_factor // self.upsample_factor,
                               W * self.upsample_factor // self.downsample_factor,
                               self.downsample_factor // self.upsample_factor)
        gt_segs = gt_segs.permute(0, 1, 2, 4, 3, 5)[..., 0, 0].contiguous()

        fg_mask = gt_depths > 0

        return gt_imgs, gt_depths, gt_segs, fg_mask

    def validation_step(self, batch, batch_idx):
        # in validation, we only consider the lidar segmentation results
        (sweep_imgs, mats, _, img_metas, _, _, inrange_pts, inrange_labels, ref_labels, ref_index, _, occ_semantics, occ_density_labels, mask_lidar, mask_camera) = batch
        if torch.cuda.is_available() and not self.debug:
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            inrange_pts = [inrange_pt.cuda() for inrange_pt in inrange_pts]
            # ref_labels = [ref_label.cuda() for ref_label in ref_labels]
            # ref_index = [ref_idx.cuda() for ref_idx in ref_index]
        pts_logits_batch, occ_logits_batch, occ_density_batch = self(sweep_imgs, mats, inrange_pts=inrange_pts, lidar_seg=True)
        # seg_results = list()
        for pts_logits, ref_idx, ref_label in zip(pts_logits_batch, ref_index, ref_labels):
            ref_logits = torch.zeros((len(ref_label), self.num_seg_classes), device=self.device)
            ref_logits.index_add_(0, ref_idx, pts_logits)
            seg_result = ref_logits[..., 1:-1].argmax(1) + 1
            self.val_iou(seg_result, ref_label)
        occ_semantics = occ_semantics[mask_camera]
        occ_logits_batch = occ_logits_batch[mask_camera]
        # occ_density_batch = occ_density_batch[mask_camera]
        # occ_free = (occ_density_batch == 0).float()
        # occ_logits = torch.cat((occ_logits_batch, occ_free * 1e8), dim=1)
        # occ_result = occ_logits.argmax(1)
        occ_result = occ_logits_batch.argmax(1)
        self.occ_val_iou(occ_result, occ_semantics)
        # for occ_logits,
            # seg_results.append((seg_result.cpu().detach().numpy(), ref_label.cpu().detach().numpy()))
            # for cnt in range(len(ref_label)):
            # self.val_iou._after_step(seg_result.cpu().detach().numpy(), ref_label.cpu().detach().numpy())
        # return seg_results

    def test_step(self, batch, batch_idx):
        # in test, we consider both detection and lidar segmentation
        (sweep_imgs, mats, _, img_metas, _, _, inrange_pts, inrange_labels, ref_labels, ref_index, lidar_tokens, occ_semantics, occ_density_labels, mask_lidar, mask_camera) = batch
        if torch.cuda.is_available() and not self.debug:
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            inrange_pts = [inrange_pt.cuda() for inrange_pt in inrange_pts]
        preds, \
        rgb_preds, seg_logits_preds, depth_preds, \
        bev_rgb_preds, bev_seg_logits_preds, bev_height_preds, bev_density, \
        pts_logits_batch, pts_sdf_batch, occ_logits_batch, occ_density_batch = self(sweep_imgs, mats, inrange_pts=inrange_pts, lidar_seg=False)

        if self.vis:
            seg_preds = torch.max(seg_logits_preds.permute(0, 1, 3, 4, 2).softmax(-1), dim=-1)[1]
            bev_seg_preds = torch.max(bev_seg_logits_preds.permute(0, 2, 3, 1).softmax(-1), dim=-1)[1]
            a = occ_logits_batch[0] * occ_density_batch[0]
            occ_preds = a.argmax(-1).cpu().detach().numpy()
            gt_rgbs, depth_preds, seg_preds, bev_height_preds, bev_seg_preds, bev_density = self.preprocess_vis_img(
                sweep_imgs[0][0].cpu().detach(),
                depth_preds[0].cpu().detach(),
                seg_preds[0].cpu().detach(),
                bev_height_preds[0].cpu().detach(),
                bev_seg_preds[0].cpu().detach(),
                bev_density[0].cpu().detach())
            vis_dict = {
                'batch_idx': batch_idx,
                'lidar_token': lidar_tokens[0],
                'input_image': gt_rgbs, # 512x2212x3
                'camera_depth': depth_preds, # 512x2212x3
                'camera_semantics': seg_preds, # 512x2212x3
                'bev_semantics': bev_seg_preds, # 200x200x3
                'bev_density': bev_density, # 200x200x3
                'occ': occ_preds, # 200x200x16
            }
            os.makedirs(self.vis_dir, exist_ok=True)
            save_dir = os.path.join(self.vis_dir, 'scene_0012_0018')
            os.makedirs(save_dir, exist_ok=True)
            full_label_name = os.path.join(self.vis_dir, 'scene_0012_0018', str(batch_idx) + '.pkl')
            if os.path.exists(full_label_name):
                print('%s already exsist...' % (full_label_name))
            else:
                with open(full_label_name, "wb") as handle:
                    pickle.dump(vis_dict, handle)
                    print("\nwrote to", full_label_name)
        else:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                det_results = self.model.module.get_bboxes(preds, img_metas)
            else:
                det_results = self.model.get_bboxes(preds, img_metas)
            for i in range(len(det_results)):
                det_results[i][0] = det_results[i][0].detach().cpu().numpy()
                det_results[i][1] = det_results[i][1].detach().cpu().numpy()
                det_results[i][2] = det_results[i][2].detach().cpu().numpy()
                det_results[i].append(img_metas[i])
            # meta_dict = {
            #     "meta": {
            #         "use_camera": True,
            #         "use_lidar": False,
            #         "use_map": False,
            #         "use_radar": False,
            #         "use_external": False,
            #     }
            # }
            # os.makedirs(os.path.join(self.lidar_seg_submit_dir, 'val'), exist_ok=True)
            # with open(os.path.join(self.lidar_seg_submit_dir, 'val', 'submission.json'), 'w', encoding='utf-8') as f:
            #     json.dump(meta_dict, f)
            # for pts_logits, ref_idx, ref_label, lidar_token in zip(pts_logits_batch, ref_index, ref_labels, lidar_tokens):
            #     ref_logits = torch.zeros((len(ref_label), self.num_seg_classes))
            #     ref_logits.index_add_(0, ref_idx.cpu(), pts_logits.cpu())
            #     seg_result = ref_logits[:, 1:-1].argmax(1) + 1
            #     original_label = seg_result.cpu().numpy().astype(np.uint8)
            #     assert all((original_label > 0) & (original_label < 17)), \
            #         "Error: Array for predictions must be between 1 and 16 (inclusive)."
            #     full_save_dir = os.path.join(self.lidar_seg_submit_dir, 'lidarseg/val')
            #     full_label_name = os.path.join(full_save_dir, lidar_token + '_lidarseg.bin')
            #     os.makedirs(full_save_dir, exist_ok=True)
            #     if os.path.exists(full_label_name):
            #         print('%s already exsist...' % (full_label_name))
            #     else:
            #         original_label.tofile(full_label_name)
            return det_results

    def preprocess_vis_img(self, gt_rgbs, depth_preds, seg_preds, bev_height_preds, bev_seg_preds, bev_density):
        num_cams, _, h, w = gt_rgbs.shape
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=gt_rgbs.device)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=gt_rgbs.device)
        gt_rgbs = gt_rgbs * rgb_std[None, :, None, None] + rgb_mean[None, :, None, None]
        gt_rgb_cat = torch.cat([torch.cat([gt_rgb for gt_rgb in gt_rgbs[:3]], dim=2),
                                torch.cat([gt_rgb for gt_rgb in torch.flip(gt_rgbs[3:], dims=[0])], dim=2)], dim=1)
        gt_rgb_cat = torch.flip(gt_rgb_cat, dims=[0])
        # plt.imshow(gt_rgb_cat.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

        depth_preds = depth_preds[:, 0, :, :]
        depth_preds_cat = torch.cat(
            [torch.cat([visualize_depth(depth_pred) for depth_pred in depth_preds[:3]], dim=2),
             torch.cat([visualize_depth(depth_pred) for depth_pred in torch.flip(depth_preds[3:], dims=[0])], dim=2)],
            dim=1)

        # plt.imshow(depth_preds_cat.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

        seg_preds_cat = torch.cat(
            [torch.cat([visualize_semantic(seg_pred) for seg_pred in seg_preds[:3]], dim=2),
             torch.cat([visualize_semantic(seg_pred) for seg_pred in torch.flip(seg_preds[3:], dims=[0])], dim=2)],
            dim=1)
        # plt.imshow(seg_preds_cat.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

        bev_height_preds = visualize_depth(torch.flip(bev_height_preds[0], dims=[0, 1]).permute(1, 0), min=-5., max=3.)
        # plt.imshow(bev_height_preds.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

        bev_seg_preds = visualize_semantic(torch.flip(bev_seg_preds, dims=[0, 1]).permute(1, 0))
        # plt.imshow(bev_seg_preds[:, 28:-28, 28:-28].permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

        bev_density = visualize_depth(torch.flip(bev_density[0].sum(0), dims=[0, 1]).permute(1, 0))
        # plt.imshow(bev_density.permute(1, 2, 0).numpy())
        # plt.show()
        # plt.close()

        return gt_rgb_cat.permute(1, 2, 0).numpy(), \
               depth_preds_cat.permute(1, 2, 0).numpy(), \
               seg_preds_cat.permute(1, 2, 0).numpy(), \
               bev_height_preds[:, 28:-28, 28:-28].permute(1, 2, 0).numpy(), \
               bev_seg_preds[:, 28:-28, 28:-28].permute(1, 2, 0).numpy(), \
               bev_density[:, 28:-28, 28:-28].permute(1, 2, 0).numpy()

    def predict_step(self, batch, batch_idx):
        # in test, we consider both detection and lidar segmentation
        (sweep_imgs, mats, _, img_metas, _, _, inrange_pts, inrange_labels, ref_labels, ref_index, lidar_tokens, _, _, _, _) = batch
        if torch.cuda.is_available() and not self.debug:
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            inrange_pts = [inrange_pt.cuda() for inrange_pt in inrange_pts]
        preds, \
        rgb_preds, seg_logits_preds, depth_preds, \
        bev_rgb_preds, bev_seg_logits_preds, bev_height_preds, bev_density, \
        pts_logits_batch, pts_sdf_batch, occ_logits_batch, occ_density_batch = self(sweep_imgs, mats,
                                                                                    inrange_pts=inrange_pts,
                                                                                    lidar_seg=False)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            det_results = self.model.module.get_bboxes(preds, img_metas)
        else:
            det_results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(det_results)):
            det_results[i][0] = det_results[i][0].detach().cpu().numpy()
            det_results[i][1] = det_results[i][1].detach().cpu().numpy()
            det_results[i][2] = det_results[i][2].detach().cpu().numpy()
            det_results[i].append(img_metas[i])
        meta_dict = {
            "meta": {
                "use_camera": True,
                "use_lidar": False,
                "use_map": False,
                "use_radar": False,
                "use_external": False,
            }
        }
        os.makedirs(os.path.join(self.lidar_seg_submit_dir, 'test'), exist_ok=True)
        with open(os.path.join(self.lidar_seg_submit_dir, 'test', 'submission.json'), 'w', encoding='utf-8') as f:
            json.dump(meta_dict, f)
        for pts_logits, ref_idx, ref_label, lidar_token in zip(pts_logits_batch, ref_index, ref_labels, lidar_tokens):
            ref_logits = torch.zeros((len(ref_label), self.num_seg_classes))
            ref_logits.index_add_(0, ref_idx.cpu(), pts_logits.cpu())
            seg_result = ref_logits[:, 1:-1].argmax(1) + 1
            original_label = seg_result.cpu().numpy().astype(np.uint8)
            assert all((original_label > 0) & (original_label < 17)), \
                "Error: Array for predictions must be between 1 and 16 (inclusive)."
            full_save_dir = os.path.join(self.lidar_seg_submit_dir, 'lidarseg/test')
            full_label_name = os.path.join(full_save_dir, lidar_token + '_lidarseg.bin')
            os.makedirs(full_save_dir, exist_ok=True)
            if os.path.exists(full_label_name):
                print('%s already exsist...' % (full_label_name))
            else:
                original_label.tofile(full_label_name)
        return det_results

    def training_epoch_end(self, training_step_outputs) -> None:
        iou = self.train_iou.compute()[1:].cpu().detach().numpy()
        miou = np.nanmean(iou)
        str_print = ''
        self.log('train/mIoU', miou)
        self.train_iou.reset()
        str_print += 'Training per class iou: '

        for class_name, class_iou in zip(self.unique_label_str, iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        str_print += '\nCurrent training miou is %.3f' % (miou * 100)
        str_print += '\n###########################################'
        self.print(str_print)

        occ_iou = self.occ_train_iou.compute()[:-1].cpu().detach().numpy()
        occ_miou = np.nanmean(occ_iou)
        str_print = ''
        self.log('train/occ_mIoU', occ_miou)
        self.occ_train_iou.reset()
        str_print += 'Training per class occupancy iou: '

        for class_name, class_iou in zip(self.occ_label_str, occ_iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        str_print += '\nCurrent train occupacny miou is %.3f' % (occ_miou * 100)
        str_print += '\n###########################################'
        self.print(str_print)

    def validation_epoch_end(self, validation_step_outputs):
        iou = self.val_iou.compute()[1:].cpu().detach().numpy()
        miou = np.nanmean(iou)
        if miou > self.best_miou:
            self.best_miou = miou
        str_print = ''
        self.log('val/mIoU', miou)
        self.val_iou.reset()
        str_print += 'Validation per class iou: '

        for class_name, class_iou in zip(self.unique_label_str, iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        str_print += '\nCurrent val miou is %.3f while the best val miou is %.3f' % (miou * 100, self.best_miou * 100)
        str_print += '\n###########################################'
        self.print(str_print)
        occ_iou = self.occ_val_iou.compute()[:-1].cpu().detach().numpy()
        occ_miou = np.nanmean(occ_iou)
        if occ_miou > self.best_occ_miou:
            self.best_occ_miou = occ_miou
        str_print = ''
        self.log('val/occ_mIoU', occ_miou)
        self.occ_val_iou.reset()
        str_print += 'Validation per class occupancy iou: '

        for class_name, class_iou in zip(self.occ_label_str, occ_iou):
            str_print += '\n%s : %.2f%%' % (class_name, class_iou * 100)

        str_print += '\nCurrent val occupacny miou is %.3f while the best val occupancy miou is %.3f' % (occ_miou * 100, self.best_occ_miou * 100)
        str_print += '\n###########################################'
        self.print(str_print)

    def test_epoch_end(self, test_step_outputs):
        if not self.vis:
            all_pred_results = list()
            all_img_metas = list()
            for test_step_output in test_step_outputs:
                for i in range(len(test_step_output)):
                    all_pred_results.append(test_step_output[i][:3])
                    all_img_metas.append(test_step_output[i][3])
            synchronize()
            # TODO: Change another way.
            dataset_length = len(self.val_dataloader().dataset)
            all_pred_results = sum(
                map(list, zip(*all_gather_object(all_pred_results))),
                [])[:dataset_length]
            all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                                [])[:dataset_length]
            if get_rank() == 0:
                self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
             self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        if self.trainer.max_epochs == 24:
            scheduler = MultiStepLR(optimizer, [19, 23])
        elif self.trainer.max_epochs == 48:
            scheduler = MultiStepLR(optimizer, [38, 46])
        else:
            raise NotImplementedError
        return [[optimizer], [scheduler]]

    # def configure_optimizers(self):
    #     lr = self.basic_lr_per_img * \
    #          self.batch_size_per_device * self.gpus
    #     optimizer = torch.optim.AdamW(self.model.parameters(),
    #                                   lr=lr,
    #                                   weight_decay=1e-7)
    #     scheduler = MultiStepLR(optimizer, [38, 46])
    #     return [[optimizer], [scheduler]]

    def train_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=self.trainval_info_paths if self.trainval else self.train_info_paths,
                                       mode='train',
                                       use_dense=False,  # True for dense pts
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn, mode='train'),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                     bda_aug_conf=self.bda_aug_conf,
                                     classes=self.class_names,
                                     data_root=self.data_root,
                                     info_paths=self.val_info_paths,
                                     mode='val',
                                     use_dense=False,
                                     img_conf=self.img_conf,
                                     num_sweeps=self.num_sweeps,
                                     sweep_idxes=self.sweep_idxes,
                                     key_idxes=self.key_idxes)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, mode='val'),
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        test_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                      bda_aug_conf=self.bda_aug_conf,
                                      classes=self.class_names,
                                      data_root=self.data_root,
                                      info_paths=self.vis_info_paths if self.vis else self.val_info_paths,
                                      mode='val',
                                      use_dense=False,
                                      img_conf=self.img_conf,
                                      num_sweeps=self.num_sweeps,
                                      sweep_idxes=self.sweep_idxes,
                                      key_idxes=self.key_idxes)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, mode='val'),
            num_workers=4,
            sampler=None,
        )
        return test_loader

    def predict_dataloader(self):
        predict_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                         bda_aug_conf=self.bda_aug_conf,
                                         classes=self.class_names,
                                         data_root=self.data_root,
                                         info_paths=self.predict_info_paths,
                                         mode='test',
                                         use_dense=False,
                                         img_conf=self.img_conf,
                                         num_sweeps=self.num_sweeps,
                                         sweep_idxes=self.sweep_idxes,
                                         key_idxes=self.key_idxes)
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, mode='test'),
            num_workers=4,
            sampler=None,
        )
        return predict_loader

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser
