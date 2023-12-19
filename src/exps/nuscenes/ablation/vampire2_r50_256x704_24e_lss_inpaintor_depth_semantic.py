from src.exps.base_cli import run_cli
from src.exps.nuscenes.base_exp import VAMPIRELightningModel as BaseVAMPIRELightningModel
from src.models.vampire2 import VAMPIRE2

# H = 900
# W = 1600
# final_dim = (256, 704)
# resize_lim = (0.386, 0.55)
# sample_factor = 4
# # final_dim = (900, 1600)
# # resize_lim = (1.0, 1.0)
# img_conf = dict(img_mean=[123.675, 116.28, 103.53],
#                 img_std=[58.395, 57.12, 57.375],
#                 to_rgb=True)
#
# backbone_conf = {
#     'x_bound_seg': [-51.2, 51.2, 0.4],
#     'y_bound_seg': [-51.2, 51.2, 0.4],
#     'z_bound_seg': [-5., 3., 0.4],
#     'x_bound_det': [-51.2, 51.2, 0.4],
#     'y_bound_det': [-51.2, 51.2, 0.4],
#     'z_bound_det': [-1., 3., 0.4],
#     'd_bound': [2.0, 70.4, 0.8],
#     # 'resolution': 0.4,
#     'final_dim':
#         final_dim,
#     'density_mode': 'sdf',
#     'sdf_bias': -1.0,
#     'cat_pos': True,
#     'cat_seg': False,
#     'mid_channels':
#         16,
#     'output_channels':
#         # 32,
#         80,
#     'downsample_factor':
#         sample_factor,
#     'upsample_factor':
#         sample_factor,
#     'img_backbone_conf':
#         dict(
#             type='ResNet',
#             depth=50,
#             frozen_stages=0,
#             out_indices=[0, 1, 2, 3],
#             # out_indices=[0, 1, 2],
#             norm_eval=False,
#             # with_cp=True,
#             init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
#         ),
#     'img_neck_conf':
#         dict(
#             type='SECONDFPN',
#             in_channels=[256, 512, 1024, 2048],
#             upsample_strides=[0.5, 1, 2, 4],
#             out_channels=[128, 128, 128, 128],
#
#             # in_channels=[256, 512, 1024],
#             # upsample_strides=[0.5, 1, 2],
#             # out_channels=[32, 32, 32],
#
#             # in_channels=[256, 512, 1024, 2048],
#             # upsample_strides=[0.5, 1, 2, 4],
#             # out_channels=[20, 20, 20, 20],
#         ),
#
#     'num_classes': 18,
# }
# ida_aug_conf = {
#     'resize_lim': resize_lim,
#     'final_dim':
#         final_dim,
#     'rot_lim': (-5.4, 5.4),
#     'H':
#         H,
#     'W':
#         W,
#     'rand_flip':
#         True,
#     'bot_pct_lim': (0.0, 0.0),
#     'cams': [
#         'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
#         'CAM_BACK', 'CAM_BACK_RIGHT'
#     ],
#     'Ncams':
#     6,
# }
#
# bda_aug_conf = {
#     # 'rot_lim': (-22.5, 22.5),
#     'rot_lim': (0.0, 0.0),
#     # 'scale_lim': (0.95, 1.05),
#     'scale_lim': (1., 1.),
#     'flip_dx_ratio': 0.,
#     'flip_dy_ratio': 0.
# }
#
# bev_backbone = dict(
#     type='ResNet',
#     in_channels=80,
#     depth=18,
#     num_stages=3,
#     strides=(1, 2, 2),
#     dilations=(1, 1, 1),
#     out_indices=[0, 1, 2],
#     norm_eval=False,
#     base_channels=160,
# )
#
# bev_neck = dict(type='SECONDFPN',
#                 in_channels=[80, 160, 320, 640],
#                 upsample_strides=[1, 2, 4, 8],
#                 out_channels=[64, 64, 64, 64])
#
# CLASSES = [
#     'car',
#     'truck',
#     'construction_vehicle',
#     'bus',
#     'trailer',
#     'barrier',
#     'motorcycle',
#     'bicycle',
#     'pedestrian',
#     'traffic_cone',
# ]
#
# TASKS = [
#     dict(num_class=1, class_names=['car']),
#     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
#     dict(num_class=2, class_names=['bus', 'trailer']),
#     dict(num_class=1, class_names=['barrier']),
#     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
#     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
# ]
#
# common_heads = dict(reg=(2, 2),
#                     height=(1, 2),
#                     dim=(3, 2),
#                     rot=(2, 2),
#                     vel=(2, 2))
#
# bbox_coder = dict(
#     type='CenterPointBBoxCoder',
#     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#     max_num=500,
#     score_threshold=0.1,
#     out_size_factor=4,
#     voxel_size=[0.2, 0.2, 8],
#     pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
#     code_size=9,
# )
#
# train_cfg = dict(
#     point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
#     grid_size=[512, 512, 1],
#     voxel_size=[0.2, 0.2, 8],
#     out_size_factor=4,
#     dense_reg=1,
#     gaussian_overlap=0.1,
#     max_objs=500,
#     min_radius=2,
#     code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
# )
#
# test_cfg = dict(
#     post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#     max_per_img=500,
#     max_pool_nms=False,
#     min_radius=[4, 12, 10, 1, 0.85, 0.175],
#     score_threshold=0.1,
#     out_size_factor=4,
#     voxel_size=[0.2, 0.2, 8],
#     nms_type='circle',
#     pre_max_size=1000,
#     post_max_size=83,
#     nms_thr=0.2,
# )
#
# head_conf = {
#     'bev_backbone_conf': bev_backbone,
#     'bev_neck_conf': bev_neck,
#     'tasks': TASKS,
#     'common_heads': common_heads,
#     'bbox_coder': bbox_coder,
#     'train_cfg': train_cfg,
#     'test_cfg': test_cfg,
#     'in_channels': 256,  # Equal to bev_neck output_channels.
#     'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
#     'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
#     'gaussian_overlap': 0.1,
#     'min_radius': 2,
# }
# unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# label_16_names = ['noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
#                   'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
#                   'manmade', 'vegetation']
# label_17_names = ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
#                   'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
#                   'manmade', 'vegetation', 'free']


class VAMPIRELightningModel(BaseVAMPIRELightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_weights = [1., 1., 0., 0., 0.]
        self.model = VAMPIRE2(self.backbone_conf,
                                  self.head_conf)

if __name__ == '__main__':
    # train
    # python src/exps/nuscenes/ablation/vampire2_r50_256x704_24e_lss_inpaintor_depth_semantic.py --amp_backend native -b 1 --gpus 8
    # test
    # python src/exps/nuscenes/ablation/vampire2_r50_256x704_24e_lss_inpaintor_depth_semantic.py -t -b 1 --gpus 8 --ckpt_path ...
    run_cli(VAMPIRELightningModel,
            'ablation/vampire2_r50_256x704_24e_lss_inpaintor_depth_semantic')