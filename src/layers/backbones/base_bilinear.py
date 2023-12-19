import math
import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from src.utils.render_utils import ModifyLaplaceDensity
from src.utils.vis_utils import visualize_geomxyz
import matplotlib.pyplot as plt

__all__ = ['BaseBiLinear']


class Unet3D(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Unet3D, self).__init__()
        self.init_dres = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.hg1 = Hourglass3D(mid_channels)
        self.hg2 = Hourglass3D(mid_channels)

    def forward(self, x):
        dres = self.init_dres(x)
        out1, pre1, post1 = self.hg1(dres, None, None)
        out1 = out1 + dres
        out2, pre2, post2 = self.hg2(out1, pre1, post1)
        out2 = out2 + dres
        return out2

class Hourglass3D(nn.Module):
    def __init__(self, mid_channels):
        super(Hourglass3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, presqu=None, postsqu=None):
        out = self.conv1(x)  # 1 64 10 128 128
        pre = self.conv2(out)  # 1 64 10 128 128

        if postsqu is not None:
            pre = F.leaky_relu(pre + postsqu, inplace=True)
        else:
            pre = F.leaky_relu(pre, inplace=True)
        out = self.conv3(pre)  # 1 64 5 64 64
        out = self.conv4(out)  # 1 64 5 64 64
        out = F.interpolate(out, (pre.shape[-3], pre.shape[-2], pre.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv5(out)  # 1 64 10 128 128
        if presqu is not None:
            post = F.leaky_relu(out + presqu, inplace=True)
        else:
            post = F.leaky_relu(out + pre, inplace=True)
        out = F.interpolate(post, (x.shape[-3], x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv6(out)
        return out, pre, post


class BaseBiLinear(nn.Module):

    def __init__(self,
                 x_bound_seg,
                 y_bound_seg,
                 z_bound_seg,
                 x_bound_det,
                 y_bound_det,
                 z_bound_det,
                 d_bound,
                 # resolution,
                 final_dim,
                 downsample_factor,
                 upsample_factor,
                 mid_channels,
                 output_channels,
                 img_backbone_conf,
                 img_neck_conf,
                 num_classes,
                 density_mode='naive',
                 sdf_bias=-1.0,
                 cat_pos=False,
                 cat_seg=False,
                 use_da=False):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound_seg (list): Boundaries for x.
            y_bound_seg (list): Boundaries for y.
            z_bound_seg (list): Boundaries for z.
            x_bound_det (list): Boundaries for x.
            y_bound_det (list): Boundaries for y.
            z_bound_det (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            num_classes (int): Number of classes for the semantic segmentation.
        """

        super(BaseBiLinear, self).__init__()

        self.downsample_factor = downsample_factor
        self.upsample_factor = upsample_factor
        self.num_classes = num_classes
        self.x_bound_seg = x_bound_seg
        self.y_bound_seg = y_bound_seg
        self.z_bound_seg = z_bound_seg
        self.x_bound_det = x_bound_det
        self.y_bound_det = y_bound_det
        self.z_bound_det = z_bound_det
        self.d_bound = d_bound
        # self.resolution = resolution
        self.final_dim = final_dim
        self.mid_channels = mid_channels
        self.output_channels = output_channels
        self.density_mode = density_mode
        self.sdf_bias = sdf_bias
        self.cat_pos = cat_pos
        self.cat_seg = cat_seg

        self.register_buffer('frustum', self.create_frustum())
        # self.frustum = self.create_frustum()
        self.fD = self.frustum.shape[0] - 1
        self.fH = self.frustum.shape[1]
        self.fW = self.frustum.shape[2]
        self.register_buffer('camera_mids', self.create_camera_mids())
        self.register_buffer('bev_mids', self.create_bev_mids())
        self.register_buffer('voxel_coords', self.create_voxel_coords(self.x_bound_seg, self.y_bound_seg, self.z_bound_seg))
        self.register_buffer('norm_occ_coords', self.create_norm_occ_coords())
        # self.voxel_coords = self.create_voxel_coords(self.resolution, self.resolution, self.resolution)
        if self.cat_pos:
            self.register_buffer('norm_voxel_coords',
                                 self.create_voxel_coords(self.x_bound_seg, self.y_bound_seg, self.z_bound_seg, norm=True))
        self.register_buffer('output_coords', self.create_voxel_coords(self.x_bound_det, self.y_bound_det, self.z_bound_det))
        self.vZ = self.voxel_coords.shape[0]
        self.vY = self.voxel_coords.shape[1]
        self.vX = self.voxel_coords.shape[2]
        self.oY = self.output_coords.shape[1]
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        img_out_channels = sum(img_neck_conf['out_channels'])

        self.channel_lower = nn.Conv2d(img_out_channels, self.mid_channels, 3, 1, 1, bias=False)
        voxel_in_channels = self.mid_channels
        if cat_pos:
            voxel_in_channels += 3
        self.base_conv = nn.Sequential(
            nn.Conv3d(voxel_in_channels, self.mid_channels, 3, 1, 1, bias=True),
            nn.Softplus(beta=100),
            # nn.Conv3d(self.mid_channels, self.mid_channels, 3, 1, 1, bias=True),
            # nn.Softplus(beta=100),
            # nn.Conv3d(self.mid_channels, self.mid_channels + 1 + self.num_classes, 3, 1, 1, bias=True),
        )
        self.density_conv = nn.Conv3d(self.mid_channels, 1, 3, 1, 1, bias=True)
        self.seg_conv = nn.Conv3d(self.mid_channels, self.num_classes, 3, 1, 1, bias=True)
        self.feature_conv = nn.Conv3d(self.mid_channels, self.mid_channels, 3, 1, 1, bias=True)
        if density_mode == 'naive':
            self.density = nn.Sigmoid()
        else:
            self.density = ModifyLaplaceDensity(beta=0.1, bias=self.sdf_bias)
        self.rgb_conv = nn.Sequential(
            nn.Conv3d(self.mid_channels, 3, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        if self.cat_seg:
            voxel_out_in = self.mid_channels + self.num_classes
        else:
            voxel_out_in = self.mid_channels
        if self.oY == 128:
            self.voxel_output = nn.Conv2d(voxel_out_in * self.output_coords.shape[0], self.output_channels, 1, 1, bias=True)
        elif self.oY == 256:
            self.voxel_output = nn.Sequential(
                nn.Conv2d(voxel_out_in * self.output_coords.shape[0], self.output_channels, 1, 1, bias=True),
                nn.UpsamplingBilinear2d(scale_factor=0.5),
            )
        self.upsample2d = nn.UpsamplingBilinear2d(scale_factor=self.upsample_factor)
        self.init_weights()
        self.img_neck.init_weights()
        self.img_backbone.init_weights()
        self.use_da = use_da

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.constant_(m.bias, 2.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                torch.nn.init.normal_(m.weight, 0.0, np.sqrt(2) / np.sqrt(m.weight.shape[1]))
        for m in self.density_conv.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                bound = 1 / math.sqrt(n)
                nn.init.uniform_(m.bias, self.sdf_bias - 10.0)

    def create_camera_mids(self):
        tdist = torch.arange(*self.d_bound, dtype=torch.float)
        t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
        return t_mids

    def create_bev_mids(self):
        tdist = torch.linspace(self.z_bound_det[0] + self.z_bound_det[2] / 2., self.z_bound_det[1] - self.z_bound_det[2] / 2.,
                            int((self.z_bound_det[1] - self.z_bound_det[0]) / self.z_bound_det[2]), dtype=torch.float)# - self.z_bound[0]
        return torch.flip(tdist, dims=[0])

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def create_voxel_coords(self, x_bound, y_bound, z_bound, norm=False):
        """Generate voxel"""
        # make grid in bev plane
        zs = torch.linspace(z_bound[0] + z_bound[2] / 2., z_bound[1] - z_bound[2] / 2.,
                            int((z_bound[1] - z_bound[0]) / z_bound[2]), dtype=torch.float)
        ys = torch.linspace(y_bound[0] + y_bound[2] / 2., y_bound[1] - y_bound[2] / 2.,
                            int((y_bound[1] - y_bound[0]) / y_bound[2]), dtype=torch.float)
        xs = torch.linspace(x_bound[0] + x_bound[2] / 2., x_bound[1] - x_bound[2] / 2.,
                            int((x_bound[1] - x_bound[0]) / x_bound[2]), dtype=torch.float)
        if norm:
            norm_zs = (zs - z_bound[0]) / (z_bound[1] - z_bound[0])
            norm_ys = (ys - y_bound[0]) / (y_bound[1] - y_bound[0])
            norm_xs = (xs - x_bound[0]) / (x_bound[1] - x_bound[0])
            norm_zs, norm_ys, norm_xs = torch.meshgrid(norm_zs, norm_ys, norm_xs)
            norm_voxel_coords = torch.stack([norm_xs, norm_ys, norm_zs], dim=-1) * 2. - 1.
            return norm_voxel_coords
        else:
            zs, ys, xs = torch.meshgrid(zs, ys, xs)
            paddings = torch.ones_like(xs)
            voxel_coords = torch.stack([xs, ys, zs, paddings], dim=-1)
            return voxel_coords

    def create_norm_occ_coords(self, point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], voxelSize=[0.4, 0.4, 0.4]):
        mask = torch.ones((200, 200, 16), dtype=torch.bool)
        occIdx = torch.where(mask)
        occ_coords = torch.cat((occIdx[0][:, None] * voxelSize[0] + voxelSize[0] / 2 + point_cloud_range[0], \
                                occIdx[1][:, None] * voxelSize[1] + voxelSize[1] / 2 + point_cloud_range[1], \
                                occIdx[2][:, None] * voxelSize[2] + voxelSize[2] / 2 + point_cloud_range[2]), dim=1)
        norm_occ_coords = (occ_coords - torch.as_tensor([self.x_bound_seg[0], self.y_bound_seg[0], self.z_bound_seg[0]])) \
                          / torch.as_tensor([self.x_bound_seg[1] - self.x_bound_seg[0],
                                             self.y_bound_seg[1] - self.y_bound_seg[0],
                                             self.z_bound_seg[1] - self.z_bound_seg[0]])
        # norm_pts = norm_pts[None, None, None, :, :]
        norm_occ_coords = norm_occ_coords * 2. - 1.
        # valid_coords = (norm_occ_coords[..., 0] >= -1.) & (norm_occ_coords[..., 0] <= 1.) & \
        #                 (norm_occ_coords[..., 1] >= -1.) & (norm_pts[..., 1] <= 1.) & \
        #                 (norm_occ_coords[..., 2] >= -1.) & (norm_occ_coords[..., 2] <= 1.)
        return norm_occ_coords.reshape(200, 200, 16, 3)

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_pixel(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from ego coord to camera coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points camera coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        sensor2ego_mat = sensor2ego_mat
        points = self.voxel_coords
        # paddings = torch.ones_like(points[..., 0:1], device=points.device)
        # points = torch.cat((points, paddings), dim=-1)
        if bda_mat is not None:
            bda_mat = bda_mat
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = bda_mat.inverse().matmul(points.unsqueeze(-1))
        else:
            points = points.unsqueeze(-1)

        # ego_to_cam
        combine = intrin_mat.matmul(torch.inverse(sensor2ego_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        # combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)).float()
        # points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).inverse().matmul(points)
        normalizer = points[:, :, :, :, :, 2:3]
        eps = 1e-6
        points = torch.cat((points[..., :2, :] / torch.clamp(normalizer, min=eps), points[..., 2:, :]), dim=5)
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.matmul(points).squeeze(-1)
        return points[..., :3]

    def volume_rendering_from_multiple_views(self, geom_xyz, density_feature, semantic_logits, voxel_features, rgb):
        """volume rendering from camera and bev views"""
        batch_size, num_cams, d, h, w, _ = geom_xyz.shape
        # density = self.density(density_feature)
        # temp_fusion_features = torch.cat([density, semantic_logits, rgb, voxel_features], dim=1)
        temp_fusion_features = torch.cat([density_feature, semantic_logits, rgb, voxel_features], dim=1)
        norm_geom_xyz = (geom_xyz[:, :, :-1, :, :] - torch.as_tensor(
            [self.x_bound_seg[0], self.y_bound_seg[0], self.z_bound_seg[0]],
            device=self.camera_mids.device)) / torch.as_tensor(
            [self.x_bound_seg[1] - self.x_bound_seg[0],
             self.y_bound_seg[1] - self.y_bound_seg[0],
             self.z_bound_seg[1] - self.z_bound_seg[0]],
            device=self.camera_mids.device)
        norm_geom_xyz = norm_geom_xyz * 2. - 1.
        geom_valid_mask = (norm_geom_xyz[..., 0] >= -1.) & (norm_geom_xyz[..., 0] <= 1.) & \
                          (norm_geom_xyz[..., 1] >= -1.) & (norm_geom_xyz[..., 1] <= 1.) & \
                          (norm_geom_xyz[..., 2] >= -1.) & (norm_geom_xyz[..., 2] <= 1.)
        norm_output_coords = (self.output_coords[..., :3] - torch.as_tensor([self.x_bound_seg[0],
                                                                             self.y_bound_seg[0],
                                                                             self.z_bound_seg[0]],
                                                                            device=self.camera_mids.device)) / \
                             torch.as_tensor(
            [self.x_bound_seg[1] - self.x_bound_seg[0],
             self.y_bound_seg[1] - self.y_bound_seg[0],
             self.z_bound_seg[1] - self.z_bound_seg[0]],
            device=self.camera_mids.device)
        norm_output_coords = norm_output_coords * 2. - 1.
        norm_output_coords = norm_output_coords[None, ...].expand(batch_size, *norm_output_coords.shape)
        frustum_features = F.grid_sample(temp_fusion_features, norm_geom_xyz.reshape(batch_size, -1, h, w, 3), align_corners=True)
        frustum_features = frustum_features.reshape(batch_size, -1, num_cams, d - 1, h, w).permute(0, 2, 1, 3, 4, 5) * geom_valid_mask.unsqueeze(2)
        frustum_features = torch.nan_to_num(frustum_features)
        # frustum_density = frustum_features[:, :, :1, ...]
        frustum_density = self.density(frustum_features[:, :, :1, ...])  # 2x6x1x70x16x44
        frustum_seg_logits = frustum_features[:, :, 1:self.num_classes+1, ...]  # 2x6x17x70x16x44
        frustum_rgb = frustum_features[:, :, self.num_classes+1:self.num_classes+4, ...]  # 2x6x3x70x16x44
        frustum_delta = torch.norm(geom_xyz[:, :, 1:, :, :, :] - geom_xyz[:, :, :-1, :, :, :], dim=-1)  # 2x6x70x16x44

        # 2x6x17x70x16x44
        frustum_density_delta = frustum_density * frustum_delta.unsqueeze(2)
        frustum_alpha = 1 - torch.exp(-frustum_density_delta)
        frustum_trans = torch.exp(-torch.cat(
            [torch.zeros_like(frustum_density_delta[:, :, :, :1, :, :]),
             torch.cumsum(frustum_density_delta[:, :, :, :-1, :, :], dim=3)], dim=3))
        frustum_weights = frustum_alpha * frustum_trans
        frustum_acc = frustum_weights.sum(dim=3)
        bg_depth = (1 - frustum_acc) * self.d_bound[1]
        # bg_rgb = (1 - frustum_acc)[..., None] * torch.tensor([1.0, 1.0, 1.0], device=frustum_acc.device).float().permute(0, 1, 4, 2, 3)
        rgb_preds = torch.sum(frustum_weights * frustum_rgb, dim=3)  # + bg_rgb
        seg_logits_preds = torch.sum(frustum_weights * frustum_seg_logits, dim=3)
        depth_preds = (frustum_weights * self.camera_mids[None, None, None, :, None, None]).sum(dim=3) + bg_depth

        voxel_features = F.grid_sample(temp_fusion_features, norm_output_coords, align_corners=True)
        voxel_features = torch.flip(voxel_features, dims=[2])  # reverse z-axis for bird's-eye-view
        # voxel_density = voxel_features[:, :1, ...]
        voxel_density = self.density(voxel_features[:, :1, ...])  # 2x1x10x128x128
        voxel_seg_logits = voxel_features[:, 1:self.num_classes+1, ...]  # 2x17x10x128x128
        voxel_rgb = voxel_features[:, self.num_classes+1:self.num_classes+4, ...]  # 2x3x10x128x128
        voxel_output = voxel_features[:, self.num_classes+4:, ...]  # 2x32x10x128x128
        if self.cat_seg:
            voxel_output = torch.cat((voxel_output, voxel_seg_logits), dim=1)
        voxel_delta = torch.ones_like(voxel_density) * self.z_bound_det[2]

        voxel_density_delta = voxel_density * voxel_delta
        voxel_alpha = 1 - torch.exp(-voxel_density_delta)
        voxel_trans = torch.exp(-torch.cat(
            [torch.zeros_like(voxel_density_delta[:, :, :1, :, :]),
             torch.cumsum(voxel_density_delta[:, :, :-1, :, :], dim=2)], dim=2))
        voxel_weights = voxel_alpha * voxel_trans
        bev_rgb_preds = torch.sum(voxel_weights * voxel_rgb, dim=2)
        bev_seg_logits_preds = torch.sum(voxel_weights * voxel_seg_logits, dim=2)
        bev_height_preds = (voxel_weights * self.bev_mids[None, None, :, None, None]).sum(dim=2)
        return (rgb_preds, seg_logits_preds, depth_preds,
                bev_rgb_preds,
                bev_seg_logits_preds,
                bev_height_preds,
                voxel_density,
                voxel_output)

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        neck_feats = self.img_neck(backbone_feats)
        img_feats = neck_feats[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def get_voxel_feats(self, img_feats, sweep_index, mats_dict, clamp_extreme=True):
        batch_size, num_cams, num_channels, h, w = img_feats.shape
        with autocast(enabled=False):
            voxel_pixel = self.get_pixel(
                mats_dict['sensor2ego_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict.get('bda_mat', None),
            )
        # voxel_pixel = voxel_pixel.to(self.camera_mids.device)
        x, y, z = voxel_pixel[..., 0], voxel_pixel[..., 1], voxel_pixel[..., 2]
        x_valid = (x > -0.5).bool() & (x < float(self.final_dim[1] - 0.5)).bool()
        y_valid = (y > -0.5).bool() & (y < float(self.final_dim[0] - 0.5)).bool()
        z_valid = (z > 0.).bool()
        valid = (x_valid & y_valid & z_valid).float()
        ###################
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # colors = ['orange', 'green', 'blue', 'red', 'yellow', 'pink']
        # for i in range(6):
        #     val = valid[0, i, ...]
        #     x = voxel_coords[..., 0][val.bool()]
        #     y = voxel_coords[..., 1][val.bool()]
        #     z = voxel_coords[..., 2][val.bool()]
        #     ax.scatter3D(x, y, z, s=0.01, c=colors[i])
        # ax.set_xlabel('x')
        # ax.set_xlim(-50., 50.)
        # ax.set_ylabel('y')
        # ax.set_ylim(-50., 50.)
        # ax.set_zlabel('z')
        # plt.show()
        # plt.close()
        ###################
        norm_x = 2.0 * (x / float(self.final_dim[1] - 1)) - 1.0
        norm_y = 2.0 * (y / float(self.final_dim[0] - 1)) - 1.0
        if clamp_extreme:
            norm_x = torch.clamp(norm_x, min=-2.0, max=2.0)
            norm_y = torch.clamp(norm_y, min=-2.0, max=2.0)
        norm_xyz = torch.stack([norm_x, norm_y, torch.zeros_like(norm_x)], dim=-1).reshape(-1, self.vZ, self.vY,
                                                                                           self.vX, 3)
        voxel_features = F.grid_sample(img_feats.reshape(-1, num_channels, 1, h, w), norm_xyz, align_corners=False)
        voxel_features = voxel_features.reshape(batch_size, num_cams, num_channels, self.vZ, self.vY, self.vX) * valid.unsqueeze(2)
        voxel_mask = (torch.abs(voxel_features) > 0).float()
        numer = torch.sum(voxel_features, dim=1)

        denom = torch.sum(voxel_mask, dim=1) + 1e-6
        mean = numer / denom

        return mean

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              inrange_pts=None,):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
                is_rendering (bool, optional): Whether to return rendered predicts.
                    Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
        img_width = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...].reshape(batch_size * num_cams, -1, img_feats.shape[-2], img_feats.shape[-1])
        low_channel_source_features = self.channel_lower(source_features).reshape(batch_size, num_cams, -1,
                                                                                  img_feats.shape[-2],
                                                                                  img_feats.shape[-1])
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )

        # visualize_geomxyz(geom_xyz, mode='bev')
        # visualize_geomxyz(geom_xyz, mode='front')
        voxel_features = self.get_voxel_feats(low_channel_source_features, sweep_index, mats_dict)  # 1,80,20,256,256
        # plt.imshow(voxel_features.sum(1).sum(1)[0].detach().cpu().numpy())
        # plt.show()
        # plt.close()
        if self.cat_pos:
            norm_voxel_coords = self.norm_voxel_coords.permute(3, 0, 1, 2)[None, ...].repeat(batch_size, 1, 1, 1, 1)
            voxel_features = torch.cat([voxel_features, norm_voxel_coords], dim=1)
        base_features = self.base_conv(voxel_features)
        # raw density features or the signed distance function (SDF)
        density_feature = self.density_conv(base_features)
        semantic_logits = self.seg_conv(base_features)
        voxel_features = self.feature_conv(base_features)
        rgb = self.rgb_conv(voxel_features)

        pts_logits_batch = list()
        pts_sdf_batch = list()
        if inrange_pts is not None:
            for i in range(batch_size):
                # assert batch_size == 1, 'SDF loss only implemented for batch size = 1 now!'
                norm_pts = (inrange_pts[i] - torch.as_tensor([self.x_bound_seg[0], self.y_bound_seg[0], self.z_bound_seg[0]],
                                                             device=self.camera_mids.device)) / torch.as_tensor(
                    [self.x_bound_seg[1] - self.x_bound_seg[0], self.y_bound_seg[1] - self.y_bound_seg[0],
                     self.z_bound_seg[1] - self.z_bound_seg[0]], device=self.camera_mids.device)
                norm_pts = norm_pts[None, None, None, :, :]
                norm_pts = norm_pts * 2. - 1.
                valid_pts = (norm_pts[..., 0] >= -1.) & (norm_pts[..., 0] <= 1.) & \
                            (norm_pts[..., 1] >= -1.) & (norm_pts[..., 1] <= 1.) & \
                            (norm_pts[..., 2] >= -1.) & (norm_pts[..., 2] <= 1.)
                pts_logits = F.grid_sample(semantic_logits[[i], ...], norm_pts,  padding_mode='border', align_corners=True)
                # pts_logits = pts_logits * valid_pts.unsqueeze(1)
                pts_logits_batch.append(pts_logits[0, :, 0, 0, :].permute(1, 0))
                if self.density_mode == 'sdf':
                    pts_sdf = F.grid_sample(density_feature[[i], ...], norm_pts, align_corners=True)
                    pts_sdf = pts_sdf.squeeze(1) * valid_pts
                    pts_sdf_batch.append(pts_sdf[0, 0, 0, :])
        # occupancy prediction
        occ_logits = F.grid_sample(semantic_logits, self.norm_occ_coords[None, ...].expand(batch_size, *self.norm_occ_coords.shape),  padding_mode='border', align_corners=True)
        occ_density = F.grid_sample(self.density(density_feature), self.norm_occ_coords[None, ...].expand(batch_size, *self.norm_occ_coords.shape), align_corners=True)
        geom_xyz = torch.nan_to_num(geom_xyz, -1e3)
        rgb_preds, seg_logits_preds, depth_preds, bev_rgb_preds, bev_seg_logits_preds, bev_height_preds, bev_density, voxel_output = \
            self.volume_rendering_from_multiple_views(geom_xyz, density_feature, semantic_logits, voxel_features, rgb)
        # plt.imshow(voxel_output.sum(1).sum(1)[0].detach().cpu().numpy())
        # plt.show()
        # plt.close()
        # plt.imshow(bev_rgb_preds[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()
        # plt.close()
        # plt.imshow(bev_height_preds[0][0].detach().cpu().numpy())
        # plt.show()
        # plt.close()
        rgb_preds = self.upsample2d(rgb_preds.reshape(batch_size * num_cams, -1, self.fH, self.fW)).reshape(batch_size,
                                                                                                            num_cams,
                                                                                                            -1,
                                                                                                            self.fH * self.upsample_factor,
                                                                                                            self.fW * self.upsample_factor)
        seg_logits_preds = self.upsample2d(
            seg_logits_preds.reshape(batch_size * num_cams, -1, self.fH, self.fW)).reshape(batch_size, num_cams, -1,
                                                                                           self.fH * self.upsample_factor,
                                                                                           self.fW * self.upsample_factor)
        depth_preds = self.upsample2d(depth_preds.reshape(batch_size * num_cams, -1, self.fH, self.fW)).reshape(
            batch_size, num_cams, -1, self.fH * self.upsample_factor, self.fW * self.upsample_factor)
        if self.density_mode == 'sdf':
            voxel_output = voxel_output * bev_density.tanh()
        else:
            voxel_output = voxel_output * bev_density
        voxel_output_features = self.voxel_output(
            voxel_output.reshape(batch_size, -1, voxel_output.shape[-2], voxel_output.shape[-1])).float()
        return (voxel_output_features.contiguous(),
                rgb_preds,
                seg_logits_preds,
                depth_preds,
                bev_rgb_preds,
                bev_seg_logits_preds,
                bev_height_preds,
                bev_density,
                pts_logits_batch,
                pts_sdf_batch,
                occ_logits.permute(0, 2, 3, 4, 1),
                occ_density.permute(0, 2, 3, 4, 1).tanh(),
                )

    def forward(self,
                sweep_imgs,
                mats_dict,
                inrange_pts=None,
                timestamps=None):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            inrange_pts(Tensor): query or supervise point clouds.
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
        img_width = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            inrange_pts=inrange_pts
        )
        if num_sweeps == 1:
            return key_frame_res
        else:
            raise NotImplementedError