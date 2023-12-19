from torch import nn

from src.layers.backbones.base_lss import BaseLSS
from src.layers.heads.bev_depth_head import BEVDepthHead

__all__ = ['LSS']


class LSS(nn.Module):
    """
    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf):
        super(LSS, self).__init__()
        self.backbone = BaseLSS(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)

    def forward(
            self,
            x,
            mats_dict,
            inrange_pts=None,
            timestamps=None,
            lidar_seg=False,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
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
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        x, \
        rgb_preds, seg_logits_preds, depth_preds, \
        bev_rgb_preds, bev_seg_logits_preds, bev_height_preds, bev_density, \
        pts_logits_batch, pts_sdf_batch, occ_logits_batch, occ_density_batch = self.backbone(x, mats_dict, inrange_pts, timestamps)
        if lidar_seg and not self.training:
            return pts_logits_batch, occ_logits_batch, occ_density_batch
        preds = self.head(x)
        return preds, rgb_preds, seg_logits_preds, depth_preds, \
               bev_rgb_preds, bev_seg_logits_preds, bev_height_preds, bev_density, \
               pts_logits_batch, pts_sdf_batch, occ_logits_batch, occ_density_batch

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
