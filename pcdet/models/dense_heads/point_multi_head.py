import torch
import torch.nn as nn
import numpy as np
from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from visual_utils import debug_utils as V
from eval_utils import eval_utils_point as E

class PointMultiHead(PointHeadTemplate):
    """ Point-based head for predicting the 
            1. intra-object part locations, 
            2. object/non-object segmentation, 
            3. drivable road semantics, 
            4. ground semantics,
            5. ground heights,
            6. bbox (optional)
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.part_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.PART_FC,
            input_channels=input_channels,
            output_channels=3
        )
        self.drivable_cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.DRIVE_FC,
            input_channels=input_channels,
            output_channels=1
        )
        self.ground_cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.GROUND_FC,
            input_channels=input_channels,
            output_channels=1
        )
        #Currently using absolute value prediction, because heights are centered on +-0m 
        self.ground_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.GROUND_HEIGHT_FC,
            input_channels=input_channels,
            output_channels=1
        )        

        self.target_cfg = self.model_cfg.TARGET_CONFIG
        if self.target_cfg.get('BOX_CODER', None) is not None:
            self.box_coder = getattr(box_coder_utils, self.target_cfg.BOX_CODER)(
                **self.target_cfg.BOX_CODER_CONFIG
            )
            self.box_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.REG_FC,
                input_channels=input_channels,
                output_channels=self.box_coder.code_size
            )
        else:
            self.box_layers = None

        if self.target_cfg.get('USE_TASK_UNCERTAINTY', False):
            # t = log(sigma^2)
            self.task_uncertainty_weights = nn.Parameter(torch.rand(self.target_cfg.NUM_TASKS))
        else:
            self.task_uncertainty_weights = None

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
                
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
            point_drivable_cls_labels: (N1 + N2 + N3 + ...), long type, 0: background, -1: ignored, 1: foreground
            point_ground_cls_labels: (N1 + N2 + N3 + ...), long type, 0: background, -1: ignored, 1: foreground
            point_ground_reg_labels: (N1 + N2 + N3 + ...), float type, ground height estimation 
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        
        # preparing point labels: point_labels: (N1 + N2 + N3 + ..., 4) [drivable_area_label, ground_label, ground_height]
        voxels_drivable_cls_labels = input_dict['voxels_drivable_cls_labels']
        voxels_drivable_cls_labels[voxels_drivable_cls_labels>0] = 1 # if a point in a voxel is on drivable road, all voxel is set drivable
        voxels_ground_cls_labels = input_dict['voxels_ground_cls_labels']
        voxels_ground_cls_labels[voxels_ground_cls_labels>0] = 1 # if a point in a voxel is on the ground, all voxel is on the ground
        voxels_ground_reg_labels = input_dict['voxels_ground_reg_labels']

        point_labels = torch.cat((voxels_drivable_cls_labels, voxels_ground_cls_labels, voxels_ground_reg_labels), dim=-1) 

        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        assert point_labels.shape.__len__() in [2], 'point_labels.shape=%s' % str(point_labels.shape)
        assert point_labels.shape[1] == 3, 'point_labels.shape=%s' % str(point_labels.shape)
        assert point_labels.shape[0] == point_coords.shape[0], 'point_labels and point_coords do not have same size! TODO: implement down-sampled voxel feature!'


        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        kwargs = {
        'ret_box_labels': self.box_layers is not None,
        'ret_part_labels': True, 
        'ret_drivable_area_labels': True, 
        'ret_ground_cls_labels': True, 
        'ret_ground_reg_labels': True, 
        }

        targets_dict = self.assign_stack_targets_multi_labels(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            point_labels=point_labels, set_ignore_flag=True, use_ball_constraint=False, **kwargs
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)

        point_loss_drivable_cls, tb_dict = self.get_drivable_cls_layer_loss(tb_dict)
        point_loss_ground_cls, tb_dict = self.get_ground_cls_layer_loss(tb_dict)
        point_loss_ground_reg, tb_dict = self.get_ground_reg_layer_loss(tb_dict)


        if self.target_cfg.get('USE_TASK_UNCERTAINTY', False):
            multi_loss = torch.stack((point_loss_cls, 
                                   point_loss_part,
                                   point_loss_drivable_cls,
                                   point_loss_ground_cls,
                                   point_loss_ground_reg,
                                   ), dim=-1)
            point_loss, tb_dict = self.get_uncertainty_aware_multitask_loss(multi_loss, tb_dict)

        else:
            point_loss = point_loss_cls\
                         + point_loss_part\
                         + point_loss_drivable_cls\
                         + point_loss_ground_cls\
                         + point_loss_ground_reg

        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_part_preds = self.part_reg_layers(point_features)
        point_drivable_cls_preds = self.drivable_cls_layers(point_features)
        point_ground_cls_preds = self.ground_cls_layers(point_features)
        point_ground_reg_preds = self.ground_reg_layers(point_features)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            'point_part_preds': point_part_preds,
            'point_drivable_cls_preds': point_drivable_cls_preds,
            'point_ground_cls_preds': point_ground_cls_preds,
            'point_ground_reg_preds': point_ground_reg_preds
        }
        if self.box_layers is not None:
            point_box_preds = self.box_layers(point_features)
            ret_dict['point_box_preds'] = point_box_preds
        if self.task_uncertainty_weights is not None:
            ret_dict['task_uncertainty_weights'] = self.task_uncertainty_weights

        point_cls_scores = torch.sigmoid(point_cls_preds)
        point_part_offset = torch.sigmoid(point_part_preds)
        point_drivable_cls_scores = torch.sigmoid(point_drivable_cls_preds)
        point_ground_cls_scores = torch.sigmoid(point_ground_cls_preds)

        batch_dict['point_cls_scores'] = point_cls_scores
        batch_dict['point_drivable_cls_scores'] = point_drivable_cls_scores
        batch_dict['point_ground_cls_scores'] = point_ground_cls_scores
        batch_dict['point_part_offset'] = point_part_offset
        batch_dict['point_ground_heights'] = point_ground_reg_preds

        targets_dict = None
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_part_labels'] = targets_dict.get('point_part_labels')
            ret_dict['point_box_labels'] = targets_dict.get('point_box_labels')
            
            ret_dict['point_drivable_cls_labels'] = targets_dict.get('point_drivable_cls_labels')
            ret_dict['point_ground_cls_labels'] = targets_dict.get('point_ground_cls_labels')
            ret_dict['point_ground_reg_labels'] = targets_dict.get('point_ground_reg_labels')

        if self.box_layers is not None and (not self.training or self.predict_boxes_when_training):
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=ret_dict['point_box_preds']
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict
        return batch_dict
        