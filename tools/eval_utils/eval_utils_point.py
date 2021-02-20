import pickle
import time
import tqdm
import os

import numpy as np
import torch
from sklearn import metrics as M
from sklearn.calibration import calibration_curve
from scipy.stats import entropy
import matplotlib
from matplotlib import pyplot as plt

from pcdet.utils import common_utils
    
POSITIVE_THRESHOLD = 0.5
DISTANCE_RANGE = [0,30,50,70]
VISUALIZE = False

def save_prediction_point_argoverse(batch_dict, save_gt=False):
    goal_dir = 'argoverse_results'
    pred_dir = os.path.join(goal_dir, 'prediction')
    label_dir = os.path.join(goal_dir, 'label')
    point_dir = os.path.join(goal_dir, 'point')
    map_file = os.path.join(goal_dir, 'val.txt')
    if not os.path.exists(goal_dir): 
        os.mkdir(goal_dir)
    if not os.path.exists(pred_dir): 
        os.mkdir(pred_dir)
    if not os.path.exists(label_dir): 
        os.mkdir(label_dir)
    if not os.path.exists(point_dir): 
        os.mkdir(point_dir)

    points = batch_dict['point_coords'].detach().cpu().numpy()
    point_cls_scores = batch_dict.get('point_cls_scores').detach().cpu().numpy()
    if point_cls_scores.shape[1]>1: 
        # multi-class prediction, transfer to fore-background class
        point_cls_scores = np.max(point_cls_scores, axis=1)

    point_cls_scores = point_cls_scores.reshape(-1,1)
    point_part_offset = batch_dict.get('point_part_offset').detach().cpu().numpy()
    point_drivable_cls_scores = batch_dict.get('point_drivable_cls_scores').detach().cpu().numpy()
    point_ground_cls_scores = batch_dict.get('point_ground_cls_scores').detach().cpu().numpy()
    point_ground_cls_scores = point_ground_cls_scores.reshape(-1,1)
    point_ground_heights = batch_dict.get('point_ground_heights').detach().cpu().numpy()
    frame_id = batch_dict['frame_id']

    batch_size = batch_dict['batch_size']
    bs_idx = points[:,0]

    if 'targets_dict' in batch_dict is not None and save_gt:
        targets_dict = batch_dict['targets_dict']
        point_drivable_cls_labels = targets_dict.get('point_drivable_cls_labels').detach().cpu().numpy()
        point_drivable_cls_labels = point_drivable_cls_labels.reshape(-1,1)
        point_ground_cls_labels = targets_dict.get('point_ground_cls_labels').detach().cpu().numpy()
        point_ground_cls_labels.reshape(-1,1)
        point_cls_labels = targets_dict.get('point_cls_labels').detach().cpu().numpy()
        point_cls_labels = point_cls_labels.reshape(-1,1)
        point_ground_reg_labels = targets_dict.get('point_ground_reg_labels').detach().cpu().numpy()
        point_part_labels = targets_dict.get('point_part_labels').detach().cpu().numpy()

    # save prediction and ground truth data
    for k in range(batch_size):
        frame = frame_id[k]
        bs_mask = (bs_idx == k)
        points_bs = points[bs_mask,1:].astype('float32')
        
        if targets_dict is not None and save_gt: 
            point_drivable_cls_labels_bs = point_drivable_cls_labels[bs_mask].reshape(-1,1)
            point_ground_cls_labels_bs = point_ground_cls_labels[bs_mask].reshape(-1,1)
            point_ground_reg_labels_bs = point_ground_reg_labels[bs_mask].reshape(-1,1)
            point_cls_labels_bs = point_cls_labels[bs_mask].reshape(-1,1)
            point_part_labels_bs = point_part_labels[bs_mask].reshape(-1,3)

            point_labels = np.concatenate([point_ground_reg_labels_bs,
                           point_ground_cls_labels_bs,
                           point_drivable_cls_labels_bs,
                           point_cls_labels_bs,
                           point_part_labels_bs,
                           ], axis=1).astype('float32')
            point_labels.tofile(os.path.join(label_dir, frame+'.bin'))#[ground_height, ground_cls, drivable_area_cls, fore_background_cls, point_part_labels] 

        point_drivable_cls_scores_bs = point_drivable_cls_scores[bs_mask].reshape(-1,1)
        point_ground_cls_scores_bs = point_ground_cls_scores[bs_mask].reshape(-1,1)
        point_ground_heights_bs = point_ground_heights[bs_mask].reshape(-1,1)
        point_cls_scores_bs = point_cls_scores[bs_mask].reshape(-1,1) 
        point_part_offset_bs = point_part_offset[bs_mask].reshape(-1,3)
        
        point_preds = np.concatenate([point_ground_heights_bs, #1
                                      point_ground_cls_scores_bs, #1
                                      point_drivable_cls_scores_bs, #1
                                      point_cls_scores_bs, #1
                                      point_part_offset_bs, #3
                                      ], axis=1).astype('float32')
        
        point_preds.tofile(os.path.join(pred_dir, frame+'.bin'))

        points_bs.tofile(os.path.join(point_dir, frame+'.bin'))#[x,y,z] in lidar coordinate system 

        #save index
        with open(os.path.join(goal_dir, 'val.txt'),'a') as f:
            f.write(frame + ' \n')


def binary_classification_statistics(y_true, y_score):
    y_true[y_true>0] = 1
    y_pred = np.zeros([y_score.shape[0], 1])
    y_pred[y_score>POSITIVE_THRESHOLD] = 1
    ap = M.average_precision_score(y_true, y_score)
    iou =M.jaccard_score(y_true, y_pred)
    accu = M.accuracy_score(y_true, y_pred)
    return {'ap':ap, 'iou': iou, 'accu': accu}

def binary_scoring_statistics(y_true, y_score):
    ap = M.average_precision_score(y_true, y_score)
    return {'ap':ap}

def regression_statistics(y_true, y_pred, dist=None):
    ret = {}

    if (len(y_true.shape)<2) or (len(y_pred.shape)<2):
        y_true = y_true.reshape(-1,1)
        y_pred = y_pred.reshape(-1,1)
    rmse = M.mean_squared_error(y_true, y_pred, squared=False)*100 #cm
    mae = M.mean_absolute_error(y_true, y_pred)*100 #cm
    ret['rmse'] = rmse
    ret['mae'] = mae

    if dist is not None:
        for i in range(len(DISTANCE_RANGE)-1):
            d_min = DISTANCE_RANGE[i]
            d_max = DISTANCE_RANGE[i+1]
            pos_mask = np.logical_and(dist>=d_min, dist<=d_max)
            y_true_i = y_true[pos_mask]
            y_pred_i = y_pred[pos_mask]
            rmse_i = M.mean_squared_error(y_true_i, y_pred_i, squared=False)*100 #cm
            mae_i = M.mean_absolute_error(y_true_i, y_pred_i)*100 #cm
            ret['rmse_'+str(d_min)+'_'+str(d_max)] = rmse_i 
            ret['mae_'+str(d_min)+'_'+str(d_max)] = mae_i 
    return ret

def inverse_transform_point_cloud(point_cloud, ego_vehicle_matrix):
    # <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
    """Undo the translation and then the rotation (Inverse SE(3) transformation)."""
    translation = ego_vehicle_matrix[:3, 3]
    rotation = ego_vehicle_matrix[:3, :3]
    point_cloud = point_cloud.copy()
    point_cloud -= translation
    return point_cloud.dot(rotation)


def simple_plane_estimation(ground_height, ego_vehicle_matrix, point):
    """Assuming a plane defined by the ego-vehicle rotation and translation
        Input:
            ground_height: ego_vehicle height in city coordinate system
            ego_vehicle_matrix: rotation and translation matrix
            point: (N,3) ego-vehicle coordinate 
        Output:
            y_pred
    """
    rotation = ego_vehicle_matrix[:3,:3]
    translation = ego_vehicle_matrix[:3,3]
    ground_pos_city_coords = np.array([translation[0], translation[1], ground_height])

    # transform to ego-vehicle coordinate
    ground_pos_ego_coords = inverse_transform_point_cloud(ground_pos_city_coords, ego_vehicle_matrix)
    ground_height_ego_coords = ground_pos_ego_coords[2]

    y_pred = ground_height_ego_coords * np.ones([point.shape[0], 1]) #assuming a horizontal ground plane in ego-vehicle coordinate

    return y_pred


def plot(y_true, y_pred):
    alpha=0.3
    size=0.1
    valid_mask1 = y_true[:,3]>=0 #-1 is invalid class
    valid_mask2 = y_true[:,3]>0 #only for foreground objects
    #[1:ground_height, 1:ground_cls, 1:drivable_area_cls, 1:fore_background_cls, 3:point_part_labels]
    y_error1 = (y_true[valid_mask1] - y_pred[valid_mask1])**2
    y_error2 = (y_true[valid_mask2] - y_pred[valid_mask2])**2

    plt.figure()
    plt.subplot(231)
    plt.scatter(y_error1[:,1], y_error1[:,2], alpha=alpha, s=size)
    plt.xlabel('BS ground_cls')
    plt.ylabel('BS drivable_area_cls')
    plt.tight_layout()

    plt.subplot(232)
    plt.scatter(y_error1[:,0], y_error1[:,1], alpha=alpha, s=size)
    plt.xlabel('Error ground_height')
    plt.ylabel('BS ground_cls')
    plt.tight_layout()

    plt.subplot(233)
    plt.scatter(y_error1[:,0], y_error1[:,2], alpha=alpha, s=size)
    plt.xlabel('Error ground_height')
    plt.ylabel('BS drivable_area_cls')
    plt.tight_layout()

    plt.subplot(234)
    plt.scatter(y_error2[:,0], y_error2[:,3], alpha=alpha, s=size)
    plt.xlabel('Error ground_height')
    plt.ylabel('BS fore_background_cls')
    plt.tight_layout()

    plt.subplot(235)
    plt.scatter(y_error2[:,2], y_error2[:,3], alpha=alpha, s=size)
    plt.xlabel('BS drivable_area_cls')
    plt.ylabel('BS fore_background_cls')
    plt.tight_layout()

    plt.subplot(236)
    plt.scatter(y_error2[:,0], np.mean(y_error2[:,4:7],axis=1), alpha=alpha, s=size)
    plt.xlabel('Error ground_height')
    plt.ylabel('Error intra-part')
    plt.tight_layout()

    plt.savefig('error_comparison.png', dpi=150)


def quick_evaluation_point_argoverse(data_dir=''):
    pred_dir = os.path.join(data_dir, 'prediction')
    label_dir = os.path.join(data_dir, 'label')
    point_dir = os.path.join(data_dir, 'point')
    ego_vehicle_pose_dir = os.path.join(data_dir, 'ego_vehicle_pose')
    ego_vehicle_ground_height_dir = os.path.join(data_dir, 'ego_vehicle_ground_height')

    compare_ground_height_algorithm = False
    if os.path.exists(ego_vehicle_pose_dir) and os.path.exists(ego_vehicle_ground_height_dir):
        compare_ground_height_algorithm = True


    val_id_list = [x.strip() for x in open(os.path.join(data_dir, 'val.txt')).readlines()]

    point_all = np.zeros([1,3])
    label_all = np.zeros([1,7])
    pred_all = np.zeros([1,7])

    if compare_ground_height_algorithm:
        ground_height_all = np.zeros([1,1])

    #TODO: currently load all points together to do evalution, which is inefficient!
    for idx in val_id_list:
        point_file = os.path.join(point_dir, '%s.bin'%idx)
        label_file = os.path.join(label_dir, '%s.bin'%idx)
        pred_file = os.path.join(pred_dir, '%s.bin'%idx)

        point = np.fromfile(str(point_file), dtype=np.float32).reshape(-1, 3)
        #[ground_height, ground_cls, drivable_area_cls, fore_background_cls, point_part_labels]
        label = np.fromfile(str(label_file), dtype=np.float32).reshape(-1, 7)
        #[ground_height, ground_cls, drivable_area_cls, fore_background_cls]
        pred = np.fromfile(str(pred_file), dtype=np.float32).reshape(-1, 7)

        point_all = np.append(point_all, point, axis=0)
        label_all = np.append(label_all, label, axis=0)
        pred_all = np.append(pred_all, pred, axis=0)

        if compare_ground_height_algorithm:
            ego_file = os.path.join(ego_vehicle_pose_dir, '%s.txt'%idx)
            ego_height_file = os.path.join(ego_vehicle_ground_height_dir, '%s.txt'%idx)
            ego_vehicle_matrix = np.loadtxt(str(ego_file), dtype=np.float32)
            height = np.loadtxt(str(ego_height_file), dtype=np.float32)
            ground_height = simple_plane_estimation(height, ego_vehicle_matrix, point)
            ground_height_all = np.append(ground_height_all, ground_height, axis=0)

    point_all = point_all[1:]
    dist_all = np.sqrt(point_all[:,0]**2 + point_all[:,1]**2)#horizontal distance
    label_all = label_all[1:]
    pred_all = pred_all[1:]
    if compare_ground_height_algorithm: 
        ground_height_all = ground_height_all[1:]

    ret_dict = {}
    ret1 = binary_classification_statistics(label_all[:,1], pred_all[:,1])
    ret_dict['ground_cls'] = ret1
    
    ret2 = binary_classification_statistics(label_all[:,2], pred_all[:,2])
    ret_dict['drivable_area_cls'] = ret2

    valid_mask = label_all[:,3]>=0 #-1 is invalid class
    ret3 = binary_classification_statistics(label_all[valid_mask,3], pred_all[valid_mask,3])
    ret_dict['foreground_cls'] = ret3

    # evaluate ground height for all points
    ret4 = regression_statistics(label_all[:,0], pred_all[:,0], dist_all)
    ret_dict['ground_height_all'] = ret4

    # evaluate ground heights only for objects
    pos_mask = label_all[:,3]>0
    ret5 = regression_statistics(label_all[pos_mask,0], pred_all[pos_mask,0], dist_all[pos_mask])
    ret_dict['ground_height_objects'] = ret5

    #evaluate part's prediction only for foreground objects
    pos_mask = label_all[:,3]>0
    ret6 = regression_statistics(label_all[pos_mask,4:7], pred_all[pos_mask,4:7], dist_all[pos_mask])
    ret_dict['object_intra_points'] = ret6

    #evaluate ground height plane: simple heuristics
    if compare_ground_height_algorithm:
        ret7 = regression_statistics(label_all[:,0], ground_height_all, dist_all)
        ret_dict['ground_height_all_plane_heuristics'] = ret7
        
        pos_mask = label_all[:,3]>0
        ret8 = regression_statistics(label_all[pos_mask,0], ground_height_all[pos_mask], dist_all[pos_mask])
        ret_dict['ground_height_objects_plane_heuristics'] = ret8

    if VISUALIZE:
        plot(label_all, pred_all)

    return ret_dict 


