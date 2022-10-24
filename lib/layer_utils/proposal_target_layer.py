# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from lib.utils.cython_bbox import bbox_overlaps

from lib.config import config as cfg
from lib.utils.bbox_transform import bbox_transform, bbox_transform_r


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes,gt_boxes_ro, _num_classes,itera,rois_from_att):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    根据gt，对rpn产生的proposal打上分类标签以及计算回归的偏差
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    # Include ground-truth boxes in the set of candidate rois
    # 去除gt_boxes的最后一列（label）,拼接第一列全是0的数组，然后并到all_rois中
    if cfg.FLAGS.proposal_use_gt:
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        # not sure if it a wise appending, but anyway i am not using it
        all_scores = np.vstack((all_scores, zeros))

    num_images = 1
    # 每一张图片允许的roi区域batch
    rois_per_image = cfg.FLAGS.batch_size / num_images
    # 计算每一张图片的batch个roi中前景的数量
    fg_rois_per_image = np.round(cfg.FLAGS.proposal_fg_fraction * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    '''    bbox_target_data[label,tx,ty,tw,th]    
    labels[0.5,0.6,0.5,0.7,0.9,....,0,0,0,0,0,0,0,....]    
    rois是对all_rois重新排列了一下，前景在前，背景在后    
    roi_scores是对all_scores重新排列了一下，前景在前，背景在后    
    '''
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights,bbox_targets_r, bbox_inside_weights_r,bbox_target_data_ro, fg_box, no_zero_num= _sample_rois(
        all_rois, all_scores, gt_boxes, gt_boxes_ro,fg_rois_per_image,
        rois_per_image, _num_classes,itera,rois_from_att)

    rois = rois.reshape(-1, 5)  # iou>0.1的前景+背景rois
    # print(bbox_target_data_ro.shape)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_targets_r = bbox_targets_r.reshape(-1, _num_classes * 5)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_inside_weights_r = bbox_inside_weights_r.reshape(-1, _num_classes * 5)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    bbox_outside_weights_r = np.array(bbox_inside_weights_r > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights,bbox_targets_r,bbox_inside_weights_r,bbox_outside_weights_r,bbox_target_data_ro, fg_box,no_zero_num


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    # 获取bbox_target_data所有的类别
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    #找出所有前景下标
    inds = np.where(clss > 0)[0]
    # print('inds',inds)
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  # 变成[tx0,ty0,tw0,th0,0,0,....],[0,0,0,0,tx1,ty1,tw1,th1,...]
        bbox_inside_weights[ind, start:end] = cfg.FLAGS2["bbox_inside_weights"]
    return bbox_targets, bbox_inside_weights

def _get_bbox_regression_labels_r(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th,ttheta)

    This function expands those targets into the 5-of-5*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 5K blob of regression targets
        bbox_inside_weights (ndarray): N x 5K blob of loss weights
    """
    # 获取bbox_target_data所有的类别
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 5 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    #找出所有前景下标
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(5 * cls)
        end = start + 5
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]  # 变成[tx0,ty0,tw0,th0,0,0,....],[0,0,0,0,tx1,ty1,tw1,th1,...]
        bbox_inside_weights[ind, start:end] = cfg.FLAGS2["bbox_inside_weights_ro"]
        # 前非零label个bbox——t不为0，其余都为0
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.FLAGS.bbox_normalize_targets_precomputed:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.FLAGS2["bbox_normalize_means"]))
                   / np.array(cfg.FLAGS2["bbox_normalize_stds"]))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _compute_targets_r(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    targets = bbox_transform_r(ex_rois, gt_rois)
    if cfg.FLAGS.bbox_normalize_targets_precomputed:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.FLAGS2["bbox_normalize_means_ro"]))
                   / np.array(cfg.FLAGS2["bbox_normalize_stds_ro"]))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes,gt_boxes_ro, fg_rois_per_image, rois_per_image, num_classes,itera,rois_from_att):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    # 计算rois和gt_boxes的overlaps
    # roi格式(0, x1, y1, x2, y2)，gt_box格式(x,y,x,y,label)
    # 只取对应的xyxy
    # print(all_rois.shape)
    if cfg.FLAGS.use_roi_from_att:
        all_rois = np.vstack((all_rois, rois_from_att))
        num = rois_from_att.shape[0]
        rois_scor_from_att = np.ones(num)
        rois_scor_from_att=rois_scor_from_att.astype(np.float32)
        rois_scor_from_att = np.reshape(rois_scor_from_att, [-1, 1])
        all_scores = np.vstack((all_scores, rois_scor_from_att))
    # print(num)
    # print(rois_scor_from_att.shape)

    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    # 返回每一行最大那一列的下标，也就是rois对应overlap最大的gt_box的索引
    gt_assignment = overlaps.argmax(axis=1)
    # print(gt_assignment)
    # 一样，只不过返回的是值
    max_overlaps = overlaps.max(axis=1)
    # 对应最大gt_box的label
    labels = gt_boxes[gt_assignment, 4]
    # Select foreground RoIs as those with >= FG_THRESH overlap
    # max_overlaps>=0.5的记录为前景,返回的也是下标
    # k = int((itera+15000) / 15000) * 0.05
    # k = 0
    fg_inds = np.where(max_overlaps >= cfg.FLAGS.roi_fg_threshold)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    # max_overlaps在[0.1,0.5]记录为背景

    bg_inds = np.where((max_overlaps < cfg.FLAGS.roi_bg_threshold_high) &
                       (max_overlaps >= cfg.FLAGS.roi_bg_threshold_low))[0]

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    if fg_inds.size > 0 and bg_inds.size > 0:
        # 下面的意思就是如果样本很多，则随机采样去除一些
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        raise Exception()

    # The indices that we're selecting (both fg and bg)
    # 将前景背景的下标拼起来
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    # 提取对应的labels，相当于重新排了一下，前景在前，背景在后
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    # 然后把背景的全部赋值为0
    labels[int(fg_rois_per_image):] = 0
    # 下两个提取对应的roi和得分

    rois = all_rois[keep_inds]
    # print(rois.shape)
    roi_scores = all_scores[keep_inds]
    # print(roi_scores.shape)


    # print(keep_inds)
    # print(labels)
    # print(gt_boxes[gt_assignment[keep_inds], :4])

    # 用_compute_targets函数把xyxy坐标转换成delta坐标 ，也就是计算偏移量
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    bbox_target_data_ro = _compute_targets_r(
        rois[:, 1:5], gt_boxes_ro[gt_assignment[keep_inds], :-1], labels)
    # print(bbox_target_data)
    # 最后bbox_target_data格式[[label,tx,ty,tw,th],[label,tx,ty,tw,th]]
    fg_box_indx=np.where(labels > 0)
    fg_box = gt_boxes_ro[gt_assignment[keep_inds], :][fg_box_indx]
    # print(fg_box)
    no_zero_num=fg_box_indx[0].size
    # 根据bbox_target_data偏移量，计算出回归的label
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)
    bbox_targets_r, bbox_inside_weights_r = \
        _get_bbox_regression_labels_r(bbox_target_data_ro, num_classes)
    #  bbox_targets = [tx0,ty0,tw0,th0,0,0,....],[0,0,0,0,tx1,ty1,tw1,th1,...]
    # print()

    return labels,rois, roi_scores, bbox_targets, bbox_inside_weights,bbox_targets_r, bbox_inside_weights_r,bbox_target_data_ro,fg_box,no_zero_num
