# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms

def add_demension(final_boxes):
    batch_inds = np.zeros((final_boxes.shape[0], 1), dtype=np.float32)
    final_boxes = np.hstack((batch_inds, final_boxes.astype(np.float32, copy=False)))  # (post_nms_topN,5),第一列全为0
    return final_boxes

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    '''
        pre_nms_topN: 在NMS处理之前，分数在前面的rois
        post_nms_topN: 在NMS处理之后，分数在前面的rois
        nms_thresh: NMS的阈值
    '''

    if cfg_key == "TRAIN":
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n  # 12000个
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n  # 2000个
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh  # 0.7
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n  # 6000个
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n  # 300个
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh

    im_info = im_info[0]
    # Get the scores and bounding boxes
    # 假设rpn_cls_prob = (1,38,50,18)
    # 其中第四维度前9位是背景的分数，后9位是前景的分数


    if cfg.FLAGS.use_fpn:
        scores = rpn_cls_prob[:, 1]
    else:
        scores = rpn_cls_prob[:, :, :, num_anchors:]  # scores = (1,38,50,9)
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))  # rpn_bbox_pred = （1,38,50,36）->(17100,4)
    scores = scores.reshape((-1, 1))  # scores = (17100,1)

    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)  # bbox_transform_inv 根据anchor和偏移量计算proposals
    proposals = clip_boxes(proposals, im_info[:2])  # clip_boxes作用：调整boxes的坐标，使其全部在图像的范围内

    # Pick the top region proposals
    # 首先变成一维，然后argsort返回数组值从小到大的索引值,然后加上[::-1]，翻转序列
    # order保存数组值从大到小的索引值
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]  # 只取前pre_nms_topN
        # order对应的是下标，然后把得分最高的前pre_nms_topN的区域保存
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False))) # (post_nms_topN,5),第一列全为0


    return blob, scores
