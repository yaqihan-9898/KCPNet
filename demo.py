#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.datasets import pascal_voc
from lib.nets.resnet import resnet
from lib.utils.timer import Timer
import xml.etree.ElementTree as ET
from lib.utils.coordinate_convert import forward_convert



def demo(sess, net, image_name,Times):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('./data/VOCdevkit2007/VOC2007/JPEGImages', image_name)
    im = cv2.imread(im_file)
    # Detect all object classes and regress object bounds

    timer = Timer()
    timer.tic()
    if cfg.FLAGS.pre_ro:
        scores, boxes, boxes_r = im_detect(sess, net, im)
    else:
        scores, boxes= im_detect(sess, net, im)

    timer.toc()
    Times.append(timer.total_time)
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))


    # Visualize detections for each class

    NMS_THRESH = 0.5
    for cls_ind, cls in enumerate(pascal_voc.CLASSES[1:]):
        filename= get_path()
        filename = filename.format(cls)
        cls_ind += 1  # because we skipped background
        cls_scores = scores[:, cls_ind]

        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
        
        keep = nms(dets, NMS_THRESH)
        dets_final = dets[keep, :]
        img = image_name.split(".")

        if dets_final.shape[0] != 0:
            for i in range(dets_final.shape[0]):
                with open(filename, 'a') as f:
                    f.write(
                        '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(img[0], dets_final[i, 4],
                                                                                    dets_final[i, 0],
                                                                                    dets_final[i, 1],
                                                                                    dets_final[i, 2], dets_final[i, 3]))

def get_path():
    filename = 'test_{:s}.txt'
    path = os.path.join(
        './output/dets/' + cfg.network,
        filename)
    return path


if __name__ == '__main__':
    # model path
    tfmodel = tf.train.latest_checkpoint(cfg.get_output_dir())
    print(tfmodel)

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if cfg.FLAGS.network.startswith('resnet'):
        net = resnet(batch_size=cfg.FLAGS.ims_per_batch)
    else:
        raise NotImplementedError

    n_classes = len(pascal_voc.CLASSES[1:])
    # create the structure of the net having a certain shape (which depends on the number of classes)
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    fi = open('./data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt')
    txt = fi.readlines()
    im_names = []
    for line in txt:
        line = line.strip('\n')
        line = (line + cfg.FLAGS.image_ext)
        im_names.append(line)
    fi.close()

    for cls_ind, cls in enumerate(pascal_voc.CLASSES[1:]):
        filename= get_path()
        if not os.path.exists(os.path.join('./output/dets_r/'+cfg.network)):
            os.makedirs(os.path.join('./output/dets_r/'+cfg.network))
        filename = filename.format(cls)
        file = open(filename, 'w')
        file.close



    i=0
    Times = []
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name,Times)
    print(np.average(np.array(times)))



