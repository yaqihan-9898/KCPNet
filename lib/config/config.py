import os
import os.path as osp

import numpy as np
import tensorflow as tf

network='ISDD_20221024'
FLAGS = tf.app.flags.FLAGS
FLAGS2 = {}

######################
# General Parameters #
######################
tf.app.flags.DEFINE_integer('rng_seed', 3, "Tensorflow seed for reproducibility")


######################
# Dataset Parameters #
######################
tf.app.flags.DEFINE_string('dataset', "ISDD", "The name of dataset")
tf.app.flags.DEFINE_string('image_ext', ".png", "The extensions name of images")
FLAGS2["pixel_means"] = np.array([[[48.7538, 48.7538,  48.7538]]])

######################
# Network Parameters #
######################
tf.app.flags.DEFINE_string('network', "resnet", "The network to be used as backbone")
tf.app.flags.DEFINE_integer('hcf_num', 84, "the dimension of visual features")
tf.app.flags.DEFINE_boolean('attention',True, "Whether to ues add attention")
tf.app.flags.DEFINE_boolean('use_roi_from_att',False, "Whether to generate additional proposals from attenton masks")

#######################
# Training Parameters #
#######################
tf.app.flags.DEFINE_float('weight_decay', 0.0001, "Weight decay, for regularization")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "Learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('gamma', 0.1, "Factor for reducing the learning rate")
tf.app.flags.DEFINE_float('EPSILON', 1e-7, "Factor for reducing the learning rate")

tf.app.flags.DEFINE_integer('batch_size', 256, "Network batch size during training")
tf.app.flags.DEFINE_integer('max_iters', 65000, "Max iteration")
tf.app.flags.DEFINE_integer('step_size_1', 20000,"lr = lr * 0.1 after step_size_1")
tf.app.flags.DEFINE_integer('step_size_2', 50000, "lr = lr * 0.01 after step_size_2")
tf.app.flags.DEFINE_integer('display', 10, "Iteration intervals for showing the loss during training, on command line interface")
tf.app.flags.DEFINE_integer('summary_per_iter', 2000, "Iteration intervals for summary in tensorboard")

tf.app.flags.DEFINE_boolean('continue_train',True, "Whether to continue train from ckpt")
tf.app.flags.DEFINE_string('my_ckpt', "./data/pretrained_weights/pre_trained.ckpt", "Network weights")
tf.app.flags.DEFINE_string('pretrained_model', "./data/pretrained_weights/pre_trained.ckpt", "Pretrained network weights")
tf.app.flags.DEFINE_string('summary_path', "./summary", "Summary path")
tf.app.flags.DEFINE_string('initializer', "truncated", "Network initialization parameters")

tf.app.flags.DEFINE_boolean('bias_decay', False, "Whether to adopt weight decay on bias")
tf.app.flags.DEFINE_boolean('double_bias', True, "Whether to double the learning rate for bias")
tf.app.flags.DEFINE_boolean('double_head',False, "Whether to double the learning rate for network head")
tf.app.flags.DEFINE_boolean('double_cls',True, "Whether to double cls loss")
tf.app.flags.DEFINE_integer('double_iter',15000, "If double_cls is Ture, learning rate for cls loss will double after double_iter")
tf.app.flags.DEFINE_boolean('use_all_gt', True, "Whether to use all ground truth bounding boxes for training, "
                                                "For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''")
tf.app.flags.DEFINE_boolean('MAX_POOL', True, "Whether to use max pooling")

tf.app.flags.DEFINE_integer('FIXED_BLOCKS',1, "Number of frozen backbone blocks")
tf.app.flags.DEFINE_integer('max_size', 1000, "Max pixel size of the longest side of a scaled input image")
tf.app.flags.DEFINE_integer('test_max_size', 1000, "Max pixel size of the longest side of a scaled input image")
tf.app.flags.DEFINE_integer('ims_per_batch', 1, "Images per minibatch")
tf.app.flags.DEFINE_integer('snapshot_iterations', 5000, "Iteration to take snapshot")

FLAGS2["scales"] = (500,)
FLAGS2["test_scales"] = (500,)

######################
# Testing Parameters #
######################
tf.app.flags.DEFINE_string('test_mode', "top", "Test mode for bbox proposal")  # nms, top

##################
# RPN Parameters #
##################
tf.app.flags.DEFINE_float('rpn_negative_overlap', 0.3, "IOU < thresh: negative example")
tf.app.flags.DEFINE_float('rpn_positive_overlap', 0.7, "IOU >= thresh: positive example")
tf.app.flags.DEFINE_float('rpn_fg_fraction', 0.5, "Max number of foreground examples")
tf.app.flags.DEFINE_float('rpn_train_nms_thresh', 0.7, "NMS threshold used on RPN proposals")
tf.app.flags.DEFINE_float('rpn_test_nms_thresh', 0.5, "NMS threshold used on RPN proposals")

tf.app.flags.DEFINE_integer('rpn_train_pre_nms_top_n', 12000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_train_post_nms_top_n', 2000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_pre_nms_top_n', 6000, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_test_post_nms_top_n', 600, "Number of top scoring boxes to keep before apply NMS to RPN proposals")
tf.app.flags.DEFINE_integer('rpn_batchsize', 256, "Total number of examples")
tf.app.flags.DEFINE_integer('rpn_positive_weight', -1,
                            'Give the positive RPN examples weight of p * 1 / {num positives} and give negatives a weight of (1 - p).'
                            'Set to -1.0 to use uniform example weighting')
tf.app.flags.DEFINE_integer('rpn_top_n', 600, "Only useful when TEST.MODE is 'top', specifies the number of top proposals to select")

tf.app.flags.DEFINE_boolean('rpn_clobber_positives', False, "If an anchor satisfied by positive and negative conditions set to negative")
IS_FILTER_OUTSIDE_BOXES = False
TRAIN_RPN_CLOOBER_POSITIVES =False
#######################
# Proposal Parameters #
#######################
tf.app.flags.DEFINE_float('proposal_fg_fraction', 0.25, "Fraction of minibatch that is labeled foreground (i.e. class > 0)")
tf.app.flags.DEFINE_boolean('proposal_use_gt', False, "Whether to add ground truth boxes to the pool when sampling regions")

###########################
# Bounding Box Parameters #
###########################
tf.app.flags.DEFINE_float('roi_fg_threshold', 0.5, "Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)")
tf.app.flags.DEFINE_float('roi_bg_threshold_high', 0.5, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")
tf.app.flags.DEFINE_float('roi_bg_threshold_low', 0.1, "Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))")

tf.app.flags.DEFINE_boolean('bbox_normalize_targets_precomputed', True, "# Normalize the targets using 'precomputed' (or made up) means and stdevs (BBOX_NORMALIZE_TARGETS must also be True)")
tf.app.flags.DEFINE_boolean('test_bbox_reg', True, "Test using bounding-box regressors")

FLAGS2["bbox_inside_weights"] = (1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_inside_weights_ro"] = (1.0, 1.0, 1.0, 1.0, 1.0)
FLAGS2["bbox_normalize_means"] = (0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds"] = (0.1, 0.1, 0.1, 0.1)
FLAGS2["bbox_normalize_means_ro"] = (0.0, 0.0, 0.0, 0.0, 0.0)
FLAGS2["bbox_normalize_stds_ro"] = (0.1, 0.1, 0.1, 0.1, 0.1)

##################
# ROI Parameters #
##################
tf.app.flags.DEFINE_integer('roi_pooling_size', 7, "Size of the pooled region after RoI pooling")

######################
# Dataset Parameters #
######################
FLAGS2["root_dir"] = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
FLAGS2["data_dir"] = osp.abspath(osp.join(FLAGS2["root_dir"], 'data'))


def get_output_dir():
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(FLAGS2["root_dir"], FLAGS2["root_dir"] , 'output','model', network))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
