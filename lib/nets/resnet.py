# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import lib.config.config as cfg
from lib.nets.network import Network
from lib.utils.tools import add_heatmap
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from lib.layer_utils.get_mask import get_mask_region
from lib.layer_utils.proposal_layer import add_demension
from lib.layer_utils.hcf import crop_and_get_hcf_feature
def gauss(x, y, sigma=3.):
    Z = 2*np.pi*sigma**2
    return 1/Z*np.exp(-(x**2+y**2)/2/sigma**2)

def resnet_arg_scope(
        is_training=True, weight_decay=cfg.FLAGS.weight_decay, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class resnet(Network):
    def __init__(self, batch_size=1, num_layers=101):
        Network.__init__(self, batch_size=batch_size)
        self._num_layers = num_layers
        self._resnet_scope = 'resnet_v1_%d' % num_layers

    def build_network(self, sess, is_training=True):

        with tf.variable_scope('resnet_v1_101', 'resnet_v1_101'):

            # select initializer
            if cfg.FLAGS.initializer == "truncated":
                initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
            else:
                initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
                initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        # Build head
        if cfg.FLAGS.attention == True:
            net, net_attention,net_attention_plus = self.build_head(is_training, initializer, initializer_bbox)

            self._predictions["net_attention"] =net_attention
            self._predictions["net_attention_plus"] = net_attention_plus
        else:
            net = self.build_head(is_training, initializer,initializer_bbox)


        with tf.variable_scope('resnet_v1_101', 'resnet_v1_101'):
            rpn_cls_prob, rpn_bbox_pred, rpn_cls_score,rpn_cls_score_reshape = self.build_rpn(net, is_training, initializer)
            self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape

            # Build proposals
            rois = self.build_proposals(is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score)

        # Build predictions
        cls_score, cls_prob, bbox_pred = self.build_predictions(net, rois, is_training, initializer,
                                                                initializer_bbox)

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["cls_score"] = cls_score
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred
        self._predictions["rois"] = rois

        self._score_summaries.update(self._predictions)


        return rois, cls_prob, bbox_pred


    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._resnet_scope + '/conv1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.startswith('faster_rcnn'):
                continue
            if v.name.startswith('fusion'):
                continue
            if v.name.startswith(self._resnet_scope + '/rpn'):
                continue

            if v.name.split(':')[0] in var_keep_dic:
                print('Varibles restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def get_variables_to_restore_head(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            if v.name.startswith('conv_new_1') and v.name.split(':')[0] in var_keep_dic:
                print('Varibles restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix Resnet V1 layers..')
        with tf.variable_scope('Fix_Resnet_V1') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)
                if cfg.FLAGS.continue_train:
                    sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                                       conv1_rgb))
                else:
                    sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))

    def build_attention(self,inputs, is_training, initializer):
        attention_conv3x3_1 = slim.conv2d(inputs, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_1')
        attention_conv3x3_2 = slim.conv2d(attention_conv3x3_1, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_2')
        attention_conv3x3_3 = slim.conv2d(attention_conv3x3_2, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_3')
        attention_conv3x3_4 = slim.conv2d(attention_conv3x3_3, 256, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=tf.nn.relu,
                                          scope='attention_conv/3x3_4')
        attention_conv3x3_5 = slim.conv2d(attention_conv3x3_4, 2, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=None,
                                          scope='attention_conv/3x3_5')
        attention_conv3x3_6 = slim.conv2d(attention_conv3x3_4, 2, [3, 3],
                                          trainable=is_training,
                                          weights_initializer=initializer,
                                          activation_fn=None,
                                          scope='attention_conv/3x3_6')
        return attention_conv3x3_5,attention_conv3x3_6



    def build_head(self, is_training,initializer,initializer_bbox):

        scope_name = 'resnet_v1_101'
        middle_num_units = 23

        blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                  resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                  # use stride 1 for the last conv4 layer.

                  resnet_v1_block('block3', base_depth=256, num_units=middle_num_units, stride=1)]
        # when use fpn, stride list is [1, 2, 2]

        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            with tf.variable_scope(scope_name, scope_name):
                # Do the first few layers manually, because 'SAME' padding can behave inconsistently
                # for images of different sizes: sometimes 0, sometimes 1
                net = resnet_utils.conv2d_same(
                    self._image, 64, 7, stride=2, scope='conv1')
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
                net = slim.max_pool2d(
                    net, [3, 3], stride=2, padding='VALID', scope='pool1')

        not_freezed = [False] * cfg.FLAGS.FIXED_BLOCKS + (4 - cfg.FLAGS.FIXED_BLOCKS) * [True]
        # Fixed_Blocks can be 1~3

        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
            C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                    blocks[0:1],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=scope_name)

        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
            C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                    blocks[1:2],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=scope_name)

        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
            C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                        blocks[2:3],
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=scope_name)

            with tf.variable_scope('fusion',
                                   regularizer=slim.l2_regularizer(cfg.FLAGS.weight_decay)):
                C3_shape = tf.shape(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)])
                C4 = tf.image.resize_bilinear(C4, (C3_shape[1], C3_shape[2]))
                C4 = slim.conv2d(C4,
                                 512, [1, 1],
                                 trainable=is_training,
                                 weights_initializer=initializer,
                                 activation_fn=tf.nn.relu,
                                 scope='C4_conv1x1')

                _C3 = self.cbam_block(end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(scope_name)], 'cbam_C3',
                                      only_channel=True)
                C4 = C4 + _C3

                _C2 = tf.image.resize_bilinear(end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(scope_name)],
                                               (C3_shape[1], C3_shape[2]))
                _C2 = slim.conv2d(_C2,
                                  512, [1, 1],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  activation_fn=tf.nn.relu,
                                  scope='C2_conv1x1')
                _C2 = self.cbam_block(_C2, 'cbam_C2',
                                      only_channel=True)

                C4 = C4 + _C2
                kernel1 = tf.Variable(tf.zeros(shape=[3, 3, 512, 256], dtype=tf.float32, name='k1'))
                kernel2 = tf.Variable(tf.zeros(shape=[3, 3, 128, 128], dtype=tf.float32, name='k2'))
                kernel3 = tf.Variable(tf.zeros(shape=[3, 3, 128, 128], dtype=tf.float32, name='k3'))

                R1 = tf.nn.atrous_conv2d(C4, kernel1, 2, 'SAME', name='R1')
                _R1 = tf.concat([C4, R1], axis=-1)
                _R1 = slim.conv2d(_R1,
                                  128, [1, 1],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  activation_fn=tf.nn.relu,
                                  scope='R1_conv1x1')
                R2 = tf.nn.atrous_conv2d(_R1, kernel2, 4, 'SAME', name='R2')
                _R2 = tf.concat([C4, R1, R2], axis=-1)
                _R2 = slim.conv2d(_R2,
                                  128, [1, 1],
                                  trainable=is_training,
                                  weights_initializer=initializer,
                                  activation_fn=tf.nn.relu,
                                  scope='R2_conv1x1')
                R3 = tf.nn.atrous_conv2d(_R2, kernel3, 8, 'SAME', name='R3')
                C4 = tf.concat([C4, R1, R2, R3], axis=-1)
                # C4 = slim.conv2d(C4,
                #                   1024, [1, 1],
                #                   trainable=is_training,
                #                   weights_initializer=initializer,
                #                   activation_fn=tf.nn.relu,
                #                   scope='P2_conv1x1')

            with tf.variable_scope('build_C4_attention',
                                   regularizer=slim.l2_regularizer(cfg.FLAGS.weight_decay)):

                add_heatmap(tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1), 'add_attention_before')
                C4_attention_layer, C4_attention_layer_plus = self.build_attention(C4, is_training, initializer)
                C4_attention = tf.nn.softmax(C4_attention_layer)
                C4_attention_plus = tf.nn.softmax(C4_attention_layer_plus)
                C4_attention = C4_attention[:, :, :, 1]
                C4_attention_plus = C4_attention_plus[:, :, :, 1]
                C4_attention = tf.expand_dims(C4_attention, axis=-1)
                C4_attention_plus = tf.expand_dims(C4_attention_plus, axis=-1)
                C4_attention = 1 + 0.5 * C4_attention + 0.5 * C4_attention_plus
                add_heatmap(C4_attention, 'C4_attention')
                C4 = tf.multiply(C4_attention, C4)
                add_heatmap(tf.expand_dims(tf.reduce_mean(C4, axis=-1), axis=-1), 'add_attention_after')
            self.attention = C4_attention
            self._layers['head'] = C4
            self.mask_boxes, self.threshimg = \
                tf.py_func(
                    get_mask_region,
                    [self.attention, self.item],
                    [tf.float32, tf.float32])

            self.mask_boxes = tf.reshape(self.mask_boxes, [-1, 4])
            if cfg.FLAGS.attention == True:
                return C4, C4_attention_layer, C4_attention_layer_plus
            else:
                return C4




    def build_rpn(self, net, is_training, initializer):
        self._anchor_component()
        # Create RPN Layer
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")

        self._act_summaries.append(rpn)

        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], stride=1, trainable=is_training,
                                    weights_initializer=initializer, padding='VALID', activation_fn=None,
                                    scope='rpn_cls_score')  # stride =2 ?

        # Change it so that the score has 2 as its channel size

        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], stride=1, padding='VALID',
                                    trainable=is_training, weights_initializer=initializer, activation_fn=None,
                                    scope='rpn_bbox_pred')
        return rpn_cls_prob, rpn_bbox_pred, rpn_cls_score, rpn_cls_score_reshape


    def build_proposals(self, is_training, fpn_cls_prob, fpn_box_pred, rpn_cls_score):
        if is_training:
            rois, roi_scores = self._proposal_layer(fpn_cls_prob, fpn_box_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                if cfg.FLAGS.use_roi_from_att:
                    cols = tf.unstack(self.mask_boxes, axis=1)

                    boxes = tf.stack([(cols[1] + cols[3]) / 2,
                                      (cols[0] + cols[2]) / 2,
                                      cols[3] + cols[1],
                                      cols[2] + cols[0]], axis=1)
                    boxes = tf.py_func(add_demension, [boxes], [tf.float32])
                    boxes = tf.reshape(boxes, [-1, 5])
                    rois_from_att = tf.reshape(boxes, [-1, 5])
                else:
                    rois_from_att = rois[0]
                    rois_from_att = tf.reshape(rois_from_att, [-1, 5])

                rois, _ = self._proposal_target_layer(rois, roi_scores, rois_from_att, "rpn_rois")
        else:
            if cfg.FLAGS.test_mode == 'nms':
                rois, _ = self._proposal_layer(fpn_cls_prob, fpn_box_pred, "rois")
            elif cfg.FLAGS.test_mode == 'top':
                rois, _ = self._proposal_top_layer(fpn_cls_prob, fpn_box_pred, "rois")
            else:
                raise NotImplementedError
            if cfg.FLAGS.use_roi_from_att:
                cols = tf.unstack(self.mask_boxes, axis=1)

                boxes = tf.stack([(cols[1] + cols[3]) / 2,
                                  (cols[0] + cols[2]) / 2,
                                  cols[3] + cols[1],
                                  cols[2] + cols[0]], axis=1)  # (?, 4)
                boxes = tf.py_func(add_demension, [boxes], [tf.float32])
                boxes = tf.reshape(boxes, [-1, 5])
                rois = tf.concat([rois, boxes], axis=0)

        return rois



    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):
        if is_training:
            hcf_feature = tf.py_func(crop_and_get_hcf_feature,
                                     [rois, self._image[0]],
                                     [tf.float32])
            hcf_feature = tf.reshape(hcf_feature, [-1, cfg.FLAGS.hcf_num])
            # hcf_feature = tf.nn.l2_normalize(hcf_feature, -1, epsilon=1e-12, name='hcf_feature_l2_normalize')
            self.hcf = hcf_feature*5


        scope_name = 'resnet_v1_101'
        pool5 = self._crop_pool_layer(net, rois, "pool5")
        pool5_context = self._crop_pool_layer(net, rois, "pool5", is_context=True)

        block4 = [resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            C5, _ = resnet_v1.resnet_v1(pool5,
                                        block4,
                                        global_pool=False,
                                        include_root_block=False,
                                        scope=scope_name)

            fc7 = tf.reduce_mean(C5, axis=[1, 2], keep_dims=False, name='global_average_pooling')


        with tf.variable_scope('faster_rcnn', 'faster_rcnn'):
            pool5_conv3x3_1 = slim.conv2d(pool5, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5/3x3_1')
            pool5_conv3x3_2 = slim.conv2d(pool5_conv3x3_1, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5/3x3_2')
            pool5_conv3x3_3 = slim.conv2d(pool5_conv3x3_2, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5/3x3_3')
            pool5_conv3x3_4 = slim.conv2d(pool5_conv3x3_3, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5/3x3_4')
            pool5_conv3x3_5 = slim.conv2d(pool5_conv3x3_4, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=None,
                                                  scope='pool5/3x3_5')
            target = tf.reduce_mean(pool5_conv3x3_5, axis=[1, 2], keep_dims=False,
                                     name='global_average_pool')
            hcf_prediction_target = slim.fully_connected(target, cfg.FLAGS.hcf_num-48, weights_initializer=initializer_bbox,
                                                  trainable=is_training, activation_fn=None, scope='hcf_pred_target')
            pool5_context_conv3x3_1 = slim.conv2d(pool5_context, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5_context/3x3_1')
            pool5_context_conv3x3_2 = slim.conv2d(pool5_context_conv3x3_1, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5_context/3x3_2')
            pool5_context_conv3x3_3 = slim.conv2d(pool5_context_conv3x3_2, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5_context/3x3_3')
            pool5_context_conv3x3_4 = slim.conv2d(pool5_context_conv3x3_3, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=tf.nn.relu,
                                                  scope='pool5_context/3x3_4')
            pool5_context_conv3x3_5 = slim.conv2d(pool5_context_conv3x3_4, 256, [3, 3],
                                                  trainable=is_training,
                                                  weights_initializer=initializer,
                                                  activation_fn=None,
                                                  scope='pool5_context/3x3_5')
            context = tf.reduce_mean(pool5_context_conv3x3_5, axis=[1, 2], keep_dims=False, name='global_average_pooling')

            hcf_predictionz_context = slim.fully_connected(context, 48, weights_initializer=initializer_bbox,
                                                   trainable=is_training, activation_fn=None, scope='hcf_pred_context')

            self.hcf_pred = tf.concat(values=[hcf_prediction_target, hcf_predictionz_context], axis=-1)
            hcf_prediction = tf.nn.l2_normalize(self.hcf_pred/5, 0, epsilon=1e-12, name='hcf_feature_l2_normalize')
            # hcf_prediction=tf.stop_gradient(hcf_prediction)

            fc7 = tf.concat(values=[fc7, hcf_prediction], axis=1)
        # Scores and predictions
        with tf.variable_scope('faster_rcnn', 'faster_rcnn'):
            cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer,
                                             trainable=is_training, activation_fn=None, scope='cls_score')
            cls_prob = self._softmax_layer(cls_score, "cls_prob")
            bbox_prediction = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                                                   trainable=is_training, activation_fn=None, scope='bbox_pred')
        return cls_score, cls_prob, bbox_prediction

    # def squeeze_excitation_layer(self,input_x, out_dim, ratio, layer_name, is_training):
    #     with tf.name_scope(layer_name):
    #         # Global_Average_Pooling
    #         squeeze = tf.reduce_mean(input_x, [1, 2])
    #
    #         excitation = slim.fully_connected(inputs=squeeze,
    #                                           num_outputs=out_dim // ratio,
    #                                           weights_initializer=cfgs.BBOX_INITIALIZER,
    #                                           activation_fn=tf.nn.relu,
    #                                           trainable=is_training,
    #                                           scope=layer_name + '_fully_connected1')
    #
    #         excitation = slim.fully_connected(inputs=excitation,
    #                                           num_outputs=out_dim,
    #                                           weights_initializer=cfgs.BBOX_INITIALIZER,
    #                                           activation_fn=tf.nn.sigmoid,
    #                                           trainable=is_training,
    #                                           scope=layer_name + '_fully_connected2')
    #
    #         excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    #
    #         # scale = input_x * excitation
    #
    #         return excitation

    def build_inception(self, inputs, is_training,initializer,scope):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(inputs, 384, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(inputs, 192, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 224, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0b_1x7')
                branch_1 = slim.conv2d(branch_1, 256, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0c_7x1')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(inputs, 192, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 192, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0b_7x1')
                branch_2 = slim.conv2d(branch_2, 224, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'Conv2d_0c_1x7')
                branch_2 = slim.conv2d(branch_2, 224, [7, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0d_7x1')
                branch_2 = slim.conv2d(branch_2, 256, [1, 7],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope=scope+'conv2d_0e_1x7')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='avgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 128, [1, 1],
                                       trainable=is_training,
                                       weights_initializer=initializer,
                                       activation_fn=tf.nn.relu,
                                       scope='conv2d_0b_1x1')
            inception_out = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            return inception_out

    def build_inception_attention(self,inputs, is_training,initializer):
        """Builds Inception-B block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        inception_out = self.build_inception(inputs, is_training,initializer)

        inception_attention_out = slim.conv2d(inception_out, 2, [3, 3],
                                              trainable=is_training,
                                              weights_initializer=initializer,
                                              activation_fn=None,
                                              scope='inception_attention_out')
        return inception_attention_out

    def squeeze_excitation_layer(self,input_x, out_dim, ratio, layer_name, is_training,initializer_bbox):
        with tf.name_scope(layer_name):
            # Global_Average_Pooling
            squeeze = tf.reduce_mean(input_x, [1, 2])

            excitation = slim.fully_connected(inputs=squeeze,
                                              num_outputs=out_dim // ratio,
                                              weights_initializer=initializer_bbox,
                                              activation_fn=tf.nn.relu,
                                              trainable=is_training,
                                              scope=layer_name + '_fully_connected1')

            excitation = slim.fully_connected(inputs=excitation,
                                              num_outputs=out_dim,
                                              weights_initializer=initializer_bbox,
                                              activation_fn=tf.nn.sigmoid,
                                              trainable=is_training,
                                              scope=layer_name + '_fully_connected2')

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

            # scale = input_x * excitation

            return excitation

    def  cbam_block(self,input_feature, name, only_channel=False,ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """

        with tf.variable_scope(name):

            attention_feature = self.channel_attention(input_feature, 'ch_at', ratio,mode=1)
            if not only_channel:
                attention_feature = self.spatial_attention(attention_feature, 'sp_at')
            print("CBAM Hello")
        return attention_feature


    def channel_attention(self,input_feature, name, ratio=8,mode=1):

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        with tf.variable_scope(name):
            channel = input_feature.get_shape()[-1]
            avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

            assert avg_pool.get_shape()[1:] == (1, 1, channel)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_0',
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name='mlp_1',
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel)

            max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       name='mlp_0',
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel,
                                       name='mlp_1',
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)

            scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
        if mode == 1:
            return input_feature * scale
        else:
            return  scale

    def spatial_attention(self,input_feature, name):
        kernel_size = 7
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
            assert avg_pool.get_shape()[-1] == 1
            max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
            assert max_pool.get_shape()[-1] == 1
            concat = tf.concat([avg_pool, max_pool], 3)
            assert concat.get_shape()[-1] == 2

            concat = tf.layers.conv2d(concat,
                                      filters=1,
                                      kernel_size=[kernel_size, kernel_size],
                                      strides=[1, 1],
                                      padding="same",
                                      activation=None,
                                      kernel_initializer=kernel_initializer,
                                      use_bias=False,
                                      name='conv')
            assert concat.get_shape()[-1] == 1
            concat = tf.sigmoid(concat, 'sigmoid')

        return input_feature * concat



