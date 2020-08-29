import tensorflow as tf
import numpy as np

from network.layers import *
from network.utils.utils import *
from network.utils.data_utils import *
from network.augment import *


def slim_backbon(backbone, image):
    import tensorflow.contrib.slim as slim
    if backbone == "resnet_v1_50":
        with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
            with slim.arg_scope([slim.model_variable, slim.variable]):
                logits, end_points = slim.nets.resnet_v1.resnet_v1_50(
                    image, num_classes=1000, is_training=False)
            # mapping from https://github.com/wuzheng-sjtu/FastFPN/blob/master/libs/nets/pyramid_network.py
            mapping = {"C1": "resnet_v1_50/conv1/Relu:0",
                       "C2": "resnet_v1_50/block1/unit_2/bottleneck_v1",
                       "C3": "resnet_v1_50/block2/unit_3/bottleneck_v1",
                       "C4": "resnet_v1_50/block3/unit_5/bottleneck_v1",
                       "C5": "resnet_v1_50/block4/unit_3/bottleneck_v1"}
            return [end_points[mapping[c]] for c in ["C2", "C3", "C4", "C5"]]

############################################################
#  Resnet Graph
############################################################


resnet_version = {"resnet50": [3, 4, 6, 3],
                  "resnet101": [3, 4, 23, 3],
                  "resnet152": [3, 8, 36, 3]}


def resnet_graph(backbone, inputs, training, use_bn):
    """Build convolutional network layers attached to the given input tensor"""

    blocks = resnet_version[backbone]

    with tf.variable_scope("resnet_backbone"):
        # stage 1
        C1 = conv_layer(inputs, 64, kernel_size=7, strides=2)
        C1 = norm_layer(C1, training, use_bn)
        C1 = pool_layer(C1, (3, 3), stride=(2, 2))

        # stage2
        C2 = res_block_v1(C1, [64, 64, 256], training,
                          use_bn, strides=1, downsample=True)
        for i in range(blocks[0] - 1):
            C2 = res_block_v1(C2, [64, 64, 256], training, use_bn)

        # stage3
        C3 = res_block_v1(C2, [128, 128, 512], training,
                          use_bn, strides=2, downsample=True)
        for i in range(blocks[1] - 1):
            C3 = res_block_v1(C3, [128, 128, 512], training, use_bn)

        # stage4
        C4 = res_block_v1(C3, [256, 256, 1024], training,
                          use_bn, strides=2, downsample=True)
        for i in range(blocks[2] - 1):
            C4 = res_block_v1(C4, [256, 256, 1024], training, use_bn)

        # stage5
        C5 = res_block_v1(C4, [512, 512, 2048], training,
                          use_bn, strides=2, downsample=True)
        for i in range(blocks[3] - 1):
            C5 = res_block_v1(C5, [512, 512, 2048], training, use_bn)

        return C2, C3, C4, C5

############################################################
#  Feature Pyramid Network Graph
############################################################


def fpn_graph(resnet_feature_map, config, C2_mode=False):
    """
    RetinaNet
    https://arxiv.org/pdf/1708.02002.pdf
    P6 is obtained via a 3×3 stride-2 conv on C5
    P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6
    """
    C2, C3, C4, C5 = resnet_feature_map
    with tf.variable_scope("fpn"):  # TODO: check FPN for ReinaNet
        P5_up = conv_layer(C5, 256, kernel_size=1)
        P4_up = upsampling(P5_up, size=(2, 2)) + \
            conv_layer(C4, 256, kernel_size=1)
        P3_up = upsampling(P4_up, size=(2, 2)) + \
            conv_layer(C3, 256, kernel_size=1)

        P5 = conv_layer(P5_up, 256, kernel_size=3)
        P4 = conv_layer(P4_up, 256, kernel_size=3)
        P3 = conv_layer(P3_up, 256, kernel_size=3)

        P6 = conv_layer(C5, 256, kernel_size=3, strides=2)
        P7 = relu(P6)
        P7 = conv_layer(P7, 256, kernel_size=3, strides=2)

        if C2_mode:
            assert len(config.FEATURE_STRIDES) == 6
            P2 = upsampling(P3_up, size=(2, 2)) + \
                conv_layer(C2, 256, kernel_size=1)
            P2 = conv_layer(P2, 256, kernel_size=3)
            return P2, P3, P4, P5, P6, P7

    return P3, P4, P5, P6, P7

############################################################
#  Task Specific Layer
############################################################


def cls_task_head(feature, out_dims, scope, probability=0.01):
    with tf.variable_scope(scope):
        _kernel_initializer = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01)
        for _ in range(4):
            feature = conv_layer(feature, 256, kernel_size=3,
                                 use_bias=False, kernel_initializer=_kernel_initializer)
            feature = relu(feature)

        feature = conv_layer(feature, out_dims, kernel_size=(
            3, 5), kernel_initializer=_kernel_initializer)
        bias_initial = tf.ones(out_dims, dtype=tf.float32) * - \
            tf.log((1 - probability) / probability)
        output_layer = tf.nn.bias_add(feature, bias_initial)
        return output_layer


def loc_task_head(feature, out_dims, scope, probability=0.01):
    with tf.variable_scope(scope):
        _kernel_initializer = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01)
        for _ in range(4):
            feature = conv_layer(feature, 256, kernel_size=3,
                                 use_bias=False, kernel_initializer=_kernel_initializer)
            feature = relu(feature)
        output_layer = conv_layer(feature, out_dims, kernel_size=(
            3, 5), kernel_initializer=_kernel_initializer)
        return output_layer

############################################################
#  Loss Functions
############################################################


def regression_loss_graph(pred_boxes, gt_boxes, weights=1.0, scope="regression_loss"):
    """Regression loss (Smooth L1 loss (=huber loss))

    ARGS:
        - preds_cls : Tensor(valid_anchor, 2)
        - gt_cls : Tensor(valid_anchor, 1)
    RETURNS:
            weights: Tensor of weights multiplied by loss with shape ( anchors )

    """
    with tf.variable_scope(scope):
        x = tf.abs(pred_boxes-gt_boxes)
        x = tf.where(tf.less(x, 1.0), 0.5*x**2, x-0.5)
        x = tf.reduce_sum(x)
        return x


def huber_loss_graph(pred_boxes, gt_boxes, cilp_value=1.0, scope="huber_loss"):
    """
    Huber loss

    ARGS:
        - preds_cls : Tensor(valid_anchor, 2)
        - gt_cls : Tensor(valid_anchor, 1)
    RETURNS:
            weights: Tensor of weights multiplied by loss with shape ( anchors )

    """
    with tf.variable_scope(scope):
        x = tf.abs(pred_boxes - gt_boxes)
        x = tf.where(tf.less(x, cilp_value), (0.5 * x) **
                     2, (x - 0.5 * cilp_value) * cilp_value)
        x = tf.reduce_sum(x)
        return x


def smooth_l1_loss_graph(pred_boxes, gt_boxes, weights=1.0, scope="smooth_l1_loss"):
    """
    Regression loss (Smooth L1 loss (=huber loss))

    ARGS:
        - preds_cls : Tensor(valid_anchor, 2)
        - gt_cls : Tensor(valid_anchor, 1)
    RETURNS:
            weights: Tensor of weights multiplied by loss with shape ( anchors )
    """
    with tf.variable_scope(scope):
        return tf.reduce_sum(tf.losses.huber_loss(
            gt_boxes,
            pred_boxes,
            delta=1,
            weights=tf.expand_dims(weights, axis=2),
            loss_collection=None,
            reduction=tf.losses.Reduction.NONE
        ), axis=2)


def focal_loss_graph(preds_cls, gt_cls, alpha=0.25, gamma=2.0, name=None, scope="focal_loss"):
    """Compute sigmoid focal loss

    ARGS:
        - preds_cls : Tensor(valid_anchor, 2)
        - gt_cls : Tensor(valid_anchor, 1)
    RETURNS:


    """
    with tf.variable_scope(scope):
        # prepare internal required value (alpha, gamma)
        condition = tf.equal(tf.reshape(gt_cls, [-1]), 1.0)
        # shape = (valid_anchor, num_class)
        alpha_t = tf.scalar_mul(
            alpha, tf.ones_like(preds_cls, dtype=tf.float32))
        # shape = (valid_anchor, num_class)
        alpha_t = tf.where(condition, alpha_t, 1.0 - alpha_t)
        # shape = (valid_anchor, num_class)
        gamma_t = tf.scalar_mul(gamma, tf.ones_like(preds_cls, tf.float32))

        # shape = (valid_anchor, num_class)
        preds_cls = tf.nn.sigmoid(preds_cls)

        # binary cross entropy -> if y=1 : pt=p /
        #                         otherwise : pt=1-p
        # shape = (valid_anchor, num_class)
        predictions_pt = tf.where(condition, preds_cls, 1.0 - preds_cls)

        # clip to avoid 0
        focal_losses = alpha_t * (-tf.pow(1.0 - predictions_pt, gamma_t) * tf.log(
            tf.clip_by_value(predictions_pt, 1e-10, 1.0)))  # shape = (valid_anchor, num_class)

        focal_losses = tf.reduce_sum(
            focal_losses, axis=-1)  # shape = (valid_anchor,)
    return focal_losses

############################################################
#  TextBoxes++ Network
############################################################


class TextBoxesNet():
    def __init__(self, config):

        # Set tune scope
        self.scope = "resnet_backbone|fpn|task_head"
        self.backbone = config.BACKBONE
        self.config = config
        self.use_bn = config.USE_BN
        self.probability = 0.01
        self.tune_scope = None
        self.anchor_list = pyramid_gen_anchor_boxes(self.config.IMAGE_SHAPE[0],
                                                    self.config.IMAGE_SHAPE[1],
                                                    self.config.ANCHOR_VERTICAL_STEP,
                                                    self.config.ANCHOR_HORIZON_STEP,
                                                    self.config.FEATURE_STRIDES,
                                                    self.config.ANCHOR_AREAS,
                                                    self.config.ANCHOR_RATIOS)
        self.slim_backbones = ["slim_resnet_v1_50"]

    def forward(self, image, **kwargs):
        """Forwarding and get logits output

        ARGS:
            input : Tensor(batch, image_width, image_height, channel=3)
            mode :
        RETURN:
            logits :  Tensor(batch,num_anchors,4+8),Tensor(batch,num_anchors,num_classes)
        """
        features_resnet = []
        if self.backbone in resnet_version.keys():
            features_resnet = resnet_graph(
                backbone=self.backbone, training=True, inputs=image, use_bn=True)
        elif self.backbone in self.slim_backbones:
            import tensorflow.contrib.slim as slim
            import tensorflow.contrib.slim.nets
            features_resnet = slim_backbon(self.backbone, image)

        # feature pyramid network
        features = fpn_graph(features_resnet, self.config, self.config.C2_MODE)

        with tf.variable_scope("task_head"):
            loc_subnet = []
            class_subnet = []
            loc_output_dims = self.config.NUM_ANCHOR_PER_GRID * (4 + 8)
            cls_output_dims = self.config.NUM_ANCHOR_PER_GRID * self.config.NUM_CLASSES

            for n, feature in enumerate(features):

                _loc = cls_task_head(
                    feature, loc_output_dims, "P%d_loc_head" % (n+3), self.probability)
                _loc = tf.reshape(
                    _loc, [self.config.IMAGES_PER_GPU, -1, 4 + 8])

                _class = loc_task_head(
                    feature, cls_output_dims, "P%d_cls_head" % (n+3))
                _class = tf.reshape(
                    _class, [self.config.IMAGES_PER_GPU, -1, self.config.NUM_CLASSES])

                loc_subnet.append(_loc)
                class_subnet.append(_class)

            # shape = (batch_size,anchor_size,12)
            loc_subnet = tf.concat(loc_subnet, axis=1)
            # shape = (batch_size,anchor_size,2)
            class_subnet = tf.concat(class_subnet, axis=1)
            return loc_subnet, class_subnet

    def decode(self, logists):
        """
        Decode data to normal coordinate

        ARGS:
            - logists : Tensor(batch_size,num_anchor,12),Tensor(batch_size,num_anchor,num_classes)
        RETURN:
            - decode_data_list : Tensor(batch_size,num_anchor,14)
                => 14, quad_boxes(8),rect_boxes(4),score(1),labels(1)
        """

        loc_output, cls_output = logists
        decode_data_list = []

        # anchor_list shape normalization
        if np.array(self.anchor_list).ndim != 2:
            dense_anchor_list = self.anchor_list[0]
            for i in range(1, len(self.anchor_list)):
                dense_anchor_list = np.vstack(
                    (dense_anchor_list, self.anchor_list[i]))
            anchor_list = dense_anchor_list

        for i in range(self.config.IMAGES_PER_GPU):
            decode_data = decoder(
                self.anchor_list, loc_output[i], cls_output[i], self.config)
            decode_data_list.append(decode_data)
        return decode_data_list

    def calc_loss(self, y_pred, y_true):
        """
        Loss calculation
            loc : foreground
            cls : background/foreground
        ARGS:
            - y_pred : logists output from network
                shape => Tensor(batch_size,num_anchor,12),Tensor(batch_size,num_anchor,num_classes)
            - y_true : encode data
                shape => Tensor(batch_size,num_anchor,12),Tensor(batch_size,num_anchor,1)
        RETURN:
            - loc_loss : float, location loss
            - cls_loss : float, class loss
            - tvars : list, trainable_variables
            - extra_update_ops
        """
        loc_preds, cls_preds = y_pred
        loc_gt, cls_gt = y_true

        ############################
        # Location Regression loss #
        ############################
        # get valid loc indice
        # skip negative and ignored anchors
        # valid => 1
        # invalid => 0
        # ignored=> -1

        # valid_anchor_indices => shape = [num_valid_anchor, 3]
        # 3 numbers to describe "True" index in => [batch,num_anchor,indice], where indice=1
        valid_anchor_indices = tf.where(tf.greater(cls_gt, 0))
        # number of positive anchors
        gt_anchor_nums = tf.shape(valid_anchor_indices)[0]  # num_valid_anchor
        # shape = (num_valid_anchor,12)
        valid_loc_preds = tf.gather_nd(loc_preds, valid_anchor_indices)
        # shape = (num_valid_anchor,12)
        valid_loc_gt = tf.gather_nd(loc_gt, valid_anchor_indices)

        loc_loss = regression_loss_graph(valid_loc_preds, valid_loc_gt)
        loc_loss = tf.truediv(tf.reduce_sum(loc_loss),
                              tf.to_float(gt_anchor_nums))  # mean the loss
        # avoid no valid data
        loc_loss = tf.where(tf.shape(valid_loc_gt)[
                            0] > 0, loc_loss, tf.constant(0.0))

        loc_loss *= self.config.LOSS_WEIGHTS["loc_loss"]

        #######################
        # Classification loss #
        #######################
        # get valid cls indice
        # skip ignored anchors (iou belong to 0.4 to 0.5)
        # valid => 0,1
        # ignored => -1

        # valid_cls_indices => shape = [num_valid_anchor, 3]
        # 3 numbers to describe "True" index in => [batch,num_anchor,indice], where indice=1
        # but only require => [batch,num_anchor]
        valid_cls_indices = tf.where(tf.greater(cls_gt, -1))
        # shape = (num_valid_anchor,2)
        valid_cls_preds = tf.gather_nd(cls_preds, valid_cls_indices[:, :2])
        # shape = (num_valid_anchor, 1)
        valid_cls_gt = tf.gather_nd(cls_gt, valid_cls_indices[:, :2])

        cls_loss = focal_loss_graph(valid_cls_preds, valid_cls_gt)
        cls_loss = tf.truediv(tf.reduce_sum(cls_loss),
                              tf.to_float(gt_anchor_nums))  # mean the loss
        # avoid no valid data
        cls_loss = tf.where(tf.shape(valid_cls_gt)[
                            0] > 0, cls_loss, tf.constant(0.0))

        cls_loss *= self.config.LOSS_WEIGHTS["class_loss"]

        ####################################
        # Variables for multi-GPU training #
        ####################################
        tvars = tf.trainable_variables()
        if self.config.NUM_GPU > 1:
            scope = self.scope or self.tune_scope
            scope = '|'.join(['train_[0-9]+/' + s for s in scope.split('|')])

            tvars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        return loc_loss, cls_loss, tvars, extra_update_ops

    def data_generator(self, dataset_path, batch_size, augment=False):
        """ load ground truth and prepare training data

        Preprocess:
            Step 0 : (Done when init) Generate Anchors
            Step 1 : Read ground truth data from tfrecord dataset
            Step 2 : Augmentation
            Step 3 : Encode Anchors with ground truth
            Step 4 : Make Batch

        ARGS:
             - dataset_path : str, path to tfrecord
             - batch_size : int, IMAGES_PER_GPU if NUM_GPU==1 else BATCH_SIZE
             - augment : bool

        RETURN:
            - batch_image : Tensor(batch_size, image_width, image_height, 3)
            - batch_loc : Tensor(batch_size, num_anchors, 12)
            - batch_cls : Tensor(batch_size, num_anchors, 1)
            - batch_gt_boxes : Tensor(batch_size, num_ground_truth, 13)
        """
        data_list = read_tfrecord(dataset_path, self.config, shuffle=True)
        image = data_list[0]
        boxes = data_list[1:13]
        label = data_list[13:]

        if augment:
            image, boxes = random_horizontal_flip(image, boxes)
            image, boxes = random_vertical_flip(image, boxes)
            image = normalize_image(image)
            image = random_adjust_brightness(image)
            image = random_adjust_contrast(image)
            image = random_adjust_hue(image)
            image = random_adjust_saturation(image)
        else:
            image = normalize_image(image)

        gt_pair = encoder(self.anchor_list, boxes, label, self.config)

        batch_image, batch_loc, batch_cls, batch_gt_boxes, batch_gt_labels = tf.train.batch([image, gt_pair[:, :12], gt_pair[:, 12:], boxes, label],
                                                                                            dynamic_pad=True,
                                                                                            batch_size=batch_size,
                                                                                            capacity=self.config.CAPACITY,
                                                                                            num_threads=self.config.NUM_THREADS)
        return batch_image, batch_loc, batch_cls, batch_gt_boxes, batch_gt_labels
