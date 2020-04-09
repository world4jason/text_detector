import tensorflow as tf
import numpy as np
import collections

from network.layers import *
from network.utils.utils import *
from network.utils.data_utils import *
from network.augment import *


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
        ## stage 1
        C1 = conv_layer(inputs, 64, kernel_size=7, strides=2)
        C1 = norm_layer(C1, training, use_bn)
        C1 = pool_layer(C1, (3, 3), stride=(2, 2))

        ## stage2
        C2 = res_block_v1(C1, [64, 64, 256], training, use_bn, strides=1, downsample=True)
        for i in range(blocks[0] - 1):
            C2 = res_block_v1(C2, [64, 64, 256], training, use_bn)

        ## stage3
        C3 = res_block_v1(C2, [128, 128, 512], training, use_bn, strides=2, downsample=True)
        for i in range(blocks[1] - 1):
            C3 = res_block_v1(C3, [128, 128, 512], training, use_bn)

        ## stage4
        C4 = res_block_v1(C3, [256, 256, 1024], training, use_bn, strides=2, downsample=True)
        for i in range(blocks[2] - 1):
            C4 = res_block_v1(C4, [256, 256, 1024], training, use_bn)

        ## stage5
        C5 = res_block_v1(C4, [512, 512, 2048], training, use_bn, strides=2, downsample=True)
        for i in range(blocks[3] - 1):
            C5 = res_block_v1(C5, [512, 512, 2048], training, use_bn)

        return C1, C2, C3, C4, C5

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
    C1, C2, C3, C4, C5 = resnet_feature_map
    with tf.variable_scope("FPN"):  # TODO: check FPN for ReinaNet
        P5_up = conv_layer(C5, 256, kernel_size=1)
        P4_up = upsampling(P5_up, size=(2, 2)) + conv_layer(C4, 256, kernel_size=1)
        P3_up = upsampling(P4_up, size=(2, 2)) + conv_layer(C3, 256, kernel_size=1)

        P5 = conv_layer(P5_up, 256, kernel_size=3)
        P4 = conv_layer(P4_up, 256, kernel_size=3)
        P3 = conv_layer(P3_up, 256, kernel_size=3)

        P6 = conv_layer(C5, 256, kernel_size=3, strides=2)
        P7 = relu(P6)
        P7 = conv_layer(P7, 256, kernel_size=3, strides=2)

        if C2_mode:
            assert len(config.FEATURE_STRIDES)==6
            P2 = upsampling(P3_up, size=(2, 2)) + conv_layer(C2, 256, kernel_size=1)
            P2 = conv_layer(P2, 256, kernel_size=3)
            return P2, P3, P4, P5, P6, P7

    return P3, P4, P5, P6, P7

############################################################
#  Task Specific Layer
############################################################

def cls_task_head(feature, out_dims, scope, probability=0.01):
    with tf.variable_scope(scope):
        _kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        for _ in range(4):
            feature = conv_layer(feature, 256, kernel_size=3, use_bias=False, kernel_initializer=_kernel_initializer)
            feature = relu(feature)

        feature = conv_layer(feature, out_dims, kernel_size=(3, 5), kernel_initializer=_kernel_initializer)
        bias_initial = tf.ones(out_dims, dtype=tf.float32) * -tf.log((1 - probability) / probability)
        output_layer = tf.nn.bias_add(feature, bias_initial)
        return output_layer

def loc_task_head(feature, out_dims, scope, probability=0.01):
    with tf.variable_scope(scope):
        _kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        for _ in range(4):
            feature = conv_layer(feature, 256, kernel_size=3, use_bias=False, kernel_initializer=_kernel_initializer)
            feature = relu(feature)
        output_layer=conv_layer(feature, out_dims, kernel_size=(3, 5), kernel_initializer=_kernel_initializer)
        return output_layer

############################################################
#  Loss Functions
############################################################

def regression_loss_graph(pred_boxes, gt_boxes, weights=1.0):
    """Regression loss (Smooth L1 loss (=huber loss))

    ARGS:
        pred_boxes: [# anchors, 4]
        gt_boxes: [# anchors, 4]
        weights: Tensor of weights multiplied by loss with shape [# anchors]
    RETURN:

    """
    with tf.variable_scope("regression_loss"):
        x = tf.abs(pred_boxes-gt_boxes)
        x = tf.where(tf.less(x, 1.0), 0.5*x**2, x-0.5)
        x = tf.reduce_sum(x)
        return x
def regression_loss_graph(pred_boxes, gt_boxes, weights=1.0):
    """Regression loss

    ARGS:
        pred_boxes: [# anchors, 4]
        gt_boxes: [# anchors, 4]
        weights: Tensor of weights multiplied by loss with shape [# anchors]
    RETURN:

    """
    x = tf.abs(pred_boxes-gt_boxes)
    x = tf.where(tf.less(x, 1.0), 0.5*x**2, x-0.5)
    x = tf.reduce_sum(x)
    return x

def huber_loss_graph(pred_boxes, gt_boxes, cilp_value=1.0):
    """Regression loss

    ARGS:
        pred_boxes: [# anchors, 4]
        gt_boxes: [# anchors, 4]
        weights: Tensor of weights multiplied by loss with shape [# anchors]
    RETURN:

    """
    x = tf.abs(pred_boxes-gt_boxes)
    x = tf.where(tf.less(x, cilp_value), 0.5*x**2, (x-0.5*cilp_value)*cilp_value)
    x = tf.reduce_sum(x)
    return x

def smooth_l1_loss_graph(pred_boxes, gt_boxes, weights=1.0):
    """Regression loss (Smooth L1 loss (=huber loss))
            pred_boxes: [# anchors, 4]
            gt_boxes: [# anchors, 4]
            weights: Tensor of weights multiplied by loss with shape [# anchors]
    """
    with tf.variable_scope("smooth_l1_loss"):
        return tf.reduce_sum(tf.losses.huber_loss(
            gt_boxes,
            pred_boxes,
            delta=1,
            weights=tf.expand_dims(weights, axis=2),
            loss_collection=None,
            reduction=tf.losses.Reduction.NONE
        ), axis=2)

def focal_loss_graph(preds_cls, gt_cls,
                config, alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute sigmoid focal loss between logits and onehot labels"""

    gt_cls = tf.one_hot(indices=tf.cast(gt_cls,tf.int32), depth=config.NUM_CLASSES) #shape = (val_anchor, 1, num_class)

    preds_cls = tf.nn.sigmoid(preds_cls)            #shape = (val_anchor, num_class)
    gt_cls = tf.reshape(gt_cls,tf.shape(preds_cls)) #shape: (val_anchor,1, num_class)=>(val_anchor, num_class)

    # cross-entropy -> if y=1 : pt=p / otherwise : pt=1-p
    predictions_pt = tf.where(tf.equal(gt_cls, 1), preds_cls, 1.0 - preds_cls) #shape = (val_anchor, num_class)

    # clip small value to avoid 0
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(predictions_pt, dtype=tf.float32)) #shape = (val_anchor, num_class)
    alpha_t = tf.where(tf.equal(gt_cls, 1.0), alpha_t, 1.0 - alpha_t) #shape = (val_anchor, num_class)
    gamma_t = tf.scalar_mul(gamma, tf.ones_like(predictions_pt, tf.float32)) #shape = (val_anchor, num_class)

    focal_losses = alpha_t * (-tf.pow(1.0 - predictions_pt, gamma_t) * tf.log(tf.clip_by_value(predictions_pt,1e-10,1.0))) #shape = (val_anchor, num_class)

    focal_losses = tf.reduce_sum(focal_losses, axis=-1) #shape = (val_anchor,)
    return focal_losses




def focal_loss_graph(preds_cls, gt_cls,
                config, alpha=0.25, gamma=2.0, name=None, scope=None):
    """Compute sigmoid focal loss between logits and onehot labels

    ARGS:


    RETURNS:


    """
    # prepare internal required value (alpha, gamma)
    condition = tf.equal(tf.reshape(gt_cls,[-1]), 1.0)
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(preds_cls, dtype=tf.float32)) #shape = (val_anchor, num_class)
    alpha_t = tf.where(condition, alpha_t, 1.0 - alpha_t)  #shape = (val_anchor, num_class)
    gamma_t = tf.scalar_mul(gamma, tf.ones_like(preds_cls, tf.float32)) #shape = (val_anchor, num_class)

    preds_cls = tf.nn.sigmoid(preds_cls)            #shape = (val_anchor, num_class)

    # binary cross entropy -> if y=1 : pt=p /
    #                         otherwise : pt=1-p
    predictions_pt = tf.where(condition, preds_cls, 1.0 - preds_cls) #shape = (val_anchor, num_class)

    # clip to avoid 0
    focal_losses = alpha_t * (-tf.pow(1.0 - predictions_pt, gamma_t) * tf.log(tf.clip_by_value(predictions_pt,1e-10,1.0))) #shape = (val_anchor, num_class)

    focal_losses = tf.reduce_sum(focal_losses, axis=-1) #shape = (val_anchor,)
    return focal_losses
############################################################
#  TextBoxes++ Network
############################################################

class TextBoxesNet():
    def __init__(self, config):

        # Set tune scope
        self.scope="resnet_backbone|fpn|task_head"

        assert config.BACKBONE in resnet_version.keys()
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
                                                self.config.ASPECT_RATIOS)

    def forward(self, image, **kwargs):
        """Forwarding and get logits output

        ARGS:
            input : [batch, image_width, image_height, channel=3]
            mode :
        RETURN:
            logits :  [batch,num_anchors,4+8],[batch,num_anchors,num_classes]
        """
        features_resnet = resnet_graph(backbone="resnet101", training=True,inputs=image, use_bn=True)
        features = fpn_graph(features_resnet, self.config, C2_mode=False)

        with tf.variable_scope("task_head"):
            loc_subnet = []
            class_subnet = []
            loc_output_dims = self.config.NUM_ANCHOR_PER_GRID * (4 + 8)
            cls_output_dims = self.config.NUM_ANCHOR_PER_GRID  * self.config.NUM_CLASSES

            for n, feature in enumerate(features):

                _loc = cls_task_head(feature, loc_output_dims, "P%d_loc_head" % (n+3),self.probability)
                _loc = tf.reshape(_loc, [self.config.IMAGES_PER_GPU, -1, 4 + 8])

                _class = loc_task_head(feature, cls_output_dims , "P%d_cls_head" % (n+3))
                _class = tf.reshape(_class, [self.config.IMAGES_PER_GPU, -1, self.config.NUM_CLASSES])

                loc_subnet.append(_loc)
                class_subnet.append(_class)

            loc_subnet = tf.concat(loc_subnet, axis=1) #shape = (batch_size,anchor_size,12)
            class_subnet = tf.concat(class_subnet, axis=1) #shape = (batch_size,anchor_size,2)
            logits = tf.concat([loc_subnet,class_subnet],axis=-1) #shape = (batch_size,anchor_size,14)
            return logits
    def decode(self, logists):
        """

        ARGS:

        RETURN:

        """
        decode_data_list = []
        # anchor_list shape normalization
        if np.array(self.anchor_list).ndim!=2:
            dense_anchor_list = self.anchor_list[0]
            for i in range(1,len(self.anchor_list)):
                dense_anchor_list = np.vstack((dense_anchor_list,self.anchor_list[i]))
            anchor_list=dense_anchor_list

        for i in range(self.config.IMAGES_PER_GPU):
            decode_data = decoder(self.anchor_list, logists[i], self.config)
            decode_data_list.append(decode_data)
        return decode_data_list

    def calc_loss(self, y_pred, y_true):
        """
        Loss calculation
            loc : foreground
            cls : background/foreground

        ARGS:

        RETURN:

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
        # shape = [num_valid_anchor,3]
        # 3 numbers to describe "True" index in => [batch,num_anchor,indice], where indice=1
        valid_anchor_indices = tf.where(tf.greater(cls_gt, 0))
        # number of positive anchors
        gt_anchor_nums = tf.shape(valid_anchor_indices)[0] # num_valid_anchor
        valid_loc_preds = tf.gather_nd(loc_preds, valid_anchor_indices)
        valid_loc_gt = tf.gather_nd(loc_gt, valid_anchor_indices)

        loc_loss = regression_loss_graph(valid_loc_preds, valid_loc_gt)
        loc_loss = tf.truediv(tf.reduce_sum(loc_loss), tf.to_float(gt_anchor_nums)) # mean the loss
        # avoid no valid data
        loc_loss = tf.where(tf.shape(valid_loc_gt)[0]>0, loc_loss, tf.constant(0.0))

        loc_loss *= self.config.LOSS_WEIGHTS["loc_loss"]

        #######################
        # Classification loss #
        #######################
        # get valid cls indice
        # skip ignored anchors (iou belong to 0.4 to 0.5)
        # valid => 0,1
        # ignored => -1
        # 3 numbers to describe "True" index in => [batch,num_anchor,1]
        # but only require => [batch,num_anchor]
        valid_cls_indices = tf.where(tf.greater(cls_gt, -1))
        valid_cls_preds = tf.gather_nd(cls_preds, valid_cls_indices[:,:2])
        valid_cls_gt = tf.gather_nd(cls_gt, valid_cls_indices[:,:2])

        cls_loss = focal_loss_graph(valid_cls_preds, valid_cls_gt, self.config)
        cls_loss =  tf.truediv(tf.reduce_sum(cls_loss), tf.to_float(gt_anchor_nums))# mean the loss
        # avoid no valid data
        cls_loss = tf.where(tf.shape(valid_cls_gt)[0]>0, cls_loss,tf.constant(0.0))

        cls_loss *= self.config.LOSS_WEIGHTS["class_loss"]

        ####################################
        # Variables for multi-GPU training #
        ####################################
        scope = self.scope or self.tune_scope
        scope = '|'.join(['train_[0-9]+/' + s for s in scope.split('|')])

        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        return loc_loss, cls_loss, tvars, extra_update_ops

    def data_generator(self, dataset_path, batch_size, augment=False):
        """
        load_img_gt

        preprocess:


        ARGS:

        RETURN:

        """
        data_list = read_tfrecord(dataset_path, self.config, shuffle=True)
        image = data_list[0]
        boxes = data_list[1:]

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

        gt_pair =  encoder(self.anchor_list, boxes, self.config)
        """
        batch_image, batch_loc, batch_cls = tf.train.shuffle_batch([image, gt_pair[:,:12],gt_pair[:,:-1]],
                                                                    batch_size=batch_size,
                                                                    capacity=200,
                                                                    min_after_dequeue=100,
                                                                    num_threads=4)
        """
        batch_image, batch_loc, batch_cls, batch_gt_boxes = tf.train.batch([image,gt_pair[:,:12],gt_pair[:,12:],boxes],
                                                                dynamic_pad=True,
                                                                batch_size=self.config.IMAGES_PER_GPU,
                                                                capacity=self.config.CAPACITY,
                                                                num_threads=self.config.NUM_THREADS)
        return batch_image, batch_loc, batch_cls, batch_gt_boxes




