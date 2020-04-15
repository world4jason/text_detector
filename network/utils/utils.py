from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from shapely.geometry import Polygon

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import tf_export

import os
import sys
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# find root dir
sys.path.append(ROOT_DIR)

from network.optimizer import RAdam, Ranger

############################################################
#  Tensorflow funtion Porting
############################################################
#
# tf version 1.4 dont have sort and argsort
# Code is from tf 1.8 official
#
@tf_export('sort')
def sort(values, axis=-1, direction='ASCENDING', name=None):
  """Sorts a tensor.

  Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.sort(a,axis=-1,direction='ASCENDING',name=None)
  c = tf.keras.backend.eval(b)
  # Here, c = [  1.     2.8   10.    26.9   62.3  166.32]
  ```

  Args:
    values: 1-D or higher numeric `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    name: Optional name for the operation.
  Returns:
    A `Tensor` with the same dtype and shape as `values`, with the elements
        sorted along the given `axis`.
  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  """
  with framework_ops.name_scope(name, 'sort'):
    return _sort_or_argsort(values, axis, direction, return_argsort=False)


@tf_export('argsort')
def argsort(values, axis=-1, direction='ASCENDING', stable=False, name=None):
    """Returns the indices of a tensor that give its sorted order along an axis.
    For a 1D tensor, `tf.gather(values, tf.argsort(values))` is equivalent to
    `tf.sort(values)`. For higher dimensions, the output has the same shape as
    `values`, but along the given axis, values represent the index of the sorted
    element in that slice of the tensor at the given position.

    Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.argsort(a,axis=-1,direction='ASCENDING',stable=False,name=None)
    c = tf.keras.backend.eval(b)
    # Here, c = [0 3 1 2 5 4]
    ```
    Args:
    values: 1-D or higher numeric `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    stable: If True, equal elements in the original tensor will not be
      re-ordered in the returned order. Unstable sort is not yet implemented,
      but will eventually be the default for performance reasons. If you require
      a stable order, pass `stable=True` for forwards compatibility.
    name: Optional name for the operation.
    Returns:
    An int32 `Tensor` with the same shape as `values`. The indices that would
        sort each slice of the given `values` along the given `axis`.
    Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
    """
    del stable  # Unused.
    with framework_ops.name_scope(name, 'argsort'):
        return _sort_or_argsort(values, axis, direction, return_argsort=True)


def _sort_or_argsort(values, axis, direction, return_argsort):
    """Internal sort/argsort implementation.
    Args:
    values: The input values.
    axis: The axis along which to sort.
    direction: 'ASCENDING' or 'DESCENDING'.
    return_argsort: Whether to return the argsort result.
    Returns:
    Either the sorted values, or the indices of the sorted values in the
        original tensor. See the `sort` and `argsort` docstrings.
    Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
    """
    if direction not in _SORT_IMPL:
        raise ValueError('%s should be one of %s' % (direction, ', '.join(
            sorted(_SORT_IMPL.keys()))))
    # Axis must be an integer, not a Tensor.
    axis = framework_ops.convert_to_tensor(axis, name='axis')
    axis_static = tensor_util.constant_value(axis)
    if axis.shape.ndims != 0 or axis_static is None:
        raise ValueError('axis must be a constant scalar')
    axis_static = int(axis_static)  # Avoids NumPy casting error

    values = framework_ops.convert_to_tensor(values, name='values')

    return _SORT_IMPL[direction](values, axis_static, return_argsort)


def _descending_sort(values, axis, return_argsort=False):
    """Sorts values in reverse using `top_k`.
    Args:
    values: Tensor of numeric values.
    axis: Index of the axis which values should be sorted along.
    return_argsort: If False, return the sorted values. If True, return the
      indices that would sort the values.
    Returns:
    The sorted values.
    """
    k = array_ops.shape(values)[axis]
    rank = array_ops.rank(values)
    static_rank = values.shape.ndims
    # Fast path: sorting the last axis.
    if axis == -1 or axis + 1 == values.get_shape().ndims:
        top_k_input = values
        transposition = None
    else:
        # Otherwise, transpose the array. Swap axes `axis` and `rank - 1`.
        if axis < 0:
            # Calculate the actual axis index if counting from the end. Use the static
            # rank if available, or else make the axis back into a tensor.
            axis += static_rank or rank
        if static_rank is not None:
            # Prefer to calculate the transposition array in NumPy and make it a
            # constant.
            transposition = constant_op.constant(
              np.r_[
                  # Axes up to axis are unchanged.
                  np.arange(axis),
                  # Swap axis and rank - 1.
                  [static_rank - 1],
                  # Axes in [axis + 1, rank - 1) are unchanged.
                  np.arange(axis + 1, static_rank - 1),
                  # Swap axis and rank - 1.
                  [axis]],
              name='transposition')
        else:
            # Generate the transposition array from the tensors.
            transposition = array_ops.concat(
              [
                  # Axes up to axis are unchanged.
                  math_ops.range(axis),
                  # Swap axis and rank - 1.
                  [rank - 1],
                  # Axes in [axis + 1, rank - 1) are unchanged.
                  math_ops.range(axis + 1, rank - 1),
                  # Swap axis and rank - 1.
                  [axis]
              ],
              axis=0)
    top_k_input = array_ops.transpose(values, transposition)

    values, indices = nn_ops.top_k(top_k_input, k)
    return_value = indices if return_argsort else values
    if transposition is not None:
        # transposition contains a single cycle of length 2 (swapping 2 elements),
        # so it is an involution (it is its own inverse).
        return_value = array_ops.transpose(return_value, transposition)
    return return_value


def _ascending_sort(values, axis, return_argsort=False):
  # Negate the values to get the ascending order from descending sort.
  values_or_indices = _descending_sort(-values, axis, return_argsort)
  # If not argsort, negate the values again.
  return values_or_indices if return_argsort else -values_or_indices


_SORT_IMPL = {
    'ASCENDING': _ascending_sort,
    'DESCENDING': _descending_sort,
}


############################################################
#  Optimizer
############################################################

def get_optimizer(config, learning_rate=None):
    if learning_rate==None:
        learning_rate = config.LEARNING_RATE

    if config.OPTIMIZER=="Ranger":
        return Ranger(learning_rate=learning_rate,
                            beta1=0.90,
                            epsilon=1e-8)
    elif config.OPTIMIZER=="RAdam":
        return RAdamOptimizer(
                    learning_rate=learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-6,
                    decay=0.,
                    warmup_proportion= 0.1,
                    weight_decay=0.,
                    amsgrad=False,
                )
    elif config.OPTIMIZER=="Adam":
        return tf.train.AdamOptimizer(learning_rate)
    elif config.OPTIMIZER=="Momentum":
        return tf.train.MomentumOptimizer(learning_rate, config.LEARNING_MOMENTUM)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate)



############################################################
#  Anchors Boxes
############################################################
def gen_anchor_boxes(image_width,image_height,
                     anchor_vertical_step,
                     anchor_horizontal_step,
                     feature_stride,
                     anchor_area,
                     anchor_ratios):
    """
    Generate anchors of sinlgle layer
    ARGS:
        - image_width : int, image width of input image
        - image_height : int, image height of input image
        - anchor_vertical_step : float, vertical offset in anchor grid setting
        - anchor_horizontal_step : float, horizontal offset in anchor grid setting
        - feature_stride : int, output scale from fpn
        - anchor_area : int array, an array of anchor size in pixel
        - anchor_ratios : float array, an array of anchor ratio
    RETURN :
        shape=> np.array(num_anchor_a_layer,12)
    """
    scales = np.tile(np.array(anchor_area),len(anchor_ratios)).flatten()
    ratios = anchor_ratios

    # normal coord
    # center x, center y
    x = (np.arange(0, image_width/feature_stride,anchor_horizontal_step)+0.5)*feature_stride
    y = (np.arange(0, image_height/feature_stride,anchor_vertical_step)+0.5)*feature_stride
    cx, cy = np.meshgrid(x,y)

    # width, height
    heights = scales / np.sqrt(ratios)
    widths  = scales * np.sqrt(ratios)
    heights_set,cy_set = np.meshgrid(heights,cy)
    widths_set,cx_set  = np.meshgrid(widths,cx)

    # quad coord
    # top-left clockwise
    x0_set = np.array([np.clip(i-widths/2,0,image_width) for i in cx.flatten()])
    x1_set = np.array([np.clip(i+widths/2,0,image_width) for i in cx.flatten()])
    x2_set = np.array([np.clip(i+widths/2,0,image_width) for i in cx.flatten()])
    x3_set = np.array([np.clip(i-widths/2,0,image_width) for i in cx.flatten()])

    y0_set = np.array([np.clip(i-heights/2,0,image_height) for i in cy.flatten()])
    y1_set = np.array([np.clip(i-heights/2,0,image_height) for i in cy.flatten()])
    y2_set = np.array([np.clip(i+heights/2,0,image_height) for i in cy.flatten()])
    y3_set = np.array([np.clip(i+heights/2,0,image_height) for i in cy.flatten()])

    return np.concatenate((np.expand_dims(x0_set.flatten(), axis=-1),
                          np.expand_dims(y0_set.flatten(), axis=-1),
                          np.expand_dims(x1_set.flatten(), axis=-1),
                          np.expand_dims(y1_set.flatten(), axis=-1),
                          np.expand_dims(x2_set.flatten(), axis=-1),
                          np.expand_dims(y2_set.flatten(), axis=-1),
                          np.expand_dims(x3_set.flatten(), axis=-1),
                          np.expand_dims(y3_set.flatten(), axis=-1),
                          np.expand_dims(cx_set.flatten(), axis=-1),
                          np.expand_dims(cy_set.flatten(), axis=-1),
                          np.expand_dims(widths_set.flatten(), axis=-1),
                          np.expand_dims(heights_set.flatten(), axis=-1),
                          ),axis=-1)

def pyramid_gen_anchor_boxes(image_width,
                            image_height,
                            ANCHOR_VERTICAL_STEP,
                            ANCHOR_HORIZON_STEP,
                            FEATURE_STRIDES,
                            ANCHOR_AREAS,
                            ANCHOR_RATIOS,
                            dense=True):
    """
    Dense is for training and inference purpose, another is for debug purpose
    ARGS:
        - image_width : int, image width of input image
        - image_height : int, image height of input image
        - ANCHOR_VERTICAL_STEP : float, vertical offset in anchor grid setting
        - ANCHOR_HORIZON_STEP : float, horizontal offset in anchor grid setting
        - FEATURE_STRIDES : int array, an array of output scale from fpn
        - ANCHOR_AREAS : int array, an array of anchor size in pixel
        - anchor_ratios : float array, an array of anchor ratio
        - dense : bool, False when debug mode
    RETURN :
        shape=> np.array(num_anchor, 12)
    """
    if dense:
        pyramid_anchors = np.array([])
        for FEATURE_STRIDE,ANCHOR_AREA in zip(FEATURE_STRIDES,ANCHOR_AREAS):
            pyramid_anchors = np.append(pyramid_anchors,gen_anchor_boxes(image_width,
                                                    image_height,
                                                    ANCHOR_VERTICAL_STEP,
                                                    ANCHOR_HORIZON_STEP,
                                                    FEATURE_STRIDE,
                                                    ANCHOR_AREA,
                                                    ANCHOR_RATIOS))

        return np.reshape(pyramid_anchors,(-1,12))

    pyramid_anchors = []
    for FEATURE_STRIDE,ANCHOR_AREA in zip(FEATURE_STRIDES,ANCHOR_AREAS):
        pyramid_anchors.append(gen_anchor_boxes(image_width,
                                                 image_height,
                                                 ANCHOR_VERTICAL_STEP,
                                                 ANCHOR_HORIZON_STEP,
                                                 FEATURE_STRIDE,
                                                 ANCHOR_AREA,
                                                 ANCHOR_RATIOS))
    return pyramid_anchors

############################################################
#  Bounding Boxes Encode
############################################################


def scale2coord(unscale_data, image_shape):
    """
    Scale coordination from related to absolute.
    """
    image_width, image_height = image_shape
    y0,x0,y1,x1,y2,x2,y3,x3,cx,cy,w,h = unscale_data

    x0,x1,x2,x3,cx,w = [tf.cast(tf.clip_by_value(tf.round(tf.multiply(i,image_width)) ,0,image_width),tf.float32) for i in [x0,x1,x2,x3,cx,w]]
    y0,y1,y2,y3,cy,h = [tf.cast(tf.clip_by_value(tf.round(tf.multiply(i,image_height)) ,0,image_height),tf.float32) for i in [y0,y1,y2,y3,cy,h]]

    return x0,y0,x1,y1,x2,y2,x3,y3,cx,cy,w,h

def center2point(center_x, center_y, width, height):
    """
    Turn xywh format to xyxy format

    RETURN:
        top_left_x, top_left_y, right_down_x, right_down_y
    """
    return tf.truediv(center_x - width , 2.), tf.truediv(center_y - height, 2.), tf.truediv(center_x + width , 2.),tf.truediv(center_y + height, 2.)

def _encode_v1(anchor_list, gt_box_list,gt_label, config):
    """
    1. Calculate IoU
    2. Match Strategy
    3. Encode

    ARGS:


    RETURN :

    """
    image_width, image_height = config.IMAGE_SHAPE

    # ground truth data order
    # gt_x0,gt_y0,gt_x1,gt_y1,gt_x2,gt_y2,gt_x3,gt_y3,gt_cx,gt_cy,gt_w,gt_h
    tf_gt = scale2coord(gt_box_list, (image_width, image_height))
    gt_xmin,gt_ymin,gt_xmax,gt_ymax = center2point(tf_gt[8],tf_gt[9],tf_gt[10],tf_gt[11])

    # anchors_list
    # shape=(num_feature_maps, anchor_points, 12+1)
    targets_list = []
    for index in range(len(anchor_list)):

        # anchor bbox data order
        # a_x0,a_y0,a_x1,a_y1,a_x2,a_y2,a_x3,a_y3,a_cx,a_cy,a_w,a_h
        tf_anchors = tf.cast(tf.transpose(anchor_list[index]),tf.float32)
        a_xmin,a_ymin,a_xmax,a_ymax = center2point(tf_anchors[8],tf_anchors[9],tf_anchors[10],tf_anchors[11])

        #################
        #               #
        # Iou Calculate #
        #               #
        #################
        #intersection
        int_xmin = tf.maximum(tf.expand_dims(a_xmin, 1), tf.transpose(tf.expand_dims(gt_xmin, 1)))
        int_ymin = tf.maximum(tf.expand_dims(a_ymin, 1), tf.transpose(tf.expand_dims(gt_ymin, 1)))
        int_xmax = tf.minimum(tf.expand_dims(a_xmax, 1), tf.transpose(tf.expand_dims(gt_xmax, 1)))
        int_ymax = tf.minimum(tf.expand_dims(a_ymax, 1), tf.transpose(tf.expand_dims(gt_ymax, 1)))
        w = tf.maximum(int_xmax - int_xmin, 0)
        h = tf.maximum(int_ymax - int_ymin, 0)
        intersection = w*h #shape=[num_anchor,num_gt]

        #area
        areas1 = (a_xmax - a_xmin)*(a_ymax - a_ymin)      #shape=[num_anchor]
        areas2 = (gt_xmax - gt_xmin)*(gt_ymax - gt_ymin) #shape=[num_gt]

        #union
        unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersection) #shape=[num_anchor,num_gt]

        #iou
        iou_condition = tf.equal(intersection, 0.0)              #shape=[num_anchor,num_gt]
        condition_a = tf.zeros_like(intersection)                #shape=[num_anchor,num_gt]
        condition_b = tf.truediv(intersection, unions)           #shape=[num_anchor,num_gt]
        iou  = tf.where(iou_condition, condition_a, condition_b) #shape=[num_anchor,num_gt]

        ##################
        #                #
        # Match Strategy #
        #                #
        ##################
        max_ids = tf.argmax(iou, axis=1) #shape=[num_anchor]
        max_ious = tf.reduce_max(iou, axis=1)#shape=[num_anchor]


        gt_quad_set = tf.transpose(tf_gt[:8]) #shape=[num_gt, 8]
        gt_rect_set = tf.transpose(tf_gt[8:]) #shape=[num_gt, 4]


        gt_quad_boxes = tf.gather(gt_quad_set, max_ids) #shape=[num_anchor, 8]
        gt_rect_boxes = tf.gather(gt_rect_set, max_ids) #shape=[num_anchor, 4]

        # offset calculation
        tf_anchors_transpose = tf.transpose(tf_anchors)

        # Encoding for Rectangle boxes
        loc_rect_xy = (gt_rect_boxes[:, :2] - tf_anchors_transpose[:, 8:10]) / tf_anchors_transpose[:,10:] #shape=[num_anchor, 2]
        loc_rect_wh = tf.log(gt_rect_boxes[:, :2] / (tf_anchors_transpose[:, 10:])) #shape=[num_anchor, 2]
        loc_rect_xy = (loc_rect_xy / config.RECT_BBOX_XY_PRIOR_VARIANCE) if config.PRIOR_VARIANCE else loc_rect_xy
        loc_rect_wh = (loc_rect_wh / config.RECT_BBOX_WH_PRIOR_VARIANCE) if config.PRIOR_VARIANCE else loc_rect_wh

        #  Encoding for Quad boxes
        anchor_boxes_wh = tf.tile(tf_anchors_transpose[:, 10:], [1, 4]) #shape=[num_anchor, 8]
        loc_quad_xy = (gt_quad_boxes -  tf_anchors_transpose[:, :8]) / anchor_boxes_wh  #shape=[num_anchor, 8]
        loc_quad_xy = (loc_quad_xy / config.QUAD_BBOX_PRIOR_VARIANCE) if config.PRIOR_VARIANCE else loc_quad_xy

        # calc for class
        cls_targets = tf.gather(tf.one_hot(tf.transpose(gt_label), config.NUM_CLASSES), max_ids) # shape=[num_anchor, num_classes=2]

        # iou > THRESHOLD :keep(1)
        # iou < THRESHOLD : background(0)
        # OBJECTS_THRESHOLD < iou < BACKGROUND_THRESHOLD : ignore(-1)
        cls_targets = tf.where(tf.less(max_ious, config.OBJECTS_THRESHOLD), -tf.ones_like(cls_targets), cls_targets)
        cls_targets = tf.where(tf.less(max_ious, config.BACKGROUND_THRESHOLD), tf.zeros_like(cls_targets), cls_targets)

        ###################
        #                 #
        # Outputs prepare #
        #                 #
        ###################
        targets = tf.concat([loc_quad_xy,
                             loc_rect_xy,
                             loc_rect_wh,
                             cls_targets], 1) # shape=[num_anchor, 12+num_classes]
        if index==0:
            targets_list = targets
        else:
            targets_list = tf.concat([targets_list,targets],0)

    return targets_list

def _encode_v2(anchor_list, gt_box_list, gt_label, config):
    """
    1. Calculate IoU
    2. Match Strategy
    3. Encode

    ARGS:
        - anchor_list : np.array([num_feature_maps, num_anchors_on_each_feature_maps,12])
        - gt_box_list:
            shape=>Tensor([num_gt, 12])
            12=> x0_set,y0_set,x1_set,y1_set,x2_set,y2_set,x3_set,y3_set,cx_set,cy_set,w_set,h_set
        - gt_label:Tensor([num_gt, 1])
    RETURN : Paired Data

    """
    image_width, image_height = config.IMAGE_SHAPE

    # ground truth data order
    # gt_x0,gt_y0,gt_x1,gt_y1,gt_x2,gt_y2,gt_x3,gt_y3,gt_cx,gt_cy,gt_w,gt_h
    tf_gt = scale2coord(gt_box_list, (image_width, image_height))
    gt_xmin,gt_ymin,gt_xmax,gt_ymax = center2point(tf_gt[8],tf_gt[9],tf_gt[10],tf_gt[11])

    # anchors_list
    # shape=(num_anchos, 12)
    # anchor bbox data order
    # a_x0,a_y0,a_x1,a_y1,a_x2,a_y2,a_x3,a_y3,a_cx,a_cy,a_w,a_h
    tf_anchors = tf.cast(tf.transpose(tf.convert_to_tensor(anchor_list)),tf.float32)
    a_xmin,a_ymin,a_xmax,a_ymax = center2point(tf_anchors[8],tf_anchors[9],tf_anchors[10],tf_anchors[11])

    #################
    #               #
    # Iou Calculate #
    #               #
    #################
    #intersection
    int_xmin = tf.maximum(tf.expand_dims(a_xmin, 1), tf.transpose(tf.expand_dims(gt_xmin, 1)))
    int_ymin = tf.maximum(tf.expand_dims(a_ymin, 1), tf.transpose(tf.expand_dims(gt_ymin, 1)))
    int_xmax = tf.minimum(tf.expand_dims(a_xmax, 1), tf.transpose(tf.expand_dims(gt_xmax, 1)))
    int_ymax = tf.minimum(tf.expand_dims(a_ymax, 1), tf.transpose(tf.expand_dims(gt_ymax, 1)))
    w = tf.maximum(int_xmax - int_xmin, 0)
    h = tf.maximum(int_ymax - int_ymin, 0)
    intersection = w*h #shape=[num_anchor,num_gt]

    #area
    areas1 = (a_xmax - a_xmin)*(a_ymax - a_ymin)      #shape=[num_anchor]
    areas2 = (gt_xmax - gt_xmin)*(gt_ymax - gt_ymin) #shape=[num_gt]

    #union
    unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersection) #shape=[num_anchor,num_gt]

    #iou
    iou_condition = tf.equal(intersection, 0.0)              #shape=[num_anchor,num_gt]
    condition_a = tf.zeros_like(intersection)                #shape=[num_anchor,num_gt]
    condition_b = tf.truediv(intersection, unions)           #shape=[num_anchor,num_gt]
    iou  = tf.where(iou_condition, condition_a, condition_b) #shape=[num_anchor,num_gt]

    ##################
    #                #
    # Match Strategy #
    #                #
    ##################
    max_ids = tf.argmax(iou, axis=1, name="encode_argmax") #shape=[num_anchor]
    max_ious = tf.reduce_max(iou, axis=1)#shape=[num_anchor]


    gt_quad_set = tf.transpose(tf_gt[:8]) #shape=[num_gt, 8]
    gt_rect_set = tf.transpose(tf_gt[8:]) #shape=[num_gt, 4]


    gt_quad_boxes = tf.gather(gt_quad_set, max_ids) #shape=[num_anchor, 8]
    gt_rect_boxes = tf.gather(gt_rect_set, max_ids) #shape=[num_anchor, 4]

    ###########
    #         #
    # Encode  #
    #         #
    ###########
    # offset calculation
    tf_anchors_transpose = tf.transpose(tf_anchors)

    # Encoding for Rectangle boxes
    loc_rect_xy = (gt_rect_boxes[:, :2] - tf_anchors_transpose[:, 8:10]) / tf_anchors_transpose[:,10:] #shape=[num_anchor, 2]
    loc_rect_wh = tf.log(gt_rect_boxes[:, :2] / (tf_anchors_transpose[:, 10:])) #shape=[num_anchor, 2]
    loc_rect_xy = (loc_rect_xy / config.RECT_BBOX_XY_PRIOR_VARIANCE) if config.PRIOR_VARIANCE else loc_rect_xy
    loc_rect_wh = (loc_rect_wh / config.RECT_BBOX_WH_PRIOR_VARIANCE) if config.PRIOR_VARIANCE else loc_rect_wh

    #  Encoding for Quad boxes
    anchor_boxes_wh = tf.tile(tf_anchors_transpose[:, 10:], [1, 4]) #shape=[num_anchor, 8]
    loc_quad_xy = (gt_quad_boxes -  tf_anchors_transpose[:, :8]) / anchor_boxes_wh  #shape=[num_anchor, 8]
    loc_quad_xy = (loc_quad_xy / config.QUAD_BBOX_PRIOR_VARIANCE) if config.PRIOR_VARIANCE else loc_quad_xy

    # calc for class
    cls_targets = tf.gather(tf.cast(tf.transpose(gt_label),tf.float32), max_ids)

    # iou > THRESHOLD : keep(1)
    # iou < THRESHOLD : background(0)
    #  BACKGROUND_THRESHOLD < iou < OBJECTS_THRESHOLD : ignore(-1)
    assert config.OBJECTS_THRESHOLD > config.BACKGROUND_THRESHOLD
    cls_targets = tf.where(tf.less(max_ious, config.OBJECTS_THRESHOLD), -tf.ones_like(cls_targets), cls_targets)
    cls_targets = tf.where(tf.less(max_ious, config.BACKGROUND_THRESHOLD), tf.zeros_like(cls_targets), cls_targets)

    ###################
    #                 #
    # Outputs prepare #
    #                 #
    ###################
    return tf.concat([loc_quad_xy,
                     loc_rect_xy,
                     loc_rect_wh,
                     cls_targets], 1) # shape=[num_anchor, 12+1]

def encoder(anchor_list, gt_box_list, gt_label_list,config, dense=True):
    """
    v1 is for debug purpose
    v2 is for training and inference purpose
    """
    if dense:
        return _encode_v2(anchor_list, gt_box_list, gt_label_list, config)
    return _encode_v1(anchor_list, gt_box_list, gt_label_list, config)


############################################################
#  Bounding Boxes Decode
############################################################
def polynms(dets, thresh):
    """Pure Python Polygon NMS .

    Args:

    Returns:

    """
    scores = dets[:, 8]
    polygon_list=[Polygon(quad_dets[i,:8].reshape(4,2)).convex_hull for i in range(len(quad_dets))]

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # intersection
        inter_list=[polygon_list[i].intersection(polygon_list[order[j+1]]).area for j in range(len(order)-1)]

        # Union
        union_list = [polygon_list[order[0]].union(polygon_list[order[i+1]]).area for i in range(len(order)-1)]

        #iou
        iou = np.array(inter_list)/np.array(union_list)

        # keep those under the threshold
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]   # cause len(iou)==len(order)-1
    return keep

def decoder(anchor_list, loc_pred, cls_pred, config):
    """
    ARGS:
        - anchor_list: np.array(num_anchor, 12)
        - loc_pred : Tensor(num_anchor,12)
        - cls_pred : Tensor(num_anchor,2)
    RETURN:

    """
    image_width, image_height = config.IMAGE_SHAPE
    tf_anchors = tf.cast(anchor_list,tf.float32)

    #####################
    #                   #
    # Coordinate Decode #
    #                   #
    #####################
    # Get anchor set
    # quad=>[:8]
    # yx  =>[8:10]
    # hw  =>[10:]

    # Get predict set
    loc_quad = loc_pred[:,:8]
    loc_xy = loc_pred[:,8:10]
    loc_wh = loc_pred[:,10:12]

    # Decoding for Rectangle boxes
    rect_xy = loc_xy * tf_anchors[:,10:] + tf_anchors[:, 8:10] #shape=[num_anchor, 2]
    rect_wh = tf.exp(loc_wh) * tf_anchors[:, 10:]  #shape=[num_anchor, 2]

    # prior variance decode
    rect_xy = rect_xy * config.RECT_BBOX_XY_PRIOR_VARIANCE if config.PRIOR_VARIANCE else rect_xy
    rect_wh = rect_wh * config.RECT_BBOX_WH_PRIOR_VARIANCE if config.PRIOR_VARIANCE else rect_wh

    # Prepare format for tf.non_max
    # cx, cy, w, h ==> xmin, ymin, xmax, ymax
    rect_boxes = tf.concat([rect_xy - rect_wh / 2, rect_xy + rect_wh / 2], axis=1)  # shape=[anchors,4]
    rect_boxes_x = tf.clip_by_value(rect_boxes[:,::2], 0, image_width)   #shape=[anchor, 4]
    rect_boxes_y = tf.clip_by_value(rect_boxes[:,1::2], 0, image_height) #shape=[anchor, 4]
    rect_boxes = tf.stack([rect_boxes_x, rect_boxes_y], axis=-1) #shape=[anchor, 2, 2]
    rect_boxes = tf.reshape(rect_boxes,(-1,4)) #shape=[anchor, 4]

    # Decoding for Quad boxes
    quad_boxes = tf_anchors[:, :8] + tf.tile(tf_anchors[:, 10:], [1, 4]) * loc_quad  #shape=[anchor, 8]
    quad_boxes = quad_boxes*config.QUAD_BBOX_PRIOR_VARIANCE if config.PRIOR_VARIANCE else quad_boxes

    quad_boxes_x = tf.clip_by_value(quad_boxes[:,::2], 0, image_width)  #shape=[anchor, 4]
    quad_boxes_y = tf.clip_by_value(quad_boxes[:,1::2], 0, image_height)#shape=[anchor, 4]
    quad_boxes = tf.stack([quad_boxes_x, quad_boxes_y], axis=-1) #shape=[anchor, 4, 2]
    quad_boxes = tf.reshape(quad_boxes,(-1,8)) #shape=[anchor, 8]

    #####################
    #                   #
    # Output Filtering  #
    #                   #
    #####################
    # Score calculation
    cls_preds = tf.nn.sigmoid(cls_pred)
    labels = tf.argmax(cls_preds, axis=1, name="decode_argmax")
    score = tf.reduce_max(cls_preds, axis=1)

    # Score threshold filtering and get indice
    ids = tf.where(score > config.CLS_THRESH)
    ids = tf.squeeze(ids, axis=1)

    # gather qualified value
    score = tf.gather(score, ids)
    labels = tf.gather(labels, ids)
    rect_boxes = tf.gather(rect_boxes, ids)
    quad_boxes = tf.gather(quad_boxes, ids)

    # qualified label filtering and get indice
    ids = tf.where(tf.equal(labels,1))
    ids = tf.squeeze(ids, axis=1)

    # gather qualified label
    score = tf.gather(score, ids)
    labels = tf.gather(labels, ids)
    rect_boxes = tf.gather(rect_boxes, ids)
    quad_boxes = tf.gather(quad_boxes, ids)

    # Casacade Nox max suppression
    # Step 1: Normal non_max
    # Step 2: Polygon non_max (Cannot process in tf pipline)

    # step 1
    keep_indice = tf.image.non_max_suppression(rect_boxes,
                                               score,
                                               max_output_size=config.NUM_OUTPUT_BOXES,
                                               iou_threshold=config.NMS_THRESH)


    labels = tf.gather(labels, keep_indice)
    score = tf.gather(score, keep_indice)
    rect_boxes = tf.gather(rect_boxes, keep_indice)
    quad_boxes = tf.gather(quad_boxes, keep_indice)

    score = tf.expand_dims(score,1)
    labels = tf.cast(tf.expand_dims(labels,1),tf.float32)

    return tf.concat([quad_boxes,
                      rect_boxes,
                      score,
                      labels,
                     ],axis=-1)

def batch_decode(anchor_list,logists,config):
    """
    ARGS:
        - anchor_list: np.array(num_anchor, 12)
        - logists :
            - loc_pred : Tensor(batch,num_anchor,12)
            - cls_pred : Tensor(batch,num_anchor,2)
    RETURN:

    """
    loc_pred, cls_pred = logists

    decode_data_list = []
     # anchor_list shape normalization
    if np.array(anchor_list).ndim!=2:
        dense_anchor_list = anchor_list[0]
        for i in range(1,len(anchor_list)):
            dense_anchor_list = np.vstack((dense_anchor_list, anchor_list[i]))
        anchor_list=dense_anchor_list

    for i in range(config.BATCH_SIZE):
        decode_data = decode(anchor_list, loc_preds[i], cls_preds[i], config)
        decode_data_list.append(decode_data)

    return decode_data_list

############################################################
#  Mean Average Precision
############################################################
def compute_iou(detect_data, gt_data):
    """


    """

    #################
    #               #
    # Iou Calculate #
    #               #
    #################
    #intersection
    int_xmin = tf.maximum(tf.expand_dims(detect_data[:,0], 1), tf.transpose(tf.expand_dims(gt_data[:,0], 1)))
    int_ymin = tf.maximum(tf.expand_dims(detect_data[:,1], 1), tf.transpose(tf.expand_dims(gt_data[:,1], 1)))
    int_xmax = tf.minimum(tf.expand_dims(detect_data[:,2], 1), tf.transpose(tf.expand_dims(gt_data[:,2], 1)))
    int_ymax = tf.minimum(tf.expand_dims(detect_data[:,3], 1), tf.transpose(tf.expand_dims(gt_data[:,3], 1)))
    w = tf.maximum(int_xmax - int_xmin, 0)
    h = tf.maximum(int_ymax - int_ymin, 0)
    intersection = w*h #shape=[num_anchor,num_gt]
    #area
    areas1 = (detect_data[:,0] - detect_data[:,2])*(detect_data[:,1] - detect_data[:,3])      #shape=[num_anchor]
    areas2 = (gt_data[:,0] - gt_data[:,2])*(gt_data[:,1] - gt_data[:,3]) #shape=[num_gt]
    #union
    unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersection) #shape=[num_anchor,num_gt]

    # iou
    # avoid divide zero
    iou_condition = tf.equal(intersection, 0)              #shape=[num_anchor,num_gt]
    condition_a = tf.zeros_like(intersection,dtype=tf.float32) #shape=[num_anchor,num_gt]
    condition_b = tf.truediv(intersection, unions)           #shape=[num_anchor,num_gt]
    iou  = tf.where(iou_condition, condition_a, condition_b) #shape=[num_anchor,num_gt]
    return iou

def compute_ap(detect_data, gt_bbox, iou_threshold=0.5):
    iou =  compute_iou(detect_data, gt_bbox)
    iou_max_value = tf.reduce_max(iou,axis=1)
    iou_idx = tf.argmax(iou,axis=1)

    # get value greater then threshold
    iou_matrix = tf.where(iou_max_value>iou_threshold, tf.ones_like(iou_max_value),tf.zeros_like(iou_max_value))

    # get unique
    unique_vals, t = tf.unique(iou_idx)
    uni_idx = tf.map_fn(
        lambda x: tf.argmax(tf.cast(tf.equal(iou_idx, x), tf.float32)),
        unique_vals)

    uni_val = tf.cast(tf.gather(iou_matrix, uni_idx),tf.float32)

    remains = tf.subtract(tf.shape(iou_max_value)[0],tf.shape(uni_val)[0])
    padding_zero = tf.cast(tf.zeros(tf.reshape(remains,[1])),tf.float32)
    tp = tf.concat([uni_val,padding_zero],axis=0)
    fp = tf.where(tf.equal(tp,0),tf.ones_like(tp),tf.zeros_like(tp))
    tp_cum = tf.cumsum(tp)
    fp_cum = tf.cumsum(fp)

    #precision = tf.divide(tp_cum,tf.range(1.,tf.cast(iou_max_value.shape.as_list()[0]+1.,tf.float32),1))
    precision = tf.divide(tp_cum,tf.range(1.,tf.cast(tf.add(tf.shape(iou_max_value)[0],1),tf.float32),1))
    recall = tf.divide(tp_cum,tf.fill(tf.shape(iou_max_value),
                                      tf.cast(tf.shape(gt_bbox)[0],tf.float32)))
    #######################################
    #  Mean Average Precision Calculation #
    #######################################
    # padding begin and end for calc purpose

    tf_mpre = tf.concat([tf.zeros(1.), precision, tf.zeros(1.)],axis=0)
    tf_mrec = tf.concat([tf.zeros(1.), recall, tf.ones(1.)],axis=0)


    #following tf code are alternative to the np code below
    """
    for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    """
    tf_mpre_ = tf.concat([ precision, tf.zeros(1.), tf.zeros(1.)],axis=0)
    tf_mpre = tf.where(tf_mpre>tf_mpre_,tf_mpre,tf_mpre_)


    # find internal interval in the recall list
    # step 1: get id of unique value in list
    idxs = tf.reshape(tf.where(tf.not_equal(tf_mrec[1:],tf_mrec[:-1])),[-1])

    # step 2: get internal interval by subtraction
    # e.g.
    # [0.  ,0.14, 0.28, 0.42, 0.57, 1.] recall
    # [0.  ,0.14, 0.28, 0.42, 0.57 ] list_1
    # [0.14,0.28, 0.42, 0.57, 1.   ] list_2
    # [0.14,0.14, 0.14, 0.15, 0.43 ] internal interval


    # area calculation for average precision
    recall_interval = tf.subtract(tf.gather(tf_mrec,tf.add(idxs,1)),
                      tf.gather(tf_mrec,idxs))

    precision_inverval = tf.gather(tf_mpre,tf.add(idxs,1))
    average_precision = tf.reduce_sum(tf.multiply(recall_interval,precision_inverval))

    return average_precision

def compute_ap_range(iou,gt_bbox,iou_thrs):
    ap_dict = {}
    for thr in iou_thrs:
        ap_dict[str(thr)] = compute_ap_td(iou,gt_bbox,thr)
    return ap_dict

