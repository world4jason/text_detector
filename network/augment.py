import tensorflow as tf
import random, cv2
import numpy as np

def tf_summary_image(image, boxes, name='image'):
    """Add image with bounding boxes to summary.
    """
    image = tf.expand_dims(image, 0)
    boxes = tf.expand_dims(boxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, boxes)
    tf.summary.image(name, image_with_box)

def normalize_image(image, mean=(0.485, 0.456, 0.406), var=(0.229, 0.224, 0.225)):
    """Normalizes pixel values in the image.
    Moves the pixel values from the current [original_minval, original_maxval]
    range to a the [target_minval, target_maxval] range.
    Args:
    image: rank 3 float32 tensor containing 1
           image -> [height, width, channels].
    Returns:
    image: image which is the same shape as input image.
    """
    with tf.name_scope('NormalizeImage', values=[image]):
        image = tf.to_float(image)
        image /= 255.0

        image -= mean
        image /= var

        return image


def resize_image_and_boxes(image, boxes, input_size,
                 method=tf.image.ResizeMethod.BILINEAR):
    with tf.name_scope('ResizeImage', values=[image, input_size, method]):
        image_resize = tf.image.resize_images(image, [input_size, input_size], method=method)
        boxes_resize = boxes * input_size

        return image_resize, boxes_resize

def random_horizontal_flip(image, boxes, seed=None):
    """Randomly decides whether to horizontally mirror the image and detections or not.
    The probability of flipping the image is 50%.
    Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) float32 tensor containing the bounding boxes and label
               shape=>Tensor([13])
               Boxes are in normalized form meaning their coordinates vary
               between [0, 1].
               Each row is in the form of [x0, y0, x1, y1, x2, y2, x3, y3, cx, cy, w, h].
    seed: random seed
    Returns:
    image: image which is the same shape as input image.
    If boxes, masks, keypoints, and keypoint_flip_permutation is not None,
    the function also returns the following tensors.
    boxes: rank 2 float32 tensor containing the bounding boxes -> [12].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
    """
    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped


    def _flip_boxes_horizontally(boxes):
        """Left-right flip the boxes.
        Args:
        boxes: float32 tensor containing the bounding boxes and label
               shape=>Tensor([13])
               Boxes are in normalized form meaning their coordinates vary
               between [0, 1].
               Each row is in the form of [x0, y0, x1, y1, x2, y2, x3, y3, cx, cy, w, h].
        Returns:
        Horizontally flipped boxes.
        """
        x0, y0, x1, y1, x2, y2, x3, y3, cx, cy, w, h = boxes
        flipped_x0 = tf.subtract(1.0, x1)
        flipped_x1 = tf.subtract(1.0, x0)
        flipped_x2 = tf.subtract(1.0, x3)
        flipped_x3 = tf.subtract(1.0, x2)
        flipped_cx = tf.subtract(1.0, cx)

        return [flipped_x0, y1,
                flipped_x1, y0,
                flipped_x2, y3,
                flipped_x3, y2,
                flipped_cx, cy, w, h]

    with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.random_uniform([], seed=seed)
        # flip only if there are bounding boxes in image!
        do_a_flip_random = tf.logical_and(
            tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, 0.5))

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(
              do_a_flip_random, lambda: _flip_boxes_horizontally(boxes), lambda: boxes)

        return image,boxes


def random_vertical_flip(image, boxes, seed=None):
    """Randomly decides whether to vertically mirror the image and detections or not.
    The probability of flipping the image is 50%.
    Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) float32 tensor containing the bounding boxes and label
               shape=>Tensor([13])
               Boxes are in normalized form meaning their coordinates vary
               between [0, 1].
               Each row is in the form of [x0, y0, x1, y1, x2, y2, x3, y3, cx, cy, w, h].
    seed: random seed
    Returns:
    image: image which is the same shape as input image.
    If boxes, masks, keypoints, and keypoint_flip_permutation is not None,
    the function also returns the following tensors.
    boxes: rank 2 float32 tensor containing the bounding boxes -> [12].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
    """
    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_up_down(image)
        return image_flipped


    def _flip_boxes_vertically(boxes):
        """Up-dowon flip the boxes.
        Args:
        boxes: float32 tensor containing the bounding boxes and label
               shape=>Tensor([13])
               Boxes are in normalized form meaning their coordinates vary
               between [0, 1].
               Each row is in the form of [x0, y0, x1, y1, x2, y2, x3, y3, cx, cy, w, h].
        Returns:
        Horizontally flipped boxes.
        """
        x0, y0, x1, y1, x2, y2, x3, y3, cx, cy, w, h = boxes
        flipped_y0 = tf.subtract(1.0, y1)
        flipped_y1 = tf.subtract(1.0, y0)
        flipped_y2 = tf.subtract(1.0, y3)
        flipped_y3 = tf.subtract(1.0, y2)
        flipped_cy = tf.subtract(1.0, cy)

        return [x1, flipped_y0,
                x0, flipped_y1,
                x3, flipped_y2,
                x2, flipped_y3,
                cx, flipped_cy, w, h]

    with tf.name_scope('RandomVerticalFlip', values=[image, boxes]):
        # random variable defining whether to do flip or not
        do_a_flip_random = tf.random_uniform([], seed=seed)
        # flip only if there are bounding boxes in image!
        do_a_flip_random = tf.logical_and(
            tf.greater(tf.size(boxes), 0), tf.greater(do_a_flip_random, 0.5))

        # flip image
        image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

        # flip boxes
        if boxes is not None:
            boxes = tf.cond(
              do_a_flip_random, lambda: _flip_boxes_vertically(boxes), lambda: boxes)

        return image, boxes

def random_pixel_value_scale(image, minval=0.9, maxval=1.1, seed=None):
    """Scales each value in the pixels of the image.
     This function scales each pixel independent of the other ones.
     For each value in image tensor, draws a random number between
     minval and maxval and multiples the values with them.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    minval: lower ratio of scaling pixel values.
    maxval: upper ratio of scaling pixel values.
    seed: random seed.
    Returns:
    image: image which is the same shape as input image.
    """
    with tf.name_scope('RandomPixelValueScale', values=[image]):
        color_coef = tf.random_uniform(
            tf.shape(image),
            minval=minval,
            maxval=maxval,
            dtype=tf.float32,
            seed=seed)
        image = tf.multiply(image, color_coef)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

def random_image_scale(image,
                       masks=None,
                       min_scale_ratio=0.5,
                       max_scale_ratio=2.0,
                       seed=None):
    """Scales the image size.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels].
    masks: (optional) rank 3 float32 tensor containing masks with
      size [height, width, num_masks]. The value is set to None if there are no
      masks.
    min_scale_ratio: minimum scaling ratio.
    max_scale_ratio: maximum scaling ratio.
    seed: random seed.
    Returns:
    image: image which is the same rank as input image.
    masks: If masks is not none, resized masks which are the same rank as input
      masks will be returned.
    """
    with tf.name_scope('RandomImageScale', values=[image]):
        result = []
        image_shape = tf.shape(image)
        image_height = image_shape[0]
        image_width = image_shape[1]
        size_coef = tf.random_uniform([],
                                      minval=min_scale_ratio,
                                      maxval=max_scale_ratio,
                                      dtype=tf.float32, seed=seed)
        image_newysize = tf.to_int32(
            tf.multiply(tf.to_float(image_height), size_coef))
        image_newxsize = tf.to_int32(
            tf.multiply(tf.to_float(image_width), size_coef))
        image = tf.image.resize_images(
            image, [image_newysize, image_newxsize], align_corners=True)
        result.append(image)
        if masks:
            masks = tf.image.resize_nearest_neighbor(
              masks, [image_newysize, image_newxsize], align_corners=True)
            result.append(masks)
        return tuple(result)


def random_adjust_brightness(image, max_delta=32. / 255.):
    """Randomly adjusts brightness.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: how much to change the brightness. A value between [0, 1).
    Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
    """
    def _random_adjust_brightness(image, max_delta):
        with tf.name_scope('RandomAdjustBrightness', values=[image]):
            image = tf.image.random_brightness(image, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image

    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_brightness(image, max_delta), lambda: image)
    return image

def random_adjust_contrast(image, min_delta=0.5, max_delta=1.25):
    """Randomly adjusts contrast.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
    Returns:
    image: image which is the same shape as input image.
    """
    def _random_adjust_contrast(image, min_delta, max_delta):
        with tf.name_scope('RandomAdjustContrast', values=[image]):
            image = tf.image.random_contrast(image, min_delta, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image

    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_contrast(image, min_delta, max_delta), lambda: image)
    return image

def random_adjust_hue(image, max_delta=0.02):
    """Randomly adjusts hue.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_delta: change hue randomly with a value between 0 and max_delta.
    Returns:
    image: image which is the same shape as input image.
    """
    def _random_adjust_hue(image, max_delta):
        with tf.name_scope('RandomAdjustHue', values=[image]):
            image = tf.image.random_hue(image, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image

    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_hue(image, max_delta), lambda: image)
    return image


def random_adjust_saturation(image, min_delta=0.5, max_delta=1.25):
    """Randomly adjusts saturation.
    Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.
    Returns:
    image: image which is the same shape as input image.
    """
    def _random_adjust_saturation(image, min_delta, max_delta):
        with tf.name_scope('RandomAdjustSaturation', values=[image]):
            image = tf.image.random_saturation(image, min_delta, max_delta)
            image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image

    do_random = tf.greater(tf.random_uniform([]), 0.90)
    image = tf.cond(do_random, lambda: _random_adjust_saturation(image, min_delta, max_delta), lambda: image)
    return image


def random_distort_color(image, color_ordering=0):
    """Randomly distorts color.
    Randomly distorts color using a combination of brightness, hue, contrast
    and saturation changes. Makes sure the output image is still between 0 and 1.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).
    Returns:
    image: image which is the same shape as input image.
    Raises:
    ValueError: if color_ordering is not in {0, 1}.
    """
    with tf.name_scope('RandomDistortColor', values=[image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        else:
            raise ValueError('color_ordering must be in {0, 1}')

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
