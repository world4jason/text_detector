import tensorflow as tf
import numpy as np

def res_block_v1(feature_maps, filters, training, use_bn, strides=1, downsample=False):
    """ResNet v1 block"""
    path_2 = feature_maps

    # conv 1x1
    path_1 = conv_layer(feature_maps, filters[0], kernel_size=1)
    path_1 = norm_layer(path_1, training, use_bn)
    path_1 = relu(path_1)   # activation?

    # conv 3x3
    path_1 = conv_layer(path_1, filters[1], kernel_size=3, strides=strides)
    path_1 = norm_layer(path_1, training, use_bn)
    path_1 = relu(path_1)

    # conv 1x1
    path_1 = conv_layer(path_1, filters[2], kernel_size=1)
    path_1 = norm_layer(path_1, training, use_bn)


    if downsample:
        # shortcut
        path_2 = conv_layer(path_2, filters[2], kernel_size=1, strides=strides)
        path_2 = norm_layer(path_2, training, use_bn)

    top = path_1 + path_2
    top = relu(top)
    return top

def res_block_v2(feature_maps, filters, training, use_bn, strides=1, downsample=False):
    """ResNet v2 block"""
    path_2 = feature_maps

    # conv 1x1
    path_1 = conv_layer(feature_maps, filters[0], kernel_size=1)
    path_1 = norm_layer(path_1, training, use_bn)
    path_1 = relu(path_1)   # activation?

    # conv 3x3
    path_1 = conv_layer(path_1, filters[1], kernel_size=3, strides=strides)
    path_1 = norm_layer(path_1, training, use_bn)
    path_1 = relu(path_1)

    # conv 1x1
    path_1 = conv_layer(path_1, filters[2], kernel_size=1)


    if downsample:
        # shortcut
        path_2 = conv_layer(path_2, filters[2], kernel_size=1, strides=strides)
        path_2 = norm_layer(path_2, training, use_bn)

    return relu(path_1 + path_2)

def conv_layer(feature_maps, filters, kernel_size, name=None,
               strides=1, padding='same', use_bias=False, kernel_initializer=None, he_init_std=None):
    """Build a convolutional layer using entry from layer_params)"""
    if kernel_initializer == 'he':
        if he_init_std is None:
            n = kernel_size * kernel_size * filters
            std = np.sqrt(2.0 / n)
        else:
            std = he_init_std
        kernel_initializer = tf.random_normal_initializer(stddev=std)
    elif kernel_initializer== 'xavier':
        kernel_initializer = tf.contrib.layers.xavier_initializer()
    elif kernel_initializer is None:
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()

    if strides is not 1:
        padding = 'valid'
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        feature_maps = tf.pad(feature_maps, [[0, 0], [pad_beg, pad_end],
                                 [pad_beg, pad_end], [0, 0]])

    bias_initializer = tf.constant_initializer(value=0.0)

    return tf.layers.conv2d(feature_maps,
                           filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           use_bias=use_bias,
                           name=name)


def pool_layer(feature_maps, pool, stride, name=None, padding='same'):
    """Short function to build a pooling layer with less syntax"""
    return tf.layers.max_pooling2d( feature_maps, pool, stride,
                                   padding=padding,
                                   name=name)


def relu(feature_maps, name=None):
    """Relu actication Function"""
    return tf.nn.relu(feature_maps, name=name)

def norm_layer(feature_maps, training, use_bn):
    """Batch Norm or Group Norm"""
    if use_bn:
        return tf.layers.batch_normalization( feature_maps, axis=3, training=training)
    else:
        return tf.contrib.layers.group_norm(feature_maps, groups=32, channels_axis=3)

def group_norm(feature_maps):
    # normalize
    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
    x = tf.transpose(x, [0, 3, 1, 2])
    N, C, H, W = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [-1, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])

    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
    output = tf.transpose(output, [0, 2, 3, 1])

    return output

def lrelu(feature_maps, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(feature_maps)


def selu(feature_maps):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(feature_maps > 0.0, feature_maps, alpha * tf.exp(feature_maps) - alpha)

def upsampling(feature_maps, size, name=None):
    """Bilinear Upsampling"""
    out_shape = tf.shape(feature_maps)[1:3] * tf.constant(size)
    return tf.image.resize_bilinear(feature_maps, out_shape, align_corners=True, name=name)

