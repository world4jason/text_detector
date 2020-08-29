import tensorflow as tf

############################################################
#  Data Shuffle
############################################################


def _shuffle(inputs, capacity=2000, min_after_dequeue=40, num_threads=16):
    """
    This function provide a manual shuffle method for tf.train.batch
    assert inputs have same data type
    """
    if isinstance(inputs, dict):
        names, dtypes = zip(*[(key, input_.dtype)
                              for key, input_ in inputs.items()])
    else:
        dtypes = [input_.dtype for input_ in inputs]

    queue = tf.RandomShuffleQueue(
        capacity,
        min_after_dequeue,
        dtypes,
        **({'names': names} if isinstance(inputs, dict) else {}))

    tf.train.add_queue_runner(tf.train.QueueRunner(
        queue,
        [queue.enqueue(inputs)] * num_threads))

    shuffled_inputs = queue.dequeue()

    for key, input_ in (inputs.items()
                        if isinstance(inputs, dict) else
                        enumerate(inputs)):
        shuffled_inputs[key].set_shape(input_.get_shape())

    return shuffled_inputs

############################################################
#  TFrecord Reader
############################################################


def read_tfrecord(path, config, shuffle=True):
    """
    A Tfrecord decoder follows the following flow:
    TFRecord > Example > Features > Feature > Raw Data

    ARGS:


    RETURN:
    """

    # Tfrecord reader preparation
    filename_queue = tf.train.string_input_producer([path], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Example Parser
    features = tf.parse_single_example(serialized_example, {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/y0': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x0': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x1': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x2': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/y3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/x3': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/cx': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/cy': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/w': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/h': tf.VarLenFeature(dtype=tf.float32),
    })

    # Parse feature and cast
    decode_image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    decode_image = tf.cast(tf.image.resize_images(
        decode_image, config.IMAGE_SHAPE), tf.int32)

    y0_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/y0']), tf.float32)
    x0_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/x0']), tf.float32)
    y1_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/y1']), tf.float32)
    x1_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/x1']), tf.float32)
    y2_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/y2']), tf.float32)
    x2_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/x2']), tf.float32)
    y3_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/y3']), tf.float32)
    x3_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/x3']), tf.float32)
    cx_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/cx']), tf.float32)
    cy_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/cy']), tf.float32)
    w_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/w']), tf.float32)
    h_set = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/bbox/h']), tf.float32)
    label = tf.cast(tf.sparse_tensor_to_dense(
        features['image/object/class/label']), tf.float32)

    return_value = [decode_image, x0_set, y0_set, x1_set, y1_set, x2_set,
                    y2_set, x3_set, y3_set, cx_set, cy_set, w_set, h_set, label]
    if shuffle:
        return_value = _shuffle(return_value,
                                capacity=config.CAPACITY,
                                min_after_dequeue=config.MIN_AFTER_DEQUEUE,
                                num_threads=config.NUM_THREADS)
    return return_value
