import numpy as np
import scipy.io as sio
from random import shuffle

import io
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/..")
from lxml import etree
import PIL.Image
import tensorflow as tf
import tfrecord_utils
ROOT_DIR = os.path.dirname(os.path.abspath(__file__+"/.."))

flags = tf.app.flags
flags.DEFINE_string('data_dir', os.path.join(ROOT_DIR,'train_data/SynthText/train/'), 'Root directory to raw pet dataset, like /startdt_data/HDA_Dataset_V1.3/VOC_fmt_training_fisheye')
flags.DEFINE_string('output_dir', os.path.join(ROOT_DIR,'train_data/SynthText/tfrecord/'), 'Path to directory to output TFRecords, like models/hda_cam_person_fisheye')
flags.DEFINE_integer('image_width', None, 'An option to resize image. Default is None.')
flags.DEFINE_integer('image_height', None, 'An option to resize image. Default is None.')

FLAGS = flags.FLAGS


def dict_to_tf_example(img_path, labels, image_size=None):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset (here only head available) directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()

  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  if image.mode != 'RGB':
    image = image.convert('RGB')

  width, height = image.size
  # quadrilateral coordinate
  q_x0 = []
  q_y0 = []
  q_x1 = []
  q_y1 = []
  q_x2 = []
  q_y2 = []
  q_x3 = []
  q_y3 = []
  q_classes = []

  # normal coordinate
  o_cx = []
  o_cy = []
  o_w = []
  o_h = []

  ########################################################
  #
  # Normalize all value to [0~1]
  # This steps makes resizing purpose much more easily
  #
  ########################################################
  # labels : [2, 4, num_boxes]
  if labels.ndim == 3:
    for i in range(labels.shape[2]):

      q_x0.append(labels[0][0][i] / width)
      q_y0.append(labels[1][0][i] / height)

      q_x1.append(labels[0][1][i] / width)
      q_y1.append(labels[1][1][i] / height)

      q_x2.append(labels[0][2][i] / width)
      q_y2.append(labels[1][2][i] / height)

      q_x3.append(labels[0][3][i] / width)
      q_y3.append(labels[1][3][i] / height)

      q_classes.append(1)

      x_min = min(labels[0][0][i],labels[0][1][i],labels[0][2][i],labels[0][3][i])
      y_min = min(labels[1][0][i],labels[1][1][i],labels[1][2][i],labels[1][3][i])

      x_max = max(labels[0][0][i],labels[0][1][i],labels[0][2][i],labels[0][3][i])
      y_max = max(labels[1][0][i],labels[1][1][i],labels[1][2][i],labels[1][3][i])

      o_cx.append((x_min + x_max) / (width*2))
      o_cy.append((y_min + y_max) / (height*2))
      o_w.append(abs(x_max - x_min) / width)
      o_h.append(abs(y_max - y_min) / height)

  # labels : [2, 4]
  else:
    q_x0.append(labels[0][0] / width )
    q_y0.append(labels[1][0] / height)

    q_x1.append(labels[0][1] / width)
    q_y1.append(labels[1][1] / height)

    q_x2.append(labels[0][2] / width)
    q_y2.append(labels[1][2] / height)

    q_x3.append(labels[0][3] / width )
    q_y3.append(labels[1][3] / height)

    q_classes.append(1)


    x_min = min(labels[0][0],labels[0][1],labels[0][2],labels[0][3])
    y_min = min(labels[1][0],labels[1][1],labels[1][2],labels[1][3])

    x_max = max(labels[0][0],labels[0][1],labels[0][2],labels[0][3])
    y_max = max(labels[1][0],labels[1][1],labels[1][2],labels[1][3])

    o_cx.append((x_min + x_max) / (width*2))
    o_cy.append((y_min + y_max) / (height*2))
    o_w.append(abs(x_max - x_min) / width)
    o_h.append(abs(y_max - y_min) / height)

  if image_size[0] and image_size[1] is not None:
    width, height = image_size
    image.resize((width, height),PIL.Image.ANTIALIAS)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': tfrecord_utils.bytes_feature(encoded_jpg),
      'image/format': tfrecord_utils.bytes_feature('jpg'.encode('utf8')),
      'image/width': tfrecord_utils.int64_feature(width),
      'image/height': tfrecord_utils.int64_feature(height),
      'image/object/bbox/y0': tfrecord_utils.float_list_feature(q_y0),
      'image/object/bbox/x0': tfrecord_utils.float_list_feature(q_x0),
      'image/object/bbox/y1': tfrecord_utils.float_list_feature(q_y1),
      'image/object/bbox/x1': tfrecord_utils.float_list_feature(q_x1),
      'image/object/bbox/y2': tfrecord_utils.float_list_feature(q_y2),
      'image/object/bbox/x2': tfrecord_utils.float_list_feature(q_x2),
      'image/object/bbox/y3': tfrecord_utils.float_list_feature(q_y3),
      'image/object/bbox/x3': tfrecord_utils.float_list_feature(q_x3),
      'image/object/class/label': tfrecord_utils.int64_list_feature(q_classes),
      'image/object/bbox/cy': tfrecord_utils.float_list_feature(o_cy),
      'image/object/bbox/cx': tfrecord_utils.float_list_feature(o_cx),
      'image/object/bbox/w': tfrecord_utils.float_list_feature(o_w),
      'image/object/bbox/h': tfrecord_utils.float_list_feature(o_h),
  }))
  return example


def create_tf_record(output_path, data_dir,image_size):
    """Creates a TFRecord file from examples.
    Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    """
    # get Ground Truth
    gt = sio.loadmat(data_dir + 'gt.mat')
    dataset_size = gt['imnames'].shape[1]
    img_files = gt['imnames'][0]
    labels = gt['wordBB'][0]

    index = list(range(dataset_size))
    shuffle(index)
    size = "raw" if ((image_size[0] or image_size[1]) is None) else str(image_size)
    print("size:",size)
    # Train_tfrecord
    writer_train = tf.python_io.TFRecordWriter(output_path + "train_all_"+size+".record")

    # 850000 for training, 8750 for val
    train_index = index[0:850000]
    train_size = len(train_index)

    print ('{} training examples.', train_size)
    for n, i in enumerate(train_index):
        if n % 100 == 0:
            print ('On image {} of {}'.format(n, train_size), end='\r')

        img_file = data_dir + str(img_files[i][0])
        label = labels[i]

        tf_example = dict_to_tf_example(img_file, label, image_size)
        writer_train.write(tf_example.SerializeToString())

    writer_train.close()

    # Valid_tfrecord
    writer_val = tf.python_io.TFRecordWriter(output_path + "val_all_"+size+".record")

    val_index = index[850000:]
    val_size = len(val_index)

    print ('{} valid examples.', val_size)
    for n, i in enumerate(val_index):
        if n % 100 == 0:
            print ('On image {} of {}'.format(n, val_size), end='\r')

        img_file = data_dir + str(img_files[i][0])
        label = labels[i]

        tf_example = dict_to_tf_example(img_file, label, image_size)
        writer_val.write(tf_example.SerializeToString())

    writer_val.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
    data_dir = FLAGS.data_dir
    print ("Generate data for model !")

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    # random.seed(42)
    # random.shuffle(examples_list)
    # num_examples = len(examples_list)
    # num_train = int(num_examples)
    # train_examples = examples_list[:num_train]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    print(FLAGS.image_width,FLAGS.image_height)
    create_tf_record(FLAGS.output_dir, data_dir, (FLAGS.image_width,FLAGS.image_height))

if __name__ == '__main__':
  tf.app.run()
