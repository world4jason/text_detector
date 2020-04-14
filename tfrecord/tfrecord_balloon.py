import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage import io
from io import BytesIO

ROOT_DIR = os.path.dirname(os.path.abspath("."))
import PIL.Image
import tensorflow as tf
import tfrecord_utils

flags = tf.app.flags
flags.DEFINE_string('data_dir', os.path.join(ROOT_DIR,'train_data/balloon/'), 'Root directory to raw pet dataset, like /startdt_data/HDA_Dataset_V1.3/VOC_fmt_training_fisheye')
flags.DEFINE_string('output_dir', os.path.join(ROOT_DIR,'train_data/balloon/tfrecord/'), 'Path to directory to output TFRecords, like models/hda_cam_person_fisheye')
flags.DEFINE_integer('image_width', None, 'An option to resize image. Default is None.')
flags.DEFINE_integer('image_height', None, 'An option to resize image. Default is None.')

FLAGS = flags.FLAGS


def extract_bboxes(mask, image_width, image_height):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (x0,y0,x1,y1,x2,y2,x3,y3,cx,cy,w,h)].
    """
    boxes = np.zeros([np.shape(mask)[-1], 12], dtype=np.int32)
    for i in range(np.shape(mask)[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            xmin, xmax = horizontal_indicies[[0, -1]]
            ymin, ymax = vertical_indicies[[0, -1]]
            # xmax and ymax should not be part of the box. Increment by 1.
            xmax += 1
            ymax += 1

            # format
            x0 = xmin / image_width
            x1 = xmax / image_width
            x2 = xmax / image_width
            x3 = xmin / image_width
            y0 = ymin / image_height
            y1 = ymin / image_height
            y2 = ymax / image_height
            y3 = ymax / image_height
            cx = (xmin+xmax)/(2*image_width)
            cy = (ymin+ymax)/(2*image_height)
            w = abs(xmin-xmax)/image_width
            h = abs(ymin-ymax)/image_height

        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x0,y0,x1,y1,x2,y2,x3,y3,cx,cy,w,h = 0,0,0,0,0,0,0,0,0,0,0,0
        boxes[i] = np.array([x0,y0,x1,y1,x2,y2,x3,y3,cx,cy,w,h])
    return np.transpose(boxes.astype(np.int32))

def dict_to_tf_example(dataset_dir, labels, image_size=None):
    polygons = []
    image_path = os.path.join(dataset_dir, labels['filename'])
    image = skimage.io.imread(image_path)
    height,width,channel = image.shape

    polygon =  [r['shape_attributes'] for r in labels['regions'].values()]

    mask = np.zeros([height, width, len(polygon)],
                    dtype=np.uint8)
    for i, p in enumerate(polygon):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc, i] = 1

    mask = mask.astype(np.bool)
    polygons.append(mask)

    x0,y0,x1,y1,x2,y2,x3,y3,cx,cy,w,h = extract_bboxes(mask, width, height)

    print("np.shape(boxes)",np.shape(x0))
    classes = np.ones(np.shape(x0)).astype(np.int32)

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if image_size[0] and image_size[1] is not None:
        width, height = image_size
    image.resize((width, height),PIL.Image.ANTIALIAS)

    example = tf.train.Example(features=tf.train.Features(feature={
          'image/encoded': tfrecord_utils.bytes_feature(encoded_jpg),
          'image/format': tfrecord_utils.bytes_feature('jpg'.encode('utf8')),
          'image/width': tfrecord_utils.int64_feature(width),
          'image/height': tfrecord_utils.int64_feature(height),
          'image/object/bbox/y0': tfrecord_utils.float_list_feature(y0),
          'image/object/bbox/x0': tfrecord_utils.float_list_feature(x0),
          'image/object/bbox/y1': tfrecord_utils.float_list_feature(y1),
          'image/object/bbox/x1': tfrecord_utils.float_list_feature(x1),
          'image/object/bbox/y2': tfrecord_utils.float_list_feature(y2),
          'image/object/bbox/x2': tfrecord_utils.float_list_feature(x2),
          'image/object/bbox/y3': tfrecord_utils.float_list_feature(y3),
          'image/object/bbox/x3': tfrecord_utils.float_list_feature(x3),
          'image/object/class/label': tfrecord_utils.int64_list_feature(classes),
          'image/object/bbox/cy': tfrecord_utils.float_list_feature(cy),
          'image/object/bbox/cx': tfrecord_utils.float_list_feature(cx),
          'image/object/bbox/w': tfrecord_utils.float_list_feature(w),
          'image/object/bbox/h': tfrecord_utils.float_list_feature(h),
      }))
    return example

def create_tf_record(output_path, data_dir, image_size=(None,None)):
    """Creates a TFRecord file from examples.
    Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
    """

    size = "raw" if image_size[0] or image_size[1] is None else str(image_size)

    # Train_tfrecord
    writer_train = tf.python_io.TFRecordWriter(output_path + "train_balloon_"+size+".record")

    train_dataset_dir = os.path.join(data_dir, "train")

    annotations = json.load(open(os.path.join(train_dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys
    train_annotations = [a for a in annotations if a['regions']]

    train_size = len(train_annotations)

    print ('{} training examples.', train_size)
    for index, annotation in enumerate(train_annotations):
        if index % 100 == 0:
            print ('On image {} of {}'.format(index, train_size), end='\r')
        tf_example = dict_to_tf_example(train_dataset_dir,annotation,image_size)
        if tf_example is not None:
            writer_train.write(tf_example.SerializeToString())

    writer_train.close()

    # Valid_tfrecord
    writer_val = tf.python_io.TFRecordWriter(output_path + "val_balloon_"+size+".record")

    val_dataset_dir = os.path.join(data_dir, "val")

    annotations = json.load(open(os.path.join(val_dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys
    val_annotations = [a for a in annotations if a['regions']]

    val_size = len(val_annotations)

    print ('{} valid examples.', val_size)
    for index, annotation in enumerate(val_annotations):
        if index % 100 == 0:
            print ('On image {} of {}'.format(index, train_size), end='\r')
        tf_example = dict_to_tf_example(val_dataset_dir,annotation,image_size)
        if tf_example is not None:
            writer_train.write(tf_example.SerializeToString())

    writer_val.close()



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
