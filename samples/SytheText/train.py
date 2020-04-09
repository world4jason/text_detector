import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

import os
import sys
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# set gpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# find root dir
sys.path.append(ROOT_DIR)

from network.utils.data_utils import *
from network.utils.utils import *
from network.layers import *
from network.config import Config
from network import augment
from network.model import TextBoxesNet



# main
with tf.Graph().as_default():
    config = Config()
    model = TextBoxesNet(config)

    batch_image, batch_loc, batch_cls, batch_gt_boxes = model.data_generator(config.TRAIN_DATASET_PATH, config.BATCH_SIZE, augment=False)
    with tf.variable_scope('{}_{}'.format('train', 0)) as scope:
        logits = model.forward(batch_image)
        loc_preds = logits[:,:,:12] #(batch, num_anchor, 2)
        cls_preds = logits[:,:,12:] #(batch, num_anchor, 12)

        loc_loss, cls_loss, tvars, extra_update_ops = model.calc_loss([loc_preds, cls_preds], [batch_loc, batch_cls])

    #loss merge
    total_loss = loc_loss + cls_loss

    # Add weight decay to the loss.
    l2_loss = config.WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tvars])
    total_loss += l2_loss

    #########################
    # Learning Rate Setting #
    #########################
    # change learning rate at specific step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    values = [config.LEARNING_RATE / pow(10, i) for i in range(3)]
    learning_rate = tf.train.piecewise_constant(global_step, config.LEARNING_DECAY_STEPS, values)
    tf.summary.scalar('learning_rate', learning_rate)

    #################################
    # Opitmize and Gradient Setting #
    #################################
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(total_loss, tvars)
    # clip gradient
    for i,(gradient,var) in enumerate(grads_and_vars):
        if gradient is not None:
            grads_and_vars[i] = (tf.clip_by_norm(gradient,config.CLIP_NORM),var)
    train_op = optimizer.apply_gradients(grads_and_vars)

    ##########################
    # Decode and Compute IoU #
    ##########################
    decode_data_list = model.decode(logits)
    mAP = []
    mIoU = []
    for i in range(config.IMAGES_PER_GPU):
        mAP.append(compute_ap(decode_data_list[i],batch_gt_boxes[i],0.5))
        mIoU.append(tf.reduce_max(compute_iou(decode_data_list[i], batch_gt_boxes[i]),axis=1))
    batch_mAP = tf.reduce_mean(mAP)
    batch_mIoU = tf.reduce_mean(mIoU)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # Train Summary
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge([s for s in summaries if 'valid_' not in s.name])

    # only save global variables
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer(),
                       tf.initialize_all_variables())


    scaffold = tf.train.Scaffold(saver=saver,
                                 init_op=init_op,
                                 summary_op=summary_op)
    valid_saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)

    with tf.train.MonitoredTrainingSession(checkpoint_dir=config.OUTPUT_LOGS,
                                                scaffold=scaffold,
                                                config=session_config,
                                                #save_checkpoint_steps=FLAGS.valid_steps,
                                                save_checkpoint_secs=1800,  #None,
                                                save_summaries_steps=config.SUMMARY_STEPS,
                                                save_summaries_secs=None,
                                                ) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(1, config.NUM_TRAINING_STEPS):
            # Run optimization op (backprop)
            step_loc_loss, step_cls_loss, _ = sess.run([loc_loss, cls_loss, train_op])
            print('STEP : %d\tTRAIN_TOTAL_LOSS : %.8f\tTRAIN_LOC_LOSS : %.8f\tTRAIN_CLS_LOSS : %.5f'
                    % (step, step_loc_loss + step_cls_loss, step_loc_loss, step_cls_loss),end='\r')
            if step%100==0:
                mAP_,mIoU_ = sess.run([batch_mAP,batch_mIoU])
                print('STEP : %d\tmAP 0.5:%.3f\tmIoU:%.3f'%(step,mAP_,mIoU_))

        print("Optimization Finished!")
