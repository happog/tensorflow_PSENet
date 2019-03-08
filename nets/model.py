#-*- coding:utf-8 -*-
import tensorflow as tf
from utils.utils_tool import logger

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets.resnet import resnet_v1

FLAGS = tf.app.flags.FLAGS

def unpool(inputs, rate):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*rate,  tf.shape(inputs)[2]*rate])

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def model(images, outputs = 6, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            #TODO:no need a list
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i], 2)
                else:
                    #F
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
    with tf.variable_scope('Output'):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            activation_fn=None):
            S = slim.conv2d(g[3], outputs, 1)

    up_S = unpool(S, 4)
    seg_S_pred = tf.nn.sigmoid(up_S)

    return seg_S_pred

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls: ground truth
    :param y_pred_cls: predict
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return dice, loss

def loss(y_true_cls, y_pred_cls,
         training_mask):
    g1, g2, g3, g4, g5, g6 = tf.split(value=y_true_cls, num_or_size_splits=6, axis=3)
    s1, s2, s3, s4, s5, s6 = tf.split(value=y_pred_cls, num_or_size_splits=6, axis=3)
    Gn = [g1, g2, g3, g4, g5, g6]
    Sn = [s1, s2, s3, s4, s5, s6]
    _, Lc = dice_coefficient(Gn[5], Sn[5], training_mask=training_mask)
    tf.summary.scalar('Lc_loss', Lc)

    one = tf.ones_like(Sn[5])
    zero = tf.zeros_like(Sn[5])
    W = tf.where(Sn[5] >= 0.5, x=one, y=zero)
    D = 0
    for i in range(5):
        di, _ = dice_coefficient(Gn[i]*W, Sn[i]*W, training_mask=training_mask)
        D += di
    Ls = 1-D/5.
    tf.summary.scalar('Ls_loss', Ls)
    lambda_ = 0.7
    L = lambda_*Lc + (1-lambda_)*Ls
    return L




