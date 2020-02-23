# Reference: https://github.com/openai/improved-gan/blob/master/imagenet/ops.py
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01))
    return tf.Variable(initial)


def bias_variable(num_filters):
    initial = tf.Variable(tf.zeros(num_filters))
    return tf.Variable(initial)


def conv2d(x, W, strides):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_conv_layer(in_layer,
                      filter_size,
                      num_channels,
                      num_filters,
                      strides=[1, 1, 1, 1],
                      use_batch_norm=True,
                      use_relu=True,
                      use_max_pool=True,
                      use_skip=False,
                      skip_layer=None,
                      layer_name=None,
                      ):
    conv_layer_pool = None

    shape = [filter_size, filter_size, num_channels, num_filters]
    W_conv = weight_variable(shape=shape)
    b_conv = bias_variable(num_filters=num_filters)
    conv = conv2d(in_layer, W_conv, strides=strides)

    if use_batch_norm:
        if layer_name:
            conv = batch_norm(conv + b_conv, is_training='is_training')
        else:
            conv = batch_norm(conv + b_conv, is_training='is_training')

        if use_skip and skip_layer is not None:
            conv = conv + skip_layer

    # Activation
    if use_relu:
        conv = tf.nn.elu(conv)

    if use_max_pool:
        conv = max_pool_2x2(conv)
        conv_layer_pool = conv

    return conv, conv_layer_pool


def create_fully_connected(in_layer, num_inputs, num_outputs, use_relu=True):
    shape = [num_inputs, num_outputs]

    W_fc = weight_variable(shape=shape)
    b_fc = bias_variable(num_filters=num_outputs)
    fc = tf.matmul(in_layer, W_fc) + b_fc

    # Activation
    if use_relu:
        fc = tf.nn.elu(fc)

    return fc


def batch_norm(input, is_training, momentum=0.9, epsilon=1e-5, in_place_update=True, name="batch_norm"):
    # Example batch_norm usage:
    # bn0 = batch_norm(conv1, is_training=is_training, name='bn0')
    if in_place_update:
        return tf.contrib.layers.batch_norm(input,
                                            decay=momentum,
                                            center=True,
                                            scale=True,
                                            epsilon=epsilon,
                                            updates_collections=None,
                                            is_training=is_training,
                                            scope=name)
    else:
        return tf.contrib.layers.batch_norm(input,
                                            decay=momentum,
                                            center=True,
                                            scale=True,
                                            epsilon=epsilon,
                                            is_training=is_training,
                                            scope=name)
