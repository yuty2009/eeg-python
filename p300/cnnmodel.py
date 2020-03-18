# -*- coding: utf-8 -*-

import tensorflow as tf

INPUT_TENSOR_IMAGE = 'input/X:0'
INPUT_TENSOR_LABEL = 'input/y:0'
INPUT_TENSOR_LR = 'input/lr:0'
INPUT_TENSOR_DROPOUT = 'input/dropout:0'
OUTPUT_TENSOR_LOGIT = 'output/y:0'
BOTTLENECK_TENSOR = 'dense1/bottleneck:0'
INPUT_SHAPE = [78, 64]
MAP_NUM_LAYER1 = 10
MAP_NUM_LAYER2 = 50
MAP_NUM_LAYER3 = 100
BOTTLENECK_SIZE = 1024

def weight_bias(weight_shape, bias_shape):
    # Create variable named 'weights'.
    weights = tf.get_variable('weights', weight_shape,
        initializer=tf.random_normal_initializer(stddev=0.1))
    # Create variable named 'biases'.
    biases = tf.get_variable('biases', bias_shape,
        initializer=tf.constant_initializer(0.1))
    return weights, biases

def conv2d(input, kernel_shape, bias_shape, strides=None):
    weights, biases = weight_bias(kernel_shape, bias_shape)
    if strides == None:
        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='VALID')
    else:
        conv = tf.nn.conv2d(input, weights, strides=strides, padding='VALID')
    return conv + biases

def maxpool(input, kernel_shape, stride_shape):
    return tf.nn.max_pool(input, ksize=kernel_shape, strides=stride_shape, padding='SAME')

def fullconnect(input, weight_shape, bias_shape):
    weights, biases = weight_bias(weight_shape, bias_shape)
    input = tf.reshape(input, [-1, weight_shape[0]])
    return tf.matmul(input, weights) + biases

def feedforward(X, inshape, dropout=0.5):
    # Convolutional layer 1
    with tf.variable_scope('conv1'):
        h_conv1 = tf.nn.relu(conv2d(X, [1, 64, 1, MAP_NUM_LAYER1], [MAP_NUM_LAYER1]))
    # Convolutional layer 2
    with tf.variable_scope('conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_conv1, [13, 1, MAP_NUM_LAYER1, MAP_NUM_LAYER2], [MAP_NUM_LAYER2], [1, 13, 1, 1]))
    # Full connected layer
    with tf.variable_scope('dense1'):
        h_fc1 = tf.nn.relu(fullconnect(h_conv2, [6 * MAP_NUM_LAYER2, BOTTLENECK_SIZE], [BOTTLENECK_SIZE]))
        h_fc1 = tf.identity(h_fc1, name="bottleneck")
    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)
    # Readout layer
    with tf.variable_scope('output'):
        y = fullconnect(h_fc1_drop, [BOTTLENECK_SIZE, 1], [1])
        y = tf.identity(y, name='y')
    return y

def createmodel(inshape):
    with tf.variable_scope('input'):
        X = tf.placeholder(tf.float32, [None, INPUT_SHAPE[0], INPUT_SHAPE[1], 1], name='X')
        ytrue = tf.placeholder(tf.float32, [None, 1], name='y')
        # learning rate
        lr = tf.placeholder(tf.float32, name='lr')
        # dropout parameter
        dropout = tf.placeholder(tf.float32, name='dropout')
    ypred = feedforward(X, inshape, dropout)
    return ypred, X, ytrue, lr, dropout

def loadmodel(modelpath):
    saver = tf.train.import_meta_graph(modelpath)
    X = tf.get_default_graph().get_tensor_by_name(INPUT_TENSOR_IMAGE)
    ytrue = tf.get_default_graph().get_tensor_by_name(INPUT_TENSOR_LABEL)
    lr = tf.get_default_graph().get_tensor_by_name(INPUT_TENSOR_LR)
    dropout = tf.get_default_graph().get_tensor_by_name(INPUT_TENSOR_DROPOUT)
    ypred = tf.get_default_graph().get_tensor_by_name(OUTPUT_TENSOR_LOGIT)
    return ypred, X, ytrue, lr, dropout


