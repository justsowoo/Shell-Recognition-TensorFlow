import math
import os
import sys
import numpy as np 
import tensorflow as tf

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(dtype = tf.float32, shape = [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(dtype = tf.float32, shape = [None, n_y])
    return X, Y

#this function needs to change
def para_shape():
    # 7 layers, 4 conv, 2 fc, 1 final result
    conv1 = [11, 11, 3, 96]
    conv2 = [5, 5, 96, 256]
    conv3 = [3, 3, 256, 256]
    conv4 = [3, 3, 256, 256]

    pool1 = [1, 3, 3, 1]
    pool2 = [1, 3, 3, 1]
    pool3 = [1, 3, 3, 1]
    pool4 = [1, 3, 3, 1]
    #this shape need to change
    W1 = 4096
    W2 = 4096
    W3 = 15  #shell species we have

    paradict = {'conv_1': conv1, 
                'conv_2': conv2,
                'conv_3': conv3,
                'conv_4': conv4,
                'pool_1': pool1,
                'pool_2': pool2,
                'pool_3': pool3,
                'pool_4': pool4,
                'W_1': W1,
                'W_2': W2,
                'W_3': W3}

    return paradict

def initialize_parameters(paradict):
    parameters = {}

    for name in paradict:
        if name.split('_')[0] == 'conv':
            value = tf.get_variable(shape = paradict[name], initializer = tf.contrib.layers.xavier_initializer(seed = 0), name = name)
            parameters[name] = value
        elif name.split('_')[0] == 'pool':
            parameters[name] = paradict[name]
        elif name.split('_')[0] == 'W':
            parameters[name] = paradict[name]

    return parameters

#this func needs to change
def forward_propagation(X, parameters):

    conv1 = parameters['conv_1']
    conv2 = parameters['conv_2']
    conv3 = parameters['conv_3']
    conv4 = parameters['conv_4']
    pool1 = parameters['pool_1']
    pool2 = parameters['pool_2']
    pool3 = parameters['pool_3']
    pool4 = parameters['pool_4']
    W1 = parameters['W_1']
    W2 = parameters['W_2']
    W3 = parameters['W_3']
    #layer 1
    Z1 = tf.nn.conv2d(X, filter = conv1, strides = [1, 4, 4, 1], padding = 'VALID')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = pool1, strides = [1, 2, 2, 1], padding = 'SAME')

    #layer 2
    Z2 = tf.nn.conv2d(P1, filter = conv2, strides = [1, 1, 1, 1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = pool2, strides = [1, 2, 2, 1], padding = 'SAME')

    #layer 3
    Z3 = tf.nn.conv2d(P2, filter = conv3, strides = [1, 1, 1, 1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize = pool3, strides = [1, 2, 2, 1], padding = 'SAME')

    #layer 4
    Z4 = tf.nn.conv2d(P3, filter = conv4, strides = [1, 1, 1, 1], padding = 'SAME')
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4, ksize = pool4, strides = [1, 2, 2, 1], padding = 'SAME')
    
    #layer 5
    A5 = tf.contrib.layers.flatten(P4)
    Z5 = tf.contrib.layers.fully_connected(A5, W1, activation_fn = tf.nn.relu)
    #Z5 = tf.contrib.layers.dropout(Z5, keep_prob = 0.8)

    #layer 6
    A6 = tf.contrib.layers.fully_connected(Z5, W2, activation_fn = tf.nn.relu)
    #Z6 = tf.contrib.layers.dropout(A6, keep_prob = 0.8)

    #layer 7
    y_hat = tf.contrib.layers.fully_connected(A6, W3, activation_fn = tf.nn.softmax)

    return y_hat

def compute_cost(y_hat, y):
    print('y_hat', y_hat)
    print('y', y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = y))
    return cost


def optimize_op(cost, learning_rate = 0.01, train_func = tf.train.AdamOptimizer):
    #   default way is AdamOptimizer

    optimizer = train_func(learning_rate).minimize(cost)
    return optimizer

def calc_accuracy(x, y, parameters):
    y_hat = forward_propagation(x, parameters)
    prediction = tf.argmax(y_hat, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return accuracy