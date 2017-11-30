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
    W3 = 30  #shell species we have

    paradict = {'conv1': conv1, 
                'conv2': conv2,
                'conv3': conv3,
                'conv4': conv4,
                'pool1': pool1,
                'pool2': pool2,
                'pool3': pool3,
                'pool4': pool4,
                'W1': W1,
                'W2': W2,
                'W3': W3}

    return paradict

def initialize_parameters(paradict):
    tf.set_random_seed(1)
    parameters = {}

    for name in paradict:
        value = tf.get_variable(shape = paradict[name], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        parameters[name] = value

    return parameters

#this func needs to change
def forward_propagation(X, parameters):
    #get all the parameters
    for name in parameters:
        locals()[name] = parameters[name]

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
    P3 = tf.nn.max_pool(A3, filter = pool3, strides = [1, 2, 2, 1], padding = 'SAME')

    #layer 4
    Z4 = tf.nn.conv2d(P3, filter = conv4, strides = [1, 1, 1, 1], padding = 'SAME')
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4, filter = pool4, strides = [1, 2, 2, 1], padding = 'SAME')
    
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
    cost = tf.nn.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = y))
    return cost

#this function may need to optimize after days
def random_batches(x, y, batch_size):

    m = x.shape[0]
    rand_batch = np.random.randint(0, m, size = batch_size)
    batch_x =  x[(rand_batch), :, :, :]
    batch_y = y[(rand_batch), :]
    batches = (batch_x, batch_y)
    return batches

def optimize_op(cost, learning_rate = 0.01, train_func = tf.train.AdamOptimizer):
    #   default way is AdamOptimizer

    optimizer = train_func(learning_rate).minimize(cost)
    return optimizer

def calc_accracy(x, y, parameters):
    y_hat = forward_propagation(x, parameters)
    prediction = tf.argmax(y_hat, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return accuracy