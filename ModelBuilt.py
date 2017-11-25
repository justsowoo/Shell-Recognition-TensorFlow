import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import math
import sys


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(dtype = tf.float32, shape = [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(dtype = tf.float32, shape = [None, n_y])
    return X, Y

#this function need to change
def para_shape():
    # 8 layers, 5 conv, 2 fc, 1 final result
    conv1 = [12, 12, 3, 16]
    conv2 = [8, 8, 16, 32]
    conv3 = [4, 4, 32, 64]
    conv4 = [3, 3, 64, 128]
    conv5 = [3, 3, 128, 256]
    #this shape need to change
    fc1 = [4096, 1111]
    fc2 = [4096, 4096]
    fc3 = [30, 4096]
    paradict = {'1': conv1, 
                '2': conv2,
                '3': conv3,
                '4': conv4,
                '5': conv5,
                '6': fc1,
                '7', fc2,
                '8', fc3}
    return paradict

def initialize_parameters(paradict):
    tf.set_random_seed(1)
    parameters = {}
    layer = 8

    for l in layer:
        shape = paradict[str(l+1)]
        var = tf.get_variable(shape = shape, initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        parameters[str(l+1)] = var

    return parameters

def AlexNet_layer(X, filter = None, layer, mode = 'conv'):

    if mode == 'conv' and layer <= 3:
        Z = tf.nn.conv2d(X, filter, strides = [1, 1, 1, 1], padding = 'VALID')
        A = tf.nn.relu(Z)
        P = tf.nn.max_pool(A, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
    elif mode = 'conv' and layer <= 5:
        Z = tf.nn.conv2d(X, filter, strides = [1, 1, 1, 1], padding = 'SAME')
        A = tf.nn.relu(Z)
        P = tf.nn.max_pool(A, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    elif mode = 'fc' and layer <= 7:
        Z = tf.contrib.layers.flatten(X)
        A = tf.contrib.layers.fully_connected(Z, 4096, activation_fn = tf.nn.relu)
        P = A
    else :
        Z = tf.contrib.layers.fully_connected(X, 30, activation_fn = tf.nn.softmax)
        P = Z

    return  P

def forward_propagation(X, parameters):
    
    #get all the parameters
    conv1 = parameters['conv1']
    conv2 = parameters['conv2']
    conv3 = parameters['conv3']
    conv4 = parameters['conv4']
    conv5 = parameters['conv5']
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    #layer 1
    P1 = AlexNet_layer(X, filter = conv1, layer = 1, mode = 'conv')

    #layer 2
    P2 = AlexNet_layer(P1, filter = conv2, layer = 2, mode = 'conv')

    #layer 3
    P3 = AlexNet_layer(P2, filter = conv3, layer =3 , mode = 'conv')

    #layer 4
    P4 = AlexNet_layer(P3, filter = conv4, layer = 4, mode = 'conv')

    #layer 5
    P5 = AlexNet_layer(P4, filter = conv5, layer = 5, mode = 'conv')

    #layer 6
    A6 = AlexNet_layer(P5, filter = None, layer = 6, mode = 'fc')

    #layer 7
    A7 = AlexNet_layer(A6, filter = None, layer = 7, mode = 'fc')

    #layer 8 output  30 need to change
    y_hat = AlexNet_layer(A7, filter = None, layer = 8, mode = 'fc')

    return y_hat

def compute_cost(y_hat, y):
    cost = tf.nn.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = y))
    return cost


def random_batches(x, y, batch_size, seed):
    """
    ..............
    """
    return batches