import math
import os
import sys
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
#x.shape == [m, n_h, n_w, n_c], y.shape == [m, n_y]
def model(x_train, y_train, x_dev, y_dev, learning_rate = 0.01, train_func = tf.train.AdamOptimizer, batch_size = 64, epoch_num = 100, print_cost = True):
    #don't know if need this yet
    tf.set_random_seed(1)
    train_accuracys = []
    dev_accuracys = []
    m, n_h, n_w, n_c = x_train.shape
    n_y = y_train.shape[1]
    
    X, Y = create_placeholders(n_h, n_w, n_c, n_y)
    paradict = para_shape()
    parameters = initialize_parameters(paradict)
    y_hat = forward_propagation(X, parameters)
    cost = compute_cost(y_hat, Y)
    optimizer = optimize_op(cost = cost, learning_rate = learning_rate, train_func = train_func)
    accuracy = calc_accuracy(X, Y, parameters)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        batch_num = int(m / batch_size)

        for epoch in range(epoch_num):
            batch_cost = 0
            batches = random_batches(x_train, y_train, batch_size)

            for batch in batches:
                batch_x, batch_y = batch
                sess.run([optimizer], feed_dict = {X: batch_x, Y: batch_y})
                #batch_cost += temp_cost / batch_num

            train_accuracy = sess.run(accuracy, feed_dict = {X: x_train, Y: y_train})
            dev_accuracy = sess.run(accuracy, feed_dict = {X: x_dev, Y: y_dev})
            train_accuracys.append(train_accuracy)
            dev_accuracys.append(dev_accuracy)

        fig, ax = plt.subplots()

        xticks = range(0, 110, 10)
        ax.set_xticks(xticks)
        ax.set_xlim([0, 110])
        
        plt.plot(np.squeeze(train_accuracys), 'x-', label = 'Train_Accuracy')
        plt.plot(np.squeeze(dev_accuracys), '+-', label = 'Dev_Accuracy')
        
        plt.ylabel('Accuracy')
        plt.xlabel('iteration(per ten epochs)')
        plt.legend(loc = 'upper right', fontsize = 'large')
        plt.title('Learning Rate = ' + str(learning_rate))
        
        plt.show()

        train_cost = sess.run(cost, feed_dict = {X: x_train, Y: y_train})
        dev_cost = sess.run(cost, feed_dict = {X: x_dev, Y: y_dev})
        print('Train Accuracy: %f' % train_accuracy)
        print('Train Cost: %f' % train_cost)
        print('Dev Accuracy: %f' % dev_accuracy)
        print('Dev Cost: %f' % dev_cost)

        return train_accuracys, dev_accuracys, parameters