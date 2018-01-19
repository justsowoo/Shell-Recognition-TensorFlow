import argparse
import math
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 

import Image_Preprocessing
import ModelBuilt

#x.shape == [m, n_h, n_w, n_c], y.shape == [m, n_y]
def model(x_train, y_train, x_dev, y_dev, learning_rate = 0.01, train_func = tf.train.AdamOptimizer, print_cost = True):

    train_accuracys = []
    dev_accuracys = []

    paradict = ModelBuilt.para_shape()

    parameters = ModelBuilt.initialize_parameters(paradict)

    y_hat = ModelBuilt.forward_propagation(x_train, parameters)

    cost = ModelBuilt.compute_cost(y_hat, y_train)
    
    optimizer = ModelBuilt.optimize_op(cost = cost, learning_rate = learning_rate, train_func = train_func)

    train_accuracy = ModelBuilt.calc_accuracy(x_train, y_train, parameters)
    
    dev_accuracy = ModelBuilt.calc_accuracy(x_dev, y_dev, parameters)
    
    init = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()

                _, train_cost = sess.run([optimizer, cost])
                train_value, dev_value = sess.run([train_accuracy, dev_accuracy])
                duration = time.time() - start_time
                
        # Print an overview fairly often.
                if step % 10 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, train_cost,
                                                     duration))
                    train_accuracys.append(train_value)
                    dev_accuracys.append(dev_value)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (100, step))
        finally:
      # When done, ask the threads to stop.
            coord.request_stop()

    # Wait for threads to finish.
        coord.join(threads)
        sess.close()

        return train_accuracys, dev_accuracys, parameters

def main():
    train_images, train_labels = Image_Preprocessing.read_and_decode(filename = 'train.tfrecords')
    dev_images, dev_labels = Image_Preprocessing.read_and_decode(filename = 'dev.tfrecords')
    print('train_images', train_images)
    print('train_labels', train_labels)
    train_accuracys, dev_accuracys, parameters = model(train_images, train_labels, dev_images, dev_labels)
    
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

        #train_cost = sess.run(cost, feed_dict = {X: x_train, Y: y_train})
        #dev_cost = sess.run(cost, feed_dict = {X: x_dev, Y: y_dev})
        #dev_cost = sess.run()
    print('Train Accuracy: %f' % train_accuracys[-1])
    #print('Train Cost: %f' % train_cost)
    print('Dev Accuracy: %f' % dev_accuracys[-1])
        #print('Dev Cost: %f' % dev_cost)


if __name__ == '__main__':
    main()

