import math
import os
import sys
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 

def model(x_train, y_train, learning_rate = 0.01, train_func = tf.train.AdamOptimizer, batch_size = 64, epoch_num = 100, print_cost = True):
    #don't know if need this yet
    tf.set_random_seed(1)
    costs = []
    m, n_h, n_w, n_c = x_train.shape
    n_y = y_train.shape[1]
    
    X, Y = create_placeholders(n_h, n_w, n_c, n_y)
    paradict = para_shape()
    parameters = initialize_parameters(paradict)
    y_hat = forward_propagation(X, parameters)
    cost = compute_cost(y_hat, Y)
    optimizer = optimize_op(cost = cost, learning_rate = learning_rate, train_func = train_func)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        batch_num = int(m / batch_size)

        for epoch in range(epoch_num):
            batch_cost = 0
            batches = random_batches(x_train, y_train, batch_size)

            for batch in batches:
                batch_x, batch_y = batch
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X: batch_x, Y: batch_y})
                batch_cost += temp_cost / batch_num

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch%d: %f\n" % (epoch, batch_cost))

            costs.append(batch_cost)
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration')
        plt.title('Learning Rate = ' + str(learning_rate))
        plt.show()

        
        train_accuracy = calc_accuracy(y_hat,y_train, parameters)
        print('Train Accuracy: %f' % train_accuracy)
        print('Cost: %f' % costs[-1])

        return train_accuracy, parameters