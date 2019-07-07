import tensorflow as tf
import numpy as np


class LinearModel:

    def __init__(self, learning_rate, number_of_epochs, batch_size, nlm):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.nlm = nlm

    def fit(self, x_train, y_train, weights):

        # clear out old graph
        tf.reset_default_graph()

        # Create graph
        sess = tf.Session()

        m = x_train.shape[0]
        n = x_train.shape[1]

        X = tf.placeholder(tf.float32, [None, n],  name="x")
        Y = tf.placeholder(tf.float32, [None, 1], name="y")
        K = tf.placeholder(tf.float32, [None, 1], name="k")

        # weights
        W = tf.Variable(tf.zeros([n, 1], dtype=np.float32), name="weight")
        b = tf.Variable(tf.zeros([1], dtype=np.float32), name="bias")

        # linear model
        activation = tf.add(tf.matmul(X, W), b)

        cost = tf.reduce_sum(tf.multiply(K, tf.square(activation - Y)))

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        sess.run(init)

        def get_batch_data(xi, yi, wi, batch_size, is_shuffle=True):
            for index in self.nlm.internal_data_loader.batch_index(len(yi), batch_size, is_shuffle):
                feed_dict = {
                    X: xi[index],
                    Y: yi[index],
                    K: wi[index]
                }
                yield feed_dict, len(index)

        for epoch in range(self.number_of_epochs):

            for train, train_num in get_batch_data(x_train, y_train, weights, self.batch_size):
                sess.run(optimizer, feed_dict=train)
                temp_loss = sess.run(cost, feed_dict=train)
                if (epoch + 1) % 10 == 0:
                    print('Loss = ' + str(temp_loss))

        # Get the optimal coefficients
        coefficients = sess.run(W)
        y_intercept = sess.run(b)

        return coefficients, y_intercept
