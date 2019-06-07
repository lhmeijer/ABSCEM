import tensorflow as tf
import numpy as np


class SingleMLPClassifier:

    def __init__(self, learning_rate, hidden_layers, number_of_epochs, keep_prob, batch_size):
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.number_of_epochs = number_of_epochs
        self.keep_prob = keep_prob
        self.batch_size = batch_size

    @staticmethod
    def multilayer_perceptron(x, weights, biases, keep_prob):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    def fit(self, train_x, train_y, test_x, test_y):

        n_input = train_x.shape[1]
        n_classes = train_y.shape[1]

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, self.hidden_layers])),
            'out': tf.Variable(tf.random_normal([self.hidden_layers, n_classes]))
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([self.hidden_layers])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        keep_prob = tf.placeholder("float")

        display_step = 1000

        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        predictions = self.multilayer_perceptron(x, weights, biases, keep_prob)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.number_of_epochs):
                avg_cost = 0.0
                total_batch = int(len(train_x) / self.batch_size)
                x_batches = np.array_split(train_x, total_batch)
                y_batches = np.array_split(train_y, total_batch)
                for i in range(total_batch):
                    batch_x, batch_y = x_batches[i], y_batches[i]
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={
                                        x: batch_x,
                                        y: batch_y,
                                        keep_prob: self.keep_prob
                                    })
                    avg_cost += c / total_batch
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print("Optimization Finished!")
            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: test_x, y: test_y, keep_prob: 1.0}))
