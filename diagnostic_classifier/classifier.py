import tensorflow as tf
import numpy as np


class SingleMLPClassifier:

    def __init__(self, learning_rate, number_hidden_units, number_of_epochs, batch_size, random_base,
                 number_of_classes, dimension, model_name):
        self.learning_rate = learning_rate
        self.number_hidden_units = number_hidden_units
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.random_base = random_base
        self.number_of_classes = number_of_classes
        self.dimension = dimension
        self.model_name = model_name
        self.id = ''

    def multilayer_perceptron(self, x):

        weights = {
            'h1':  tf.get_variable(
                name='att_w_h1_' + self.model_name + self.id,
                shape=[self.dimension, self.number_hidden_units],
                initializer=tf.random_uniform_initializer(-self.random_base, self.random_base)
            ),
            'out': tf.get_variable(
                name='att_w_out_' + self.model_name + self.id,
                shape=[self.number_hidden_units, self.number_of_classes],
                initializer=tf.random_uniform_initializer(-self.random_base, self.random_base)
            )
        }

        biases = {
            'b1': tf.get_variable(
                name='att_b_1_' + self.model_name + self.id,
                shape=[self.number_hidden_units],
                initializer=tf.random_uniform_initializer(-0., 0.)
            ),
            'out': tf.get_variable(
                name='att_b_out_' + self.model_name + self.id,
                shape=[self.number_of_classes],
                initializer=tf.random_uniform_initializer(-0., 0.)
            )
        }
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    def fit(self, train_x, train_y, nlm, file_to_save, id):

        print("train_x ", train_x.shape)

        self.id = id
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, self.dimension], name='x')
        multilayer_perceptron = self.multilayer_perceptron(x)

        tf.add_to_collection('mlp_'+ self.model_name + self.id, multilayer_perceptron)

        y = tf.placeholder(tf.float32, [None, self.number_of_classes])

        loss = -tf.reduce_sum(y * tf.log(multilayer_perceptron))
        acc_num, acc_prob = nlm.config.accuracy_function(y, multilayer_perceptron)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(multilayer_perceptron, 1)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            def get_batch_data(xi, yi, batch_size, is_shuffle=True):
                for index in nlm.internal_data_loader.batch_index(len(yi), batch_size, is_shuffle):
                    feed_dict = {
                        x: xi[index],
                        y: yi[index]
                    }
                    yield feed_dict, len(index)

            for epoch in range(self.number_of_epochs):

                train_acc, train_cnt = 0., 0
                for train, train_num in get_batch_data(train_x, train_y, self.batch_size):

                    _, _train_acc, tr_pred_y, tr_true_y = sess.run([optimizer, acc_num, pred_y, true_y],
                                                                   feed_dict=train)
                    train_acc += _train_acc
                    train_cnt += train_num

                print('all samples={}, correct prediction={}'.format(train_cnt, train_acc))
                train_acc = train_acc / train_cnt
                print('Epoch {}: train acc={:.6f}'.format(epoch, train_acc))
            saver.save(sess, file_to_save)

            sess.close()

    def predict(self, x, file_to_save, id):

        self.id = id
        graph = tf.Graph()

        session = tf.Session(graph=graph)
        with graph.as_default():

            # restore the model
            new_saver = tf.train.import_meta_graph(file_to_save + ".meta")
            new_saver.restore(session, file_to_save)

            activation = tf.get_collection('mlp_'+ self.model_name + self.id)

            new_x = session.graph.get_tensor_by_name('x:0')

            feed_dict = {
                new_x: x
            }

            predictions = session.run(activation, feed_dict=feed_dict)

        return predictions


