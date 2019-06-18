import tensorflow as tf
import numpy as np


class SingleMLPClassifier:

    def __init__(self, learning_rate, number_hidden_units, number_of_epochs, keep_prob, batch_size, random_base,
                 number_of_classes, dimension):
        self.learning_rate = learning_rate
        self.number_hidden_units = number_hidden_units
        self.number_of_epochs = number_of_epochs
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.random_base = random_base
        self.number_of_classes = number_of_classes
        self.dimension = dimension

        self.keep_prob = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, dimension])
        self.predictions = self.multilayer_perceptron(self.x, self.keep_prob, dimension, number_hidden_units,
                                                      random_base, number_of_classes)

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)

    def multilayer_perceptron(self, x, keep_prob, n_input, number_hidden_units, random_base, n_classes):

        weights = {
            'h1':  tf.get_variable(
                name='att_w_h1_',
                shape=[n_input, number_hidden_units],
                initializer=tf.random_uniform_initializer(-self.random_base, self.random_base)
            ),
            'out': tf.get_variable(
                name='att_w_out_',
                shape=[number_hidden_units, n_classes],
                initializer=tf.random_uniform_initializer(-self.random_base, self.random_base)
            )
        }

        biases = {
            'b1': tf.get_variable(
                name='att_b_1_',
                shape=[number_hidden_units],
                initializer=tf.random_uniform_initializer(-0., 0.)
            ),
            'out': tf.get_variable(
                name='att_b_out_',
                shape=[n_classes],
                initializer=tf.random_uniform_initializer(-0., 0.)
            )
        }

        x_drop_out = tf.nn.dropout(x, rate=(1-keep_prob))
        layer_1 = tf.add(tf.matmul(x_drop_out, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    def fit(self, train_x, train_y, test_x, test_y, nlm, file_to_save):

        tf.reset_default_graph()

        y = tf.placeholder(tf.float32, [None, self.number_of_classes])

        dimension = train_x.shape[1]

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predictions, labels=y))
        acc_num, acc_prob = nlm.config.accuracy_function(y, self.predictions)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(self.predictions, 1)

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            def get_batch_data(xi, yi, batch_size, kp, n_in, n_hidden, r_base, n_class, is_shuffle=True):
                for index in nlm.internal_data_loader.batch_index(len(yi), batch_size, is_shuffle):
                    feed_dict = {
                        self.x: xi[index],
                        y: yi[index],
                        self.keep_prob: kp,
                    }
                    yield feed_dict, len(index)

            max_test_acc = 0.
            max_train_acc = 0.

            max_tr_pred_n_per_class = np.zeros(self.number_of_classes, dtype=int)
            max_te_pred_n_per_class = np.zeros(self.number_of_classes, dtype=int)

            max_tr_true_n_per_class = np.zeros(self.number_of_classes, dtype=int)
            max_te_true_n_per_class = np.zeros(self.number_of_classes, dtype=int)

            for epoch in range(self.number_of_epochs):

                train_acc, train_cnt = 0., 0
                tr_pred_n_per_class = np.zeros(self.number_of_classes, dtype=int)
                tr_true_n_per_class = np.zeros(self.number_of_classes, dtype=int)

                for train, train_num in get_batch_data(train_x, train_y, self.batch_size, self.keep_prob,
                                                       self.dimension, self.number_hidden_units, self.random_base,
                                                       self.number_of_classes):
                    _, _train_acc, tr_pred_y, tr_true_y = sess.run([optimizer, acc_num, pred_y, true_y],
                                                                   feed_dict=train)
                    train_acc += _train_acc
                    train_cnt += train_num
                    for i in range(tr_pred_y.shape[0]):
                        tr_true_n_per_class[tr_true_y[i]] += 1
                        if tr_pred_y[i] == tr_true_y[i]:
                            tr_pred_n_per_class[tr_true_y[i]] += 1

                test_acc, test_cost, test_cnt = 0., 0., 0
                te_pred_n_per_class = np.zeros(self.number_of_classes, dtype=int)
                te_true_n_per_class = np.zeros(self.number_of_classes, dtype=int)

                for test, test_num in get_batch_data(test_x, test_y, 2000, 1.0, self.dimension, self.number_hidden_units,
                                                     self.random_base, self.number_of_classes, False):
                    _loss, _test_acc, te_true_y, te_pred_y, _prob = sess.run([loss, acc_num, true_y, pred_y,
                                                                              self.predictions], feed_dict=test)
                    test_acc += _test_acc
                    test_cost += _loss * test_num
                    test_cnt += test_num
                    for i in range(te_pred_y.shape[0]):
                        te_true_n_per_class[te_true_y[i]] += 1
                        if te_pred_y[i] == te_true_y[i]:
                            te_pred_n_per_class[te_true_y[i]] += 1

                print('all samples={}, correct prediction={}'.format(test_cnt, test_acc))
                train_acc = train_acc / train_cnt
                test_acc = test_acc / test_cnt
                test_cost = test_cost / test_cnt
                print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}'.format(epoch, test_cost,
                                                                                                  train_acc, test_acc))

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    max_train_acc = train_acc
                    max_tr_pred_n_per_class = tr_pred_n_per_class
                    max_te_pred_n_per_class = te_pred_n_per_class
                    max_tr_true_n_per_class = tr_true_n_per_class
                    max_te_true_n_per_class = te_true_n_per_class

            self.saver.save(sess, file_to_save)

            acc_tr_n_per_class = max_tr_pred_n_per_class / max_tr_true_n_per_class
            acc_te_n_per_class = max_te_pred_n_per_class / max_te_true_n_per_class

            return [max_train_acc, max_test_acc, acc_tr_n_per_class, acc_te_n_per_class]

    def predict(self, x, file_to_save):

        feed_dict = {
            self.x: x,
            self.keep_prob: 1.0,
        }

        with tf.Session() as session:
            # restore the model
            self.saver.restore(session, file_to_save)
            predictions = session.run([self.predictions], feed_dict=feed_dict)
        return predictions
