import tensorflow as tf
import numpy as np


class SingleMLPClassifier:

    def __init__(self, learning_rate, number_hidden_units, number_of_epochs, keep_prob, batch_size, random_base,
                 number_of_classes, dimension, model_name):
        self.learning_rate = learning_rate
        self.number_hidden_units = number_hidden_units
        self.number_of_epochs = number_of_epochs
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.random_base = random_base
        self.number_of_classes = number_of_classes
        self.dimension = dimension
        self.model_name = model_name
        self.keep_prob = keep_prob
        self.id = ''

    def multilayer_perceptron(self, x, keep_prob, n_input, number_hidden_units, random_base, n_classes):

        weights = {
            'h1':  tf.get_variable(
                name='att_w_h1_' + self.model_name + self.id,
                shape=[n_input, number_hidden_units],
                initializer=tf.random_uniform_initializer(-self.random_base, self.random_base)
            ),
            'out': tf.get_variable(
                name='att_w_out_' + self.model_name + self.id,
                shape=[number_hidden_units, n_classes],
                initializer=tf.random_uniform_initializer(-self.random_base, self.random_base)
            )
        }

        biases = {
            'b1': tf.get_variable(
                name='att_b_1_' + self.model_name + self.id,
                shape=[number_hidden_units],
                initializer=tf.random_uniform_initializer(-0., 0.)
            ),
            'out': tf.get_variable(
                name='att_b_out_' + self.model_name + self.id,
                shape=[n_classes],
                initializer=tf.random_uniform_initializer(-0., 0.)
            )
        }
        #
        # x_drop_out = tf.nn.dropout(x, rate=(1-keep_prob))
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    def fit(self, train_x, train_y, nlm, file_to_save, id):

        self.id = id

        tf_keep_prob = tf.placeholder(tf.float32, name='tf_keep_prob')
        x = tf.placeholder(tf.float32, [None, self.dimension], name='x')
        multilayer_perceptron = self.multilayer_perceptron(x, tf_keep_prob, self.dimension, self.number_hidden_units,
                                                           self.random_base, self.number_of_classes)

        tf.add_to_collection("mlp_activation_" + self.model_name + self.id, multilayer_perceptron)

        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        print("all_variables ", all_variables)
        # list_of_var = [v for v in all_variables if model_name in v.name]
        # self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000, var_list=list_of_var)
        saver = tf.train.Saver()

        y = tf.placeholder(tf.float32, [None, self.number_of_classes])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=multilayer_perceptron, labels=y))
        acc_num, acc_prob = nlm.config.accuracy_function(y, multilayer_perceptron)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(multilayer_perceptron, 1)

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            def get_batch_data(xi, yi, batch_size, kp, n_in, n_hidden, r_base, n_class, is_shuffle=True):
                for index in nlm.internal_data_loader.batch_index(len(yi), batch_size, is_shuffle):
                    feed_dict = {
                        x: xi[index],
                        y: yi[index],
                        tf_keep_prob: kp,
                    }
                    yield feed_dict, len(index)

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

                print('all samples={}, correct prediction={}'.format(train_cnt, train_acc))
                train_acc = train_acc / train_cnt
                print('Iter {}: train acc={:.6f}'.format(epoch, train_acc))
            saver.save(sess, file_to_save)

            tf.reset_default_graph()
            sess.close()

    def predict(self, x, file_to_save, id):

        self.id = id

        with tf.Session() as session:
            # restore the model
            new_saver = tf.train.import_meta_graph(file_to_save + ".meta")
            new_saver.restore(session, file_to_save)

            activation = tf.get_collection("mlp_activation_" + self.model_name + self.id)[0]
            keep_prob = session.graph.get_tensor_by_name('tf_keep_prob')
            new_x = session.graph.get_tensor_by_name('x')

            feed_dict = {
                new_x: x,
                keep_prob: 1.0
            }

            predictions = session.run(activation, feed_dict=feed_dict)

        return predictions
