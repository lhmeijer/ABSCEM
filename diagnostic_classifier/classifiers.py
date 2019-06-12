import tensorflow as tf


class SingleMLPClassifier:

    def __init__(self, learning_rate, number_hidden_units, number_of_epochs, keep_prob, batch_size, random_base, model_name):
        self.learning_rate = learning_rate
        self.number_hidden_units = number_hidden_units
        self.number_of_epochs = number_of_epochs
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.random_base = random_base
        self.model_name = model_name

    def multilayer_perceptron(self, x, keep_prob, n_input, number_hidden_units, random_base, n_classes):
        print("x ", x)

        batch_size = tf.shape(x)[0]
        max_len = tf.shape(x)[1]
        word_embeddings = tf.shape(x)[2]
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
        print("x_drop_out ", x_drop_out)
        inputs = tf.reshape(x_drop_out, [-1, n_input])
        print("inputs ", inputs)
        layer_1 = tf.reshape(tf.add(tf.matmul(inputs, weights['h1']), biases['b1']), [-1, max_len, number_hidden_units])
        print("layer_1 ", layer_1)
        layer_1 = tf.nn.relu(layer_1)
        inputs_layer_1 = tf.reshape(layer_1, [-1, number_hidden_units])
        print("inputs_layer_1 ", inputs_layer_1)
        out_layer = tf.reshape(tf.add(tf.matmul(inputs_layer_1, weights['out']), biases['out']), [-1, max_len, n_classes])
        print("out_layer ", out_layer)
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    def fit(self, train_x, train_y, test_x, test_y, nlm):

        print("train_x ", train_x.shape)
        print("train_y ", train_y.shape)

        number_of_input = train_x.shape[1]
        n_embeddings = train_x.shape[2]
        number_of_classes = train_y.shape[1]

        keep_prob = tf.placeholder(tf.float32)
        # n_input = tf.placeholder(tf.int32)
        # number_hidden_units = tf.placeholder(tf.int32)
        # random_base = tf.placeholder(tf.float32)
        # n_classes = tf.placeholder(tf.int32)

        display_step = 10

        x = tf.placeholder(tf.float32, [None, number_of_input, n_embeddings])
        y = tf.placeholder(tf.float32, [None, number_of_classes])

        predictions = self.multilayer_perceptron(x, keep_prob, n_embeddings, self.number_hidden_units, self.random_base, number_of_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
        acc_num, acc_prob = nlm.config.accuracy_function(y, predictions)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(predictions, 1)

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            def get_batch_data(xi, yi, batch_size, kp, n_in, n_hidden, r_base, n_class, is_shuffle=True):
                for index in nlm.internal_data_loader.batch_index(len(yi), batch_size, is_shuffle):
                    feed_dict = {
                        x: xi[index],
                        y: yi[index],
                        keep_prob: kp,
                        # n_input: n_in,
                        # number_hidden_units: n_hidden,
                        # random_base: r_base,
                        # n_classes: n_class
                    }
                    yield feed_dict, len(index)

            max_test_acc = 0.
            max_train_acc = 0.

            for epoch in range(self.number_of_epochs):
                train_acc, train_cnt = 0., 0
                for train, train_num in get_batch_data(train_x, train_y, self.batch_size, self.keep_prob,
                                                       n_embeddings, self.number_hidden_units, self.random_base,
                                                       number_of_classes):
                    _, _train_acc, = sess.run([optimizer, acc_num], feed_dict=train)
                    train_acc += _train_acc
                    train_cnt += train_num
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "train_acc=", "{:.9f}".format(train_acc))

                test_acc, test_cost, test_cnt = 0., 0., 0
                true_y_s, pred_y_s, prob_s = [], [], []
                for test, test_num in get_batch_data(test_x, test_y, 2000, 1.0, n_embeddings, self.number_hidden_units,
                                                     self.random_base, number_of_classes, False):
                    _loss, _test_acc, _true_y, _pred_y, _prob = sess.run([loss, acc_num, true_y, pred_y, predictions],
                                                                         feed_dict=test)
                    true_y_s.append(_true_y.tolist())
                    pred_y_s.append(_pred_y.tolist())
                    prob_s.append(_prob.tolist())
                    test_acc += _test_acc
                    test_cost += _loss * test_num
                    test_cnt += test_num

                print('all samples={}, correct prediction={}'.format(test_cnt, test_acc))
                train_acc = train_acc / train_cnt
                test_acc = test_acc / test_cnt
                test_cost = test_cost / test_cnt
                print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}'.format(epoch, test_cost,
                                                                                                  train_acc, test_acc))

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    max_train_acc = train_acc
                    results = {
                        'classification model': self.model_name,
                        'n_in_training_sample': train_cnt,
                        'n_in_test_sample': test_cnt,
                        'max_test_acc': test_acc,
                        'train_acc': train_acc,
                        'true_y': true_y_s,
                        'pred_y': pred_y_s,
                        'prob': prob_s,
                        'iteration': epoch,
                        'number_of_iterations': self.number_of_epochs,
                        'learning_rate': self.learning_rate,
                        'batch_size': self.batch_size,
                        'number_hidden_units': self.number_hidden_units,
                        'keep_prob': self.keep_prob
                    }

            return [max_train_acc, max_test_acc]
