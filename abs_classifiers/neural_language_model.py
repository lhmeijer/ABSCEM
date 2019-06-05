import tensorflow as tf


class NeuralLanguageModel:

    def __int__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    def model_itself(self, left_sentence_parts, left_sentence_lengths, right_sentence_parts, right_sentence_lengths,
                     target_parts, target_lengths):
        return None, None

    def fit(self, x_train, y_train, train_aspects, x_test, y_test, test_aspects):

        self.left_part = tf.placeholder(tf.int32, [None, self.config.max_sentence_length])
        self.left_sen_len = tf.placeholder(tf.int32, None)

        self.right_part = tf.placeholder(tf.int32, [None, self.config.max_sentence_length])
        self.right_sen_len = tf.placeholder(tf.int32, [None])

        self.target_part = tf.placeholder(tf.int32, [None, self.config.max_target_length])
        self.tar_len = tf.placeholder(tf.int32, [None])

        y = tf.placeholder(tf.float32, [None, self.config.number_of_classes])

        self.prob, self.layer_information = self.model_itself(left_sentence_parts=self.left_part, left_sentence_lengths=self.left_sen_len,
                                    right_sentence_parts=self.right_part, right_sentence_lengths=self.right_sen_len,
                                    target_parts=self.target_part, target_lengths=self.tar_len)

        loss = self.config.loss_function(y, self.prob)
        acc_num, acc_prob = self.config.accuracy_function(y, self.prob)

        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=self.config.momentum)
        train_optimizer = optimizer.minimize(loss, global_step=global_step)

        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(self.prob, 1)

        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        with tf.Session() as session:

            session.run(init)

            tr_left_part, tr_target_part, tr_right_part, tr_left_sen_len, tr_tar_len, tr_right_sen_len = \
                self.internal_data_loader.split_embeddings_in_left_target_right(x_train, train_aspects,
                                                self.config.max_sentence_length, self.config.max_target_length)

            te_left_part, te_target_part, te_right_part, te_left_sen_len, te_tar_len, te_right_sen_len = \
                self.internal_data_loader.split_embeddings_in_left_target_right(x_test, test_aspects,
                                                self.config.max_sentence_length, self.config.max_target_length)

            def get_batch_data(x_left, len_left, x_right, len_right, yi, x_target, len_target, batch_size):
                for index in self.internal_data_loader.batch_index(len(yi), batch_size):
                    feed_dict = {
                        self.left_part: x_left[index],
                        self.right_part: x_right[index],
                        y: yi[index],
                        self.left_sen_len: len_left[index],
                        self.right_sen_len: len_right[index],
                        self.target_part: x_target[index],
                        self.tar_len: len_target[index],
                    }
                    yield feed_dict, len(index)

            for i in range(self.config.number_of_iterations):
                for train, _ in get_batch_data(tr_left_part, tr_left_sen_len, tr_right_part, te_right_sen_len,
                                               y_train, tr_target_part,tr_tar_len, self.config.batch_size):
                    _ = session.run([train_optimizer], feed_dict=train)

            self.saver.save(session, self.config.file_to_save_model)

            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            for test, num in get_batch_data(te_left_part, te_left_sen_len, te_right_part, te_right_sen_len, y_test,
                                            te_target_part, te_tar_len, 1):
                _loss, _acc, _ty, _py, _p = session.run([loss, acc_num, true_y, pred_y, self.prob], feed_dict=test)
                ty += list(_ty)
                py += list(_py)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
        print
        'all samples={}, correct prediction={}'.format(cnt, acc)
        acc = acc / cnt
        cost = cost / cnt
        print
        'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, cost, acc)
        summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
        test_summary_writer.add_summary(summary, step)
        if acc > max_acc:
            max_acc = acc
            max_fw = fw
            max_bw = bw
            max_tl = tl
            max_tr = tr
            max_ty = ty
            max_py = py
            max_prob = p

    def predict(self, x, x_aspect):

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.internal_data_loader.split_test_embeddings_in_left_target_right(x, x_aspect,
                                                self.config.max_sentence_length, self.config.max_target_length)

        feed_dict = {
            self.left_part: x_left_part,
            self.right_part: x_right_part,
            self.left_sen_len: x_left_sen_len,
            self.right_sen_len: x_right_sen_len,
            self.target_part: x_target_part,
            self.tar_len: x_tar_len,
        }

        with tf.Session() as session:
            # restore the model
            self.saver.restore(session, self.config.file_to_save_model)
            predictions, layer_information = session.run([self.prob, self.layer_information], feed_dict=feed_dict)
        return predictions, layer_information


    def run(self):

        results = {
            'classification model': self.config.name_of_model,
            'size_of_training_set': len(self.internal_data_loader.lemmatized_training),
            'size_of_test_set': len(self.internal_data_loader.lemmatized_test),
            'size_of_cross_validation_sets': 0,
            'out_of_sample_accuracy_majority': 0,
            'in_sample_accuracy_majority': 0,
            'cross_val_accuracy_majority': [],
            'cross_val_mean_accuracy_majority': "cross validation is switched off",
            'cross_val_stdeviation_majority': "cross validation is switched off",
            'out_of_sample_accuracy_with_backup': "No backup model is used for ontology reasoner",
            'in_sample_accuracy_with_backup': "No backup model is used for ontology reasoner",
            'cross_val_accuracy_with_backup': [],
            'cross_val_mean_accuracy_with_backup': "cross validation is switched off",
            'cross_val_stdeviation_with_backup': "cross validation is switched off"
        }

        x_train = self.internal_data_loader.word_embeddings_training_all
        train_aspects = self.internal_data_loader.aspect_indices_training
        y_train = self.internal_data_loader.polarity_matrix_training

        x_test = self.internal_data_loader.word_embeddings_test_all
        test_aspects = self.internal_data_loader.aspect_indices_test
        y_test = self.internal_data_loader.polarity_matrix_test

        if self.config.cross_validation:

            training_indices, test_indices = self.internal_data_loader.get_random_indices_for_cross_validation(
                self.config.cross_validation_rounds, x_train.shape[0])

            for i in range(self.config.cross_validation_rounds):

                acc = self.fit(x_train=x_train[training_indices[i]], y_train=y_train[training_indices[i]],
                               train_aspects=train_aspects[training_indices[i]], x_test=x_train[test_indices[i]],
                               y_test=y_train[test_indices[i]], test_aspects=train_aspects[test_indices[i]])

        else:

            acc = self.fit(x_train=x_train, y_train=y_train, train_aspects=train_aspects, x_test=x_test,
                           y_test=y_test, test_aspects=test_aspects)



