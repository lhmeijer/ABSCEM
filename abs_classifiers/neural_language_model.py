import tensorflow as tf
import json


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

            max_test_acc = 0.

            for i in range(self.config.number_of_iterations):

                train_acc, train_cnt = 0., 0

                for train, train_num in get_batch_data(tr_left_part, tr_left_sen_len, tr_right_part, te_right_sen_len,
                                               y_train, tr_target_part,tr_tar_len, self.config.batch_size):
                    _, step, _train_acc = session.run([train_optimizer, global_step, acc_num], feed_dict=train)
                    train_acc += _train_acc
                    train_cnt += train_num

                test_acc, test_cost, test_cnt = 0., 0., 0
                true_y_s, pred_y_s, prob_s = [], [], []
                for test, test_num in get_batch_data(te_left_part, te_left_sen_len, te_right_part, te_right_sen_len, y_test,
                                            te_target_part, te_tar_len, 1):
                    _loss, _test_acc, _true_y, _pred_y, _prob = session.run([loss, acc_num, true_y, pred_y, self.prob],
                                                                            feed_dict=test)
                    true_y_s.append(_true_y)
                    pred_y_s.append(_pred_y)
                    prob_s.append(_prob)
                    test_acc += _test_acc
                    test_cost += _loss * test_num
                    test_cnt += test_num

                print('all samples={}, correct prediction={}'.format(test_cnt, test_acc))
                train_acc = train_acc / train_cnt
                test_acc = test_acc / test_cnt
                test_cost = test_cost / test_cnt
                print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}'.format(i, test_cost, train_acc, test_acc))

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    results = {
                        'classification model': self.config.name_of_model,
                        'n_in_training_sample': train_cnt,
                        'n_in_test_sample': test_cnt,
                        'max_test_acc': test_acc,
                        'train_acc': train_acc,
                        'true_y': true_y_s,
                        'pred_y': pred_y_s,
                        'prob': prob_s,
                        'iteration': i,
                        'number_of_iterations': self.config.number_of_iterations,
                        'learning_rate': self.config.learning_rate,
                        'batch_size': self.config.batch_size,
                        'L2_regularization': self.config.l2_regularization,
                        'number_hidden_units': self.config.number_hidden_units
                    }

            self.saver.save(session, self.config.file_to_save_model)

            return results

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

        x_train = self.internal_data_loader.word_embeddings_training_all
        train_aspects = self.internal_data_loader.aspect_indices_training
        y_train = self.internal_data_loader.polarity_matrix_training

        x_test = self.internal_data_loader.word_embeddings_test_all
        test_aspects = self.internal_data_loader.aspect_indices_test
        y_test = self.internal_data_loader.polarity_matrix_test

        results = None

        if self.config.cross_validation:

            training_indices, test_indices = self.internal_data_loader.get_random_indices_for_cross_validation(
                self.config.cross_validation_rounds, x_train.shape[0])

            results = {}

            for i in range(self.config.cross_validation_rounds):

                x_train = x_train[training_indices[i]]
                y_train = y_train[training_indices[i]]
                train_aspects = train_aspects[training_indices[i]]
                x_test = x_train[test_indices[i]]
                y_test = y_train[test_indices[i]]
                test_aspects = train_aspects[test_indices[i]]

                if self.config.use_of_ontology:
                    remaining_indices = self.internal_data_loader.read_remaining_data(is_cross_val=True)
                    x_test = x_test[remaining_indices]
                    y_test = y_test[remaining_indices]
                    test_aspects = test_aspects[remaining_indices]

                result = self.fit(x_train=x_train, y_train=y_train, train_aspects=train_aspects, x_test=x_test,
                                   y_test=y_test, test_aspects=test_aspects)
                results['cross_validation_' + str(i)] = result

        else:

            if self.config.use_of_ontology:

                remaining_indices = self.internal_data_loader.read_remaining_data(is_cross_val=False)
                x_test = x_test[remaining_indices]
                y_test = y_test[remaining_indices]
                test_aspects = test_aspects[remaining_indices]

            results = self.fit(x_train=x_train, y_train=y_train, train_aspects=train_aspects, x_test=x_test,
                           y_test=y_test, test_aspects=test_aspects)

        with open(self.config.file_of_results, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False)


