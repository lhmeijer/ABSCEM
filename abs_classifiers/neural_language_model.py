import tensorflow as tf
import json
import numpy as np


class NeuralLanguageModel:

    def __init__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    def model_itself(self, left_sentence_parts, left_sentence_lengths, right_sentence_parts, right_sentence_lengths,
                     target_parts, target_lengths, keep_prob1, keep_prob2):
        return None, None

    def fit(self, x_train, y_train, train_aspects, x_test, y_test, test_aspects):

        keep_prob1 = tf.placeholder(tf.float32, name='keep_prob1')
        keep_prob2 = tf.placeholder(tf.float32, name='keep_prob2')

        left_part = tf.placeholder(tf.float32, [None, self.config.max_sentence_length,
                                   self.config.embedding_dimension], name='left_x')
        left_sen_len = tf.placeholder(tf.int32, [None], name='left_len')

        right_part = tf.placeholder(tf.float32, [None, self.config.max_sentence_length,
                                    self.config.embedding_dimension], name='right_x')
        right_sen_len = tf.placeholder(tf.int32, [None], name='right_len')

        target_part = tf.placeholder(tf.float32, [None, self.config.max_target_length,
                                     self.config.embedding_dimension], name='target_x')
        tar_len = tf.placeholder(tf.int32, [None], name='target_len')

        y = tf.placeholder(tf.float32, [None, self.config.number_of_classes])

        prob, layer_information = self.model_itself(left_sentence_parts=left_part,
                                                    left_sentence_lengths=left_sen_len,
                                                    right_sentence_parts=right_part,
                                                    right_sentence_lengths=right_sen_len,
                                                    target_parts=target_part,
                                                    target_lengths=tar_len,
                                                    keep_prob1=keep_prob1, keep_prob2=keep_prob2)

        # Saving procedure for the Tensorflow graph
        tf.add_to_collection(self.config.name_of_model + "_prob", prob)
        tf.add_to_collection(self.config.name_of_model + "_lhs", layer_information['left_hidden_state'])
        tf.add_to_collection(self.config.name_of_model + "_ths", layer_information['target_hidden_state'])
        tf.add_to_collection(self.config.name_of_model + "_rhs", layer_information['right_hidden_state'])

        if "LCR_Rot_hop_model" in self.config.name_of_model:
            for i in range(self.config.n_iterations_hop):
                tf.add_to_collection(self.config.name_of_model + "_lws_" + str(i),
                                     layer_information['weighted_left_hidden_state_' + str(i)])
                tf.add_to_collection(self.config.name_of_model + "_tlws_" + str(i),
                                     layer_information['weighted_target_left_hidden_state_' + str(i)])
                tf.add_to_collection(self.config.name_of_model + "_trws_" + str(i),
                                     layer_information['weighted_target_right_hidden_state_' + str(i)])
                tf.add_to_collection(self.config.name_of_model + "_rws_" + str(i),
                                     layer_information['weighted_right_hidden_state_' + str(i)])
                tf.add_to_collection(self.config.name_of_model + "_lac_" + str(i),
                                     layer_information['left_attention_score_' + str(i)])
                tf.add_to_collection(self.config.name_of_model + "_tlac_" + str(i),
                                     layer_information['target_left_attention_score_' + str(i)])
                tf.add_to_collection(self.config.name_of_model + "_trac_" + str(i),
                                     layer_information['target_right_attention_score_' + str(i)])
                tf.add_to_collection(self.config.name_of_model + "_rac_" + str(i),
                                     layer_information['right_attention_score_' + str(i)])

        else:
            tf.add_to_collection(self.config.name_of_model + "_lws", layer_information['weighted_left_hidden_state'])
            tf.add_to_collection(self.config.name_of_model + "_tlws",
                                 layer_information['weighted_target_left_hidden_state'])
            tf.add_to_collection(self.config.name_of_model + "_trws",
                                 layer_information['weighted_target_right_hidden_state'])
            tf.add_to_collection(self.config.name_of_model + "_rws", layer_information['weighted_right_hidden_state'])
            tf.add_to_collection(self.config.name_of_model + "_lac", layer_information['left_attention_score'])
            tf.add_to_collection(self.config.name_of_model + "_tlac", layer_information['target_left_attention_score'])
            tf.add_to_collection(self.config.name_of_model + "_trac", layer_information['target_right_attention_score'])
            tf.add_to_collection(self.config.name_of_model + "_rac", layer_information['right_attention_score'])

        loss = self.config.loss_function(y, prob)
        acc_num, acc_prob = self.config.accuracy_function(y, prob)

        global_step = tf.Variable(0, name='tr_global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=self.config.momentum).\
            minimize(loss, global_step=global_step)

        true_y = tf.argmax(y, 1)
        pred_y = tf.argmax(prob, 1)

        saver = tf.train.Saver()

        with tf.Session() as session:

            session.run(tf.global_variables_initializer())

            tr_left_part, tr_target_part, tr_right_part, tr_left_sen_len, tr_tar_len, tr_right_sen_len = \
                self.internal_data_loader.split_embeddings(x_train, train_aspects, self.config.max_sentence_length,
                                                           self.config.max_target_length)

            te_left_part, te_target_part, te_right_part, te_left_sen_len, te_tar_len, te_right_sen_len = \
                self.internal_data_loader.split_embeddings(x_test, test_aspects, self.config.max_sentence_length,
                                                           self.config.max_target_length)

            def get_batch_data(x_left, len_left, x_right, len_right, yi, x_target, len_target, batch_size, kp1, kp2,
                               is_shuffle=True):
                for indices in self.internal_data_loader.batch_index(len(yi), batch_size, is_shuffle):
                    feed_dict = {
                        left_part: x_left[indices],
                        right_part: x_right[indices],
                        y: yi[indices],
                        left_sen_len: len_left[indices],
                        right_sen_len: len_right[indices],
                        target_part: x_target[indices],
                        tar_len: len_target[indices],
                        keep_prob1: kp1,
                        keep_prob2: kp2,
                    }
                    yield feed_dict, len(indices)

            max_test_acc = 0.

            for i in range(self.config.number_of_iterations):

                train_acc, train_cnt = 0., 0
                tr_prediction_y, tr_correct_prediction_y = np.array([0, 0, 0]), np.array([0, 0, 0])

                for train, train_num in get_batch_data(tr_left_part, tr_left_sen_len, tr_right_part, tr_right_sen_len,
                                                       y_train, tr_target_part, tr_tar_len, self.config.batch_size,
                                                       self.config.keep_prob1, self.config.keep_prob2):
                    _, _, _train_acc, _true_y, _pred_y = session.run([optimizer, global_step, acc_num, true_y, pred_y],
                                                                     feed_dict=train)

                    for index in range(_pred_y.shape[0]):
                        tr_prediction_y[_pred_y[index]] += 1
                        if _pred_y[index] == _true_y[index]:
                            tr_correct_prediction_y[_pred_y[index]] += 1

                    train_acc += _train_acc
                    train_cnt += train_num

                if i % 50 == 0 and not self.config.cross_validation:
                    saver.save(session, self.config.file_to_save_model, global_step=i)

                test_acc, test_cost, test_cnt = 0., 0., 0
                te_prediction_y, te_correct_prediction_y = np.array([0, 0, 0]), np.array([0, 0, 0])
                for test, test_num in get_batch_data(te_left_part, te_left_sen_len, te_right_part, te_right_sen_len,
                                                     y_test, te_target_part, te_tar_len, 2000, 1.0, 1.0, False):

                    _loss, _test_acc, _true_y, _pred_y = session.run([loss, acc_num, true_y, pred_y], feed_dict=test)
                    test_acc += _test_acc
                    test_cost += _loss * test_num
                    test_cnt += test_num

                    for index in range(_pred_y.shape[0]):
                        te_prediction_y[_pred_y[index]] += 1
                        if _pred_y[index] == _true_y[index]:
                            te_correct_prediction_y[_pred_y[index]] += 1

                print('all samples={}, correct prediction={}'.format(test_cnt, test_acc))
                train_acc = train_acc / train_cnt
                test_acc = test_acc / test_cnt
                test_cost = test_cost / test_cnt
                print('Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}'.format(i, test_cost,
                                                                                                  train_acc, test_acc))

                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    results = {
                        'classification model': self.config.name_of_model,
                        'n_in_training_sample': train_cnt,
                        'n_in_test_sample': test_cnt,
                        'max_test_acc': test_acc,
                        'train_acc': train_acc,
                        'train_predicted_per_class': tr_prediction_y.tolist(),
                        'train_correct_predicted_per_class': tr_correct_prediction_y.tolist(),
                        'test_predicted_per_class': te_prediction_y.tolist(),
                        'test_correct_predicted_per_class': te_correct_prediction_y.tolist(),
                        'iteration': i,
                        'number_of_iterations': self.config.number_of_iterations,
                        'learning_rate': self.config.learning_rate,
                        'batch_size': self.config.batch_size,
                        'L2_regularization': self.config.l2_regularization,
                        'number_hidden_units': self.config.number_hidden_units,
                        'momentum': self.config.momentum,
                        'keep_prob1': self.config.keep_prob1,
                        'keep_prob2': self.config.keep_prob2
                    }

            if not self.config.cross_validation:
                saver.save(session, self.config.file_to_save_model)

            session.close()

            return results

    def predict(self, x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len):

        graph = tf.Graph()

        session = tf.Session(graph=graph)
        with graph.as_default():

            new_saver = tf.train.import_meta_graph(self.config.file_to_save_model + ".meta")

            collections = list()

            collections.append(tf.get_collection(self.config.name_of_model + "_prob"))
            collections.append(tf.get_collection(self.config.name_of_model + "_lhs"))
            collections.append(tf.get_collection(self.config.name_of_model + "_rhs"))

            if "LCR_Rot_hop_model" in self.config.name_of_model:
                for i in range(self.config.n_iterations_hop):
                    collections.append(tf.get_collection(self.config.name_of_model + "_lws_" + str(i)))
                    collections.append(tf.get_collection(self.config.name_of_model + "_rws_" + str(i)))
                    collections.append(tf.get_collection(self.config.name_of_model + "_lac_" + str(i)))
                    collections.append(tf.get_collection(self.config.name_of_model + "_rac_" + str(i)))
            else:
                collections.append(tf.get_collection(self.config.name_of_model + "_lws"))
                collections.append(tf.get_collection(self.config.name_of_model + "_rws"))
                collections.append(tf.get_collection(self.config.name_of_model + "_lac"))
                collections.append(tf.get_collection(self.config.name_of_model + "_rac"))

            new_saver.restore(session, self.config.file_to_save_model)

            keep_prob1 = session.graph.get_tensor_by_name("keep_prob1:0")
            keep_prob2 = session.graph.get_tensor_by_name("keep_prob2:0")
            left_part = session.graph.get_tensor_by_name("left_x:0")
            right_part = session.graph.get_tensor_by_name("right_x:0")
            target_part = session.graph.get_tensor_by_name("target_x:0")
            left_sen_len = session.graph.get_tensor_by_name("left_len:0")
            right_sen_len = session.graph.get_tensor_by_name("right_len:0")
            tar_len = session.graph.get_tensor_by_name("target_len:0")

            feed_dict = {
                left_part: x_left_part,
                right_part: x_right_part,
                left_sen_len: x_left_sen_len,
                right_sen_len: x_right_sen_len,
                target_part: x_target_part,
                tar_len: x_tar_len,
                keep_prob1: 1.0,
                keep_prob2: 1.0
            }

            result_of_collections = session.run(collections, feed_dict=feed_dict)

            session.close()

        predictions = result_of_collections[0][0]
        layer_information = {
            'left_hidden_state': result_of_collections[1],
            'right_hidden_state': result_of_collections[2]
        }
        if "LCR_Rot_hop_model" in self.config.name_of_model:
            for i in range(self.config.n_iterations_hop):
                layer_information['weighted_left_hidden_state_' + str(i)] = result_of_collections[3 + i * 4]
                layer_information['weighted_right_hidden_state_' + str(i)] = result_of_collections[4 + i * 4]
                layer_information['left_attention_score_' + str(i)] = result_of_collections[5 + i * 4]
                layer_information['right_attention_score_' + str(i)] = result_of_collections[6 + i * 4]
        else:
            layer_information['weighted_left_hidden_state'] = result_of_collections[3]
            layer_information['weighted_right_hidden_state'] = result_of_collections[4]
            layer_information['left_attention_score'] = result_of_collections[5]
            layer_information['right_attention_score'] = result_of_collections[6]

        return predictions, layer_information

    def run(self):

        x_train = np.array(self.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.internal_data_loader.aspect_indices_training)
        y_train = np.array(self.internal_data_loader.polarity_matrix_training)

        x_test = np.array(self.internal_data_loader.word_embeddings_test_all)
        test_aspects = np.array(self.internal_data_loader.aspect_indices_test)
        y_test = np.array(self.internal_data_loader.polarity_matrix_test)

        if self.config.cross_validation:

            training_indices, validation_indices = self.internal_data_loader.get_indices_cross_validation()

            results = {}

            train_single_accs = list(range(self.config.cross_validation_rounds))
            validation_single_accs = list(range(self.config.cross_validation_rounds))

            for i in range(self.config.cross_validation_rounds):

                x_train_cross = x_train[training_indices[i]]
                y_train_cross = y_train[training_indices[i]]
                train_aspects_cross = train_aspects[training_indices[i]]
                x_validation_cross = x_train[validation_indices[i]]
                y_validation_cross = y_train[validation_indices[i]]
                validation_aspects_cross = train_aspects[validation_indices[i]]

                if self.config.hybrid_method:
                    remaining_indices = self.internal_data_loader.read_remaining_data(is_cross_val=True)
                    x_validation_cross = x_validation_cross[remaining_indices]
                    y_validation_cross = y_validation_cross[remaining_indices]
                    validation_aspects_cross = validation_aspects_cross[remaining_indices]

                result = self.fit(x_train=x_train_cross, y_train=y_train_cross, train_aspects=train_aspects_cross,
                                  x_test=x_validation_cross, y_test=y_validation_cross,
                                  test_aspects=validation_aspects_cross)
                train_single_accs[i] = result['train_acc']
                validation_single_accs[i] = result['max_test_acc']
                results['cross_validation_' + str(i)] = result

            results['mean_accuracy_train_single_acc'] = np.mean(train_single_accs)
            results['standard_deviation_train_single_acc'] = np.std(train_single_accs)
            results['mean_accuracy_validation_max_acc'] = np.mean(validation_single_accs)
            results['standard_deviation_validation_max_acc'] = np.std(validation_single_accs)

            with open(self.config.file_of_cross_val_results, 'w') as outfile:
                json.dump(results, outfile, ensure_ascii=False)

        else:

            if self.config.hybrid_method:

                remaining_indices = self.internal_data_loader.read_remaining_data(is_cross_val=False)
                x_test = x_test[remaining_indices]
                y_test = y_test[remaining_indices]
                test_aspects = test_aspects[remaining_indices]

            results = self.fit(x_train=x_train, y_train=y_train, train_aspects=train_aspects, x_test=x_test,
                               y_test=y_test, test_aspects=test_aspects)

            with open(self.config.file_of_results, 'w') as outfile:
                json.dump(results, outfile, ensure_ascii=False)
