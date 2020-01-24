from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierAspectSentimentConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierWordSentimentConfig, DiagnosticClassifierMentionConfig
import os
import numpy as np
import json


class DiagnosticClassifier:

    def __init__(self, nl_model, diagnostic_classifiers):
        self.nl_model = nl_model
        self.diagnostic_classifiers = diagnostic_classifiers
        self.configurations = {
            'word_sentiments': DiagnosticClassifierWordSentimentConfig,
            'aspect_sentiments': DiagnosticClassifierAspectSentimentConfig,
            'relations': DiagnosticClassifierRelationConfig,
            'pos_tags': DiagnosticClassifierPOSConfig,
            'mentions': DiagnosticClassifierMentionConfig,
        }

        if not os.path.isfile(self.nl_model.config.tr_file_of_hid_correct_pred) or not os.path.isfile(
                self.nl_model.config.te_file_of_hid_correct_pred):
            self.create_files_with_all_hidden_layers()

    def create_files_with_all_hidden_layers(self):

        print("I am creating the files containing the hidden layers")

        with open(self.nl_model.config.file_of_indices, 'r') as file:
            for line in file:
                indices = json.loads(line)

        tr_corr_pred = indices['tr_correct_predicted']
        te_corr_pred = indices['te_correct_predicted']
        te_incorrect_pred = indices['te_wrong_predicted']

        x_embedding_tr = np.array(self.nl_model.internal_data_loader.word_embeddings_training_all)
        tr_aspect_indices = np.array(self.nl_model.internal_data_loader.aspect_indices_training)
        tr_word_sentiments = np.array(self.nl_model.internal_data_loader.word_sentiments_in_training)
        tr_aspect_sentiments = np.array(self.nl_model.internal_data_loader.aspect_sentiments_in_training)
        tr_relations = np.array(self.nl_model.internal_data_loader.word_relations_training)
        tr_pos_tags = np.array(self.nl_model.internal_data_loader.part_of_speech_training)
        tr_mentions = np.array(self.nl_model.internal_data_loader.mentions_in_training)

        values_of_hidden_layers_tr = self.obtain_data_set(x_embedding_tr[tr_corr_pred], tr_aspect_indices[tr_corr_pred],
                                                          tr_word_sentiments[tr_corr_pred],
                                                          tr_aspect_sentiments[tr_corr_pred],
                                                          tr_relations[tr_corr_pred], tr_pos_tags[tr_corr_pred],
                                                          tr_mentions[tr_corr_pred])

        with open(self.nl_model.config.tr_file_of_hid_correct_pred, 'w') as outfile:
            json.dump(values_of_hidden_layers_tr, outfile, ensure_ascii=False)

        x_embedding_te = np.array(self.nl_model.internal_data_loader.word_embeddings_test_all)
        te_aspect_indices = np.array(self.nl_model.internal_data_loader.aspect_indices_test)
        te_word_sentiments = np.array(self.nl_model.internal_data_loader.word_sentiments_in_test)
        te_aspect_sentiments = np.array(self.nl_model.internal_data_loader.aspect_sentiments_in_test)
        te_relations = np.array(self.nl_model.internal_data_loader.word_relations_test)
        te_pos_tags = np.array(self.nl_model.internal_data_loader.part_of_speech_test)
        te_mentions = np.array(self.nl_model.internal_data_loader.mentions_in_test)

        values_of_hidden_layers_correct_te = self.obtain_data_set(x_embedding_te[te_corr_pred],
                                                                  te_aspect_indices[te_corr_pred],
                                                                  te_word_sentiments[te_corr_pred],
                                                                  te_aspect_sentiments[te_corr_pred],
                                                                  te_relations[te_corr_pred],
                                                                  te_pos_tags[te_corr_pred],
                                                                  te_mentions[te_corr_pred])

        with open(self.nl_model.config.te_file_of_hid_correct_pred, 'w') as outfile:
            json.dump(values_of_hidden_layers_correct_te, outfile, ensure_ascii=False)

        values_of_hidden_layers_incorrect_te = self.obtain_data_set(x_embedding_te[te_incorrect_pred],
                                                                    te_aspect_indices[te_incorrect_pred],
                                                                    te_word_sentiments[te_incorrect_pred],
                                                                    te_aspect_sentiments[te_incorrect_pred],
                                                                    te_relations[te_incorrect_pred],
                                                                    te_pos_tags[te_incorrect_pred],
                                                                    te_mentions[te_incorrect_pred])

        with open(self.nl_model.config.te_file_of_hid_incorrect_pred, 'w') as outfile:
            json.dump(values_of_hidden_layers_incorrect_te, outfile, ensure_ascii=False)

    def obtain_data_set(self, x, aspect_indices, word_sentiments, aspect_sentiments, relations, pos_tags, mentions):

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.nl_model.internal_data_loader.split_embeddings(x, aspect_indices,
                                                                self.nl_model.config.max_sentence_length,
                                                                self.nl_model.config.max_target_length)

        data_set = {'word_embeddings': [], 'hidden_states': [], 'word_sentiments': [], 'aspect_sentiments': [],
                    'relations': [], 'pos_tags': [], 'mentions': []}

        if "LCR_Rot_hop_model" in self.nl_model.config.name_of_model:
            for i in range(self.nl_model.config.n_iterations_hop):
                data_set['weighted_hidden_states_' + str(i)] = []
        else:
            data_set['weighted_hidden_states'] = []

        statistics = {
            'word_sentiments_count': np.zeros(len(word_sentiments[0][0]), dtype=int),
            'aspect_sentiments_count': np.zeros(len(aspect_sentiments[0][0]), dtype=int),
            'relations_count': np.zeros(len(relations[0][0]), dtype=int),
            'pos_tags_count': np.zeros(len(pos_tags[0][0]), dtype=int),
            'mentions_count': np.zeros(len(mentions[0][0]), dtype=int)
        }

        for index in range(x.shape[0]):

            print("index ", index)

            tr_pred, tr_layer_information = self.nl_model.predict(np.array([x_left_part[index]]),
                                                                  np.array([x_target_part[index]]),
                                                                  np.array([x_right_part[index]]),
                                                                  np.array([x_left_sen_len[index]]),
                                                                  np.array([x_tar_len[index]]),
                                                                  np.array([x_right_sen_len[index]]))
            n_left_words = x_left_sen_len[index]

            for j in range(n_left_words):
                data_set['word_embeddings'].append(x_left_part[index][j].tolist())
                data_set['hidden_states'].append(tr_layer_information['left_hidden_state'][0][0][j].tolist())

                if "LCR_Rot_hop_model" in self.nl_model.config.name_of_model:

                    for i in range(self.nl_model.config.n_iterations_hop):
                        data_set['weighted_hidden_states_' + str(i)].append(
                            tr_layer_information['weighted_left_hidden_state_' + str(i)][0][0][j].tolist())
                else:
                    data_set['weighted_hidden_states'].append(
                        tr_layer_information['weighted_left_hidden_state'][0][0][j].tolist())

                data_set['word_sentiments'].append(word_sentiments[index][j])
                index_of_one = word_sentiments[index][j].index(1)
                statistics['word_sentiments_count'][index_of_one] += 1

                data_set['aspect_sentiments'].append(aspect_sentiments[index][j])
                index_of_one = aspect_sentiments[index][j].index(1)
                statistics['aspect_sentiments_count'][index_of_one] += 1

                data_set['relations'].append(relations[index][j])
                index_of_one = relations[index][j].index(1)
                statistics['relations_count'][index_of_one] += 1

                data_set['pos_tags'].append(pos_tags[index][j])
                index_of_one = pos_tags[index][j].index(1)
                statistics['pos_tags_count'][index_of_one] += 1

                data_set['mentions'].append(mentions[index][j])
                index_of_one = mentions[index][j].index(1)
                statistics['mentions_count'][index_of_one] += 1

            n_right_words = x_right_sen_len[index]
            last_aspect_index = aspect_indices[index][-1]

            for j in range(n_right_words):
                data_set['word_embeddings'].append(x_right_part[index][j].tolist())
                data_set['hidden_states'].append(tr_layer_information['right_hidden_state'][0][0][j].tolist())

                if "LCR_Rot_hop_model" in self.nl_model.config.name_of_model:

                    for i in range(self.nl_model.config.n_iterations_hop):
                        data_set['weighted_hidden_states_' + str(i)].append(
                            tr_layer_information['weighted_right_hidden_state_' + str(i)][0][0][j].tolist())
                else:
                    data_set['weighted_hidden_states'].append(
                        tr_layer_information['weighted_right_hidden_state'][0][0][j].tolist())

                data_set['word_sentiments'].append(word_sentiments[index][last_aspect_index + 1 + j])
                index_of_one = word_sentiments[index][last_aspect_index + 1 + j].index(1)
                statistics['word_sentiments_count'][index_of_one] += 1

                data_set['aspect_sentiments'].append(aspect_sentiments[index][last_aspect_index + 1 + j])
                index_of_one = aspect_sentiments[index][last_aspect_index + 1 + j].index(1)
                statistics['aspect_sentiments_count'][index_of_one] += 1

                data_set['relations'].append(relations[index][last_aspect_index + 1 + j])
                index_of_one = relations[index][last_aspect_index + 1 + j].index(1)
                statistics['relations_count'][index_of_one] += 1

                data_set['pos_tags'].append(pos_tags[index][last_aspect_index + 1 + j])
                index_of_one = pos_tags[index][last_aspect_index + 1 + j].index(1)
                statistics['pos_tags_count'][index_of_one] += 1

                data_set['mentions'].append(mentions[index][last_aspect_index + 1 + j])
                index_of_one = mentions[index][last_aspect_index + 1 + j].index(1)
                statistics['mentions_count'][index_of_one] += 1

        # set the numpy array to a list s.t. it can be configured to json
        statistics['word_sentiments_count'] = statistics['word_sentiments_count'].tolist()
        statistics['aspect_sentiments_count'] = statistics['aspect_sentiments_count'].tolist()
        statistics['relations_count'] = statistics['relations_count'].tolist()
        statistics['pos_tags_count'] = statistics['pos_tags_count'].tolist()
        statistics['mentions_count'] = statistics['mentions_count'].tolist()

        # Append statistics to the data set
        data_set['statistics'] = statistics

        return data_set

    def fit_diagnostic_classifiers(self):

        print("I am fitting the diagnostic classifiers")

        with open(self.nl_model.config.tr_file_of_hid_correct_pred, 'r') as file:
            for line in file:
                training_set = json.loads(line)

        for interest, value in self.diagnostic_classifiers.items():

            if value:

                config = self.configurations[interest]
                for i in range(config.cross_validation_rounds):
                    print("cross validation round ", i)
                    self.fit_diagnostic_classifier(config, training_set, interest, i)

    def fit_diagnostic_classifier(self, config, x, interest, cross_validation_round):

        counts = np.array(x['statistics'][interest+'_count'])
        print("count ", counts)
        arg_max = np.argmax(counts)
        items = np.delete(np.arange(counts.shape[0]), arg_max)
        mean_count = int(np.floor(np.mean(counts[items])))

        y = np.array(x[interest])
        random_indices = []

        for i in range(counts.shape[0]):

            a_range = np.arange(y.shape[0], dtype=int)
            y_arg_max = np.argmax(y, axis=1)
            selected_items = a_range[y_arg_max == i]
            count_indices = selected_items.shape[0]
            random_indices.append(
                np.random.choice(selected_items, min(count_indices, mean_count), replace=False).tolist())

        random_indices = [a for b in random_indices for a in b]

        file = config.get_file_of_model_savings(self.nl_model.config.name_of_model, 'embeddings_',
                                                cross_validation_round)

        if not os.path.isfile(file+".index"):
            config.classifier_embeddings.fit(np.array(x['word_embeddings'])[random_indices], y[random_indices],
                                             self.nl_model, file, '_e')

        file = config.get_file_of_model_savings(self.nl_model.config.name_of_model, 'states_', cross_validation_round)

        if not os.path.isfile(file+".index"):
            config.classifier_states.fit(
                np.array(x['hidden_states'])[random_indices], y[random_indices], self.nl_model, file, '_s')

        if "LCR_Rot_hop_model" in self.nl_model.config.name_of_model:

            for i in range(self.nl_model.config.n_iterations_hop):

                file = config.get_file_of_model_savings(self.nl_model.config.name_of_model, 'weighted_' + str(i) + "_",
                                                        cross_validation_round)
                if not os.path.isfile(file+".index"):
                    config.classifier_states.fit(
                        np.array(x['weighted_hidden_states_' + str(i)])[random_indices], y[random_indices],
                        self.nl_model, file, '_w' + str(i))
        else:
            file = config.get_file_of_model_savings(self.nl_model.config.name_of_model, 'weighted_',
                                                    cross_validation_round)

            if not os.path.isfile(file+".index"):
                config.classifier_states.fit(np.array(x['weighted_hidden_states'])[random_indices],
                                             y[random_indices], self.nl_model, file, '_w')

    def predict_diagnostic_classifiers(self):

        print("I am predicting the test set using the diagnostic classifiers")

        with open(self.nl_model.config.tr_file_of_hid_correct_pred, 'r') as file:
            for line in file:
                correct_training_set = json.loads(line)

        with open(self.nl_model.config.te_file_of_hid_correct_pred, 'r') as file:
            for line in file:
                correct_test_set = json.loads(line)

        with open(self.nl_model.config.te_file_of_hid_incorrect_pred, 'r') as file:
            for line in file:
                incorrect_test_set = json.loads(line)

        for interest, value in self.diagnostic_classifiers.items():
            if value:
                config = self.configurations[interest]
                if not os.path.isfile(
                        config.get_file_of_results(self.nl_model.config.name_of_model, 'tr_correct_pred')):
                    self.predict_diagnostic_classifier(config, correct_training_set, interest, 'tr_correct_pred')
                if not os.path.isfile(
                        config.get_file_of_results(self.nl_model.config.name_of_model, 'te_correct_pred')):
                    self.predict_diagnostic_classifier(config, correct_test_set, interest, 'te_correct_pred')
                if not os.path.isfile(
                        config.get_file_of_results(self.nl_model.config.name_of_model, 'te_incorrect_pred')):
                    self.predict_diagnostic_classifier(config, incorrect_test_set, interest, 'te_incorrect_pred')

    def predict_diagnostic_classifier(self, config, x, interest, correctness):

        y = np.array(x[interest])
        arg_max_y = np.argmax(y, axis=1)

        n_options = y.shape[1]
        counts = x['statistics'][interest+'_count']
        sum_of_counts = int(sum(x['statistics'][interest+'_count']))
        counts.append(sum_of_counts)

        accuracies = {
            'true_values': counts,
            'n_correct_embeddings': np.zeros((config.cross_validation_rounds, y.shape[1] + 1), dtype=int),
            'n_correct_hidden_states': np.zeros((config.cross_validation_rounds, y.shape[1] + 1), dtype=int)
        }

        if "LCR_Rot_hop_model" in self.nl_model.config.name_of_model:
            for i in range(self.nl_model.config.n_iterations_hop):
                accuracies['n_correct_weighted_hidden_states_' + str(i)] = np.zeros((config.cross_validation_rounds,
                                                                                     y.shape[1] + 1), dtype=int)
        else:
            accuracies['n_correct_weighted_hidden_states'] = np.zeros((config.cross_validation_rounds, y.shape[1] + 1),
                                                                      dtype=int)
        for i in range(config.cross_validation_rounds):

            print("cross validation round ", i)

            file = config.get_file_of_model_savings(self.nl_model.config.name_of_model, 'embeddings_', i)

            if os.path.isfile(file + ".index"):

                pred_word_embeddings = np.array(config.classifier_embeddings.predict(np.array(x['word_embeddings']),
                                                                                     file, '_e'))[0]
                arg_max_pred = np.argmax(pred_word_embeddings, axis=1)

                for index in range(arg_max_pred.shape[0]):
                    if arg_max_pred[index] == arg_max_y[index]:
                        accuracies['n_correct_embeddings'][i][arg_max_pred[index]] += 1
                        accuracies['n_correct_embeddings'][i][n_options] += 1

            file = config.get_file_of_model_savings(self.nl_model.config.name_of_model, 'states_', i)

            if os.path.isfile(file+".index"):

                pred_hidden_states = np.array(config.classifier_states.predict(np.array(x['hidden_states']),
                                                                               file, '_s'))[0]
                arg_max_pred = np.argmax(pred_hidden_states, axis=1)

                for index in range(arg_max_pred.shape[0]):
                    if arg_max_pred[index] == arg_max_y[index]:
                        accuracies['n_correct_hidden_states'][i][arg_max_pred[index]] += 1
                        accuracies['n_correct_hidden_states'][i][n_options] += 1

            if "LCR_Rot_hop_model" in self.nl_model.config.name_of_model:

                for j in range(self.nl_model.config.n_iterations_hop):

                    file = config.get_file_of_model_savings(
                        self.nl_model.config.name_of_model, 'weighted_' + str(j) + "_", i)

                    if os.path.isfile(file+".index"):
                        pred_weighted_hidden_states = np.array(config.classifier_states.predict(
                            np.array(x['weighted_hidden_states_' + str(j)]), file, '_w' + str(j)))[0]
                        arg_max_pred = np.argmax(pred_weighted_hidden_states, axis=1)

                        for index in range(arg_max_pred.shape[0]):
                            if arg_max_pred[index] == arg_max_y[index]:
                                accuracies['n_correct_weighted_hidden_states_' + str(j)][i][arg_max_pred[index]] += 1
                                accuracies['n_correct_weighted_hidden_states_' + str(j)][i][n_options] += 1
            else:
                file = config.get_file_of_model_savings(self.nl_model.config.name_of_model, 'weighted_', i)

                if os.path.isfile(file+".index"):
                    pred_weighted_hidden_states = np.array(config.classifier_states.predict(
                        np.array(x['weighted_hidden_states']), file, '_w'))[0]
                    arg_max_pred = np.argmax(pred_weighted_hidden_states, axis=1)

                    for index in range(arg_max_pred.shape[0]):
                        if arg_max_pred[index] == arg_max_y[index]:
                            accuracies['n_correct_weighted_hidden_states'][i][arg_max_pred[index]] += 1
                            accuracies['n_correct_weighted_hidden_states'][i][n_options] += 1

        accuracies['acc_mean_correct_embeddings'] = np.mean(
            np.divide(accuracies.get('n_correct_embeddings'), accuracies.get('true_values')), axis=0).tolist()
        accuracies['acc_std_correct_embeddings'] = np.std(
            np.divide(accuracies.get('n_correct_embeddings'), accuracies.get('true_values')), axis=0).tolist()
        accuracies['n_correct_embeddings'] = accuracies.get('n_correct_embeddings').tolist()
        accuracies['acc_mean_correct_hidden_states'] = np.mean(
            np.divide(accuracies.get('n_correct_hidden_states'), accuracies.get('true_values')), axis=0).tolist()
        accuracies['acc_std_correct_hidden_states'] = np.std(
            np.divide(accuracies.get('n_correct_hidden_states'), accuracies.get('true_values')), axis=0).tolist()
        accuracies['n_correct_hidden_states'] = accuracies.get('n_correct_hidden_states').tolist()

        if "LCR_Rot_hop_model" in self.nl_model.config.name_of_model:
            for j in range(self.nl_model.config.n_iterations_hop):
                accuracies['acc_mean_correct_weighted_hidden_states_' + str(j)] = np.mean(
                    np.divide(accuracies.get('n_correct_weighted_hidden_states_' + str(j)),
                              accuracies.get('true_values')), axis=0).tolist()
                accuracies['acc_std_correct_weighted_hidden_states_' + str(j)] = np.std(
                    np.divide(accuracies.get('n_correct_weighted_hidden_states_' + str(j)),
                              accuracies.get('true_values')), axis=0).tolist()
                accuracies['n_correct_weighted_hidden_states_' + str(j)] = \
                    accuracies.get('n_correct_weighted_hidden_states_' + str(j)).tolist()
        else:
            accuracies['acc_mean_correct_weighted_hidden_states'] = np.mean(np.divide(
                accuracies.get('n_correct_weighted_hidden_states'), accuracies.get('true_values')), axis=0).tolist()
            accuracies['acc_std_correct_weighted_hidden_states'] = np.std(np.divide(
                accuracies.get('n_correct_weighted_hidden_states'), accuracies.get('true_values')), axis=0).tolist()
            accuracies['n_correct_weighted_hidden_states'] = accuracies.get('n_correct_weighted_hidden_states').tolist()

        file = config.get_file_of_results(self.nl_model.config.name_of_model, correctness)
        with open(file, 'w') as outfile:
            json.dump(accuracies, outfile, ensure_ascii=False, indent=0)
