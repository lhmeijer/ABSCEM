import numpy as np
import json
import os
from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierAspectSentimentConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierWordSentimentConfig, DiagnosticClassifierMentionConfig, \
    DiagnosticClassifierFullAspectSentimentConfig


class DiagnosticClassifier:

    def __init__(self, neural_language_model, diagnostic_classifiers):
        self.neural_language_model = neural_language_model
        self.diagnostic_classifiers = diagnostic_classifiers

        # check whether the hidden layers of the neural language model are already saved.
        if not os.path.isfile(neural_language_model.config.tr_file_of_hid_corr_pred) or not \
                os.path.isfile(neural_language_model.config.te_file_of_hid_corr_pred):
            self.create_file_hidden_layers()

    def create_file_hidden_layers(self):

        with open(self.neural_language_model.config.file_of_indices, 'r') as file:
            for line in file:
                indices = json.loads(line)

        tr_corr_pred = indices['tr_correct_predicted']
        tr_wrong_pred = indices['tr_wrong_predicted']

        te_corr_pred = indices['te_correct_predicted']
        te_wrong_pred = indices['te_wrong_predicted']

        x_training = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)

        tr_word_sentiments = np.array(self.neural_language_model.internal_data_loader.word_sentiments_in_training)
        tr_aspect_sentiments = np.array(self.neural_language_model.internal_data_loader.aspect_sentiments_in_training)
        tr_relations = np.array(self.neural_language_model.internal_data_loader.word_relations_training)
        tr_pos_tags = np.array(self.neural_language_model.internal_data_loader.part_of_speech_training)
        tr_mentions = np.array(self.neural_language_model.internal_data_loader.mentions_in_training)

        tr_corr_pred_fv = self.create_feature_set(x_training[tr_corr_pred], train_aspects[tr_corr_pred],
                                                  tr_word_sentiments[tr_corr_pred], tr_aspect_sentiments[tr_corr_pred],
                                                  tr_relations[tr_corr_pred], tr_pos_tags[tr_corr_pred],
                                                  tr_mentions[tr_corr_pred])

        with open(self.neural_language_model.config.tr_file_of_hid_corr_pred, 'w') as outfile:
            json.dump(tr_corr_pred_fv, outfile, ensure_ascii=False)

        tr_wrong_pred_fv = self.create_feature_set(x_training[tr_wrong_pred], train_aspects[tr_wrong_pred],
                                                   tr_word_sentiments[tr_wrong_pred],
                                                   tr_aspect_sentiments[tr_wrong_pred], tr_relations[tr_wrong_pred],
                                                   tr_pos_tags[tr_wrong_pred], tr_mentions[tr_wrong_pred])

        with open(self.neural_language_model.config.tr_file_of_hid_wrong_pred, 'w') as outfile:
            json.dump(tr_wrong_pred_fv, outfile, ensure_ascii=False)

        x_test = np.array(self.neural_language_model.internal_data_loader.word_embeddings_test_all)
        test_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_test)

        te_word_sentiments = np.array(self.neural_language_model.internal_data_loader.word_sentiments_in_test)
        te_aspect_sentiments = np.array(self.neural_language_model.internal_data_loader.aspect_sentiments_in_test)
        te_relations = np.array(self.neural_language_model.internal_data_loader.word_relations_test)
        te_pos_tags = np.array(self.neural_language_model.internal_data_loader.part_of_speech_test)
        te_mentions = np.array(self.neural_language_model.internal_data_loader.mentions_in_test)

        te_corr_pred_fv = self.create_feature_set(x_test[te_corr_pred], test_aspects[te_corr_pred],
                                                  te_word_sentiments[te_corr_pred],
                                                  te_aspect_sentiments[te_corr_pred], te_relations[te_corr_pred],
                                                  te_pos_tags[te_corr_pred], te_mentions[te_corr_pred])

        with open(self.neural_language_model.config.te_file_of_hid_corr_pred, 'w') as outfile:
            json.dump(te_corr_pred_fv, outfile, ensure_ascii=False)

        te_wrong_pred_fv = self.create_feature_set(x_test[te_wrong_pred], test_aspects[te_wrong_pred],
                                                   te_word_sentiments[te_wrong_pred],
                                                   te_aspect_sentiments[te_wrong_pred], te_relations[te_wrong_pred],
                                                   te_pos_tags[te_wrong_pred], te_mentions[te_wrong_pred])

        with open(self.neural_language_model.config.te_file_of_hid_wrong_pred, 'w') as outfile:
            json.dump(te_wrong_pred_fv, outfile, ensure_ascii=False)

    def create_feature_set(self, x, aspects, word_sentiments, aspect_sentiments, relations, pos_tags, mentions):

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.internal_data_loader.split_embeddings(
                x, aspects, self.neural_language_model.config.max_sentence_length,
                self.neural_language_model.config.max_target_length)

        left_word_embeddings = []
        right_word_embeddings = []
        left_hidden_states = []
        right_hidden_states = []

        weighted_hidden_state = {}

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):
                weighted_hidden_state['weighted_left_hidden_state_' + str(i)] = []
                weighted_hidden_state['weighted_right_hidden_state_' + str(i)] = []
        else:
            weighted_hidden_state['weighted_left_hidden_state'] = []
            weighted_hidden_state['weighted_right_hidden_state'] = []

        word_sentiments_per_word_left = []
        word_sentiments_count_left = np.zeros(len(word_sentiments[0][0]), dtype=int)
        word_sentiments_per_word_right = []
        word_sentiments_count_right = np.zeros(len(word_sentiments[0][0]), dtype=int)

        aspect_sentiments_per_word_left = []
        aspect_sentiments_count_left = np.zeros(len(aspect_sentiments[0][0]), dtype=int)
        aspect_sentiments_per_word_right = []
        aspect_sentiments_count_right = np.zeros(len(aspect_sentiments[0][0]), dtype=int)

        relations_per_word_left = []
        relations_count_left = np.zeros(len(relations[0][0]), dtype=int)
        relations_per_word_right = []
        relations_count_right = np.zeros(len(relations[0][0]), dtype=int)

        pos_tags_per_word_left = []
        pos_tags_count_left = np.zeros(len(pos_tags[0][0]), dtype=int)
        pos_tags_per_word_right = []
        pos_tags_count_right = np.zeros(len(pos_tags[0][0]), dtype=int)

        mentions_per_word_left = []
        mentions_count_left = np.zeros(len(mentions[0][0]), dtype=int)
        mentions_per_word_right = []
        mentions_count_right = np.zeros(len(mentions[0][0]), dtype=int)

        for index in range(x.shape[0]):

            print("index ", index)

            tr_pred, tr_layer_information = self.neural_language_model.predict(np.array([x_left_part[index]]),
                                                                               np.array([x_target_part[index]]),
                                                                               np.array([x_right_part[index]]),
                                                                               np.array([x_left_sen_len[index]]),
                                                                               np.array([x_tar_len[index]]),
                                                                               np.array([x_right_sen_len[index]]))
            n_left_words = x_left_sen_len[index]

            for j in range(n_left_words):
                left_word_embeddings.append(x_left_part[index][j].tolist())
                left_hidden_states.append(tr_layer_information['left_hidden_state'][0][0][j].tolist())

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        weighted_hidden_state['weighted_left_hidden_state_' + str(i)].append(
                            tr_layer_information['weighted_left_hidden_state_' + str(i)][0][0][j].tolist())
                else:
                    weighted_hidden_state['weighted_left_hidden_state'].append(
                        tr_layer_information['weighted_left_hidden_state'][0][0][j].tolist())

                word_sentiments_per_word_left.append(word_sentiments[index][j])
                index_of_one = word_sentiments[index][j].index(1)
                word_sentiments_count_left[index_of_one] += 1

                aspect_sentiments_per_word_left.append(aspect_sentiments[index][j])
                index_of_one = aspect_sentiments[index][j].index(1)
                aspect_sentiments_count_left[index_of_one] += 1

                relations_per_word_left.append(relations[index][j])
                index_of_one = relations[index][j].index(1)
                relations_count_left[index_of_one] += 1

                pos_tags_per_word_left.append(pos_tags[index][j])
                index_of_one = pos_tags[index][j].index(1)
                pos_tags_count_left[index_of_one] += 1

                mentions_per_word_left.append(mentions[index][j])
                index_of_one = mentions[index][j].index(1)
                mentions_count_left[index_of_one] += 1

            n_right_words = x_right_sen_len[index]

            end_index = aspects[index][-1]

            for j in range(n_right_words):
                right_word_embeddings.append(x_right_part[index][j].tolist())
                right_hidden_states.append(tr_layer_information['right_hidden_state'][0][0][j].tolist())

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        weighted_hidden_state['weighted_right_hidden_state_' + str(i)].append(
                            tr_layer_information['weighted_right_hidden_state_' + str(i)][0][0][j].tolist())
                else:
                    weighted_hidden_state['weighted_right_hidden_state'].append(
                        tr_layer_information['weighted_right_hidden_state'][0][0][j].tolist())

                word_sentiments_per_word_right.append(word_sentiments[index][end_index + 1 + j])
                index_of_one = word_sentiments[index][end_index + 1 + j].index(1)
                word_sentiments_count_right[index_of_one] += 1

                aspect_sentiments_per_word_right.append(aspect_sentiments[index][end_index + 1 + j])
                index_of_one = aspect_sentiments[index][end_index + 1 + j].index(1)
                aspect_sentiments_count_right[index_of_one] += 1

                relations_per_word_right.append(relations[index][end_index + 1 + j])
                index_of_one = relations[index][end_index + 1 + j].index(1)
                relations_count_right[index_of_one] += 1

                pos_tags_per_word_right.append(pos_tags[index][end_index + 1 + j])
                index_of_one = pos_tags[index][end_index + 1 + j].index(1)
                pos_tags_count_right[index_of_one] += 1

                mentions_per_word_right.append(mentions[index][end_index + 1 + j])
                index_of_one = mentions[index][end_index + 1 + j].index(1)
                mentions_count_right[index_of_one] += 1

        feature_values = {
            'left_word_embedding': left_word_embeddings,
            'right_word_embedding': right_word_embeddings,
            'left_hidden_state': left_hidden_states,
            'right_hidden_state': right_hidden_states,
            'weighted_hidden_state': weighted_hidden_state,
            'word_sentiments_per_word_left': word_sentiments_per_word_left,
            'word_sentiments_count_left': word_sentiments_count_left.tolist(),
            'word_sentiments_per_word_right': word_sentiments_per_word_right,
            'word_sentiments_count_right': word_sentiments_count_right.tolist(),
            'aspect_sentiments_per_word_left': aspect_sentiments_per_word_left,
            'aspect_sentiments_count_left': aspect_sentiments_count_left.tolist(),
            'aspect_sentiments_per_word_right': aspect_sentiments_per_word_right,
            'aspect_sentiments_count_right': aspect_sentiments_count_right.tolist(),
            'relations_per_word_left': relations_per_word_left,
            'relations_count_left': relations_count_left.tolist(),
            'relations_per_word_right': relations_per_word_right,
            'relations_count_right': relations_count_right.tolist(),
            'pos_tags_per_word_left': pos_tags_per_word_left,
            'pos_tags_count_left': pos_tags_count_left.tolist(),
            'pos_tags_per_word_right': pos_tags_per_word_right,
            'pos_tags_count_right': pos_tags_count_right.tolist(),
            'mentions_per_word_left': mentions_per_word_left,
            'mentions_count_left': mentions_count_left.tolist(),
            'mentions_per_word_right': mentions_per_word_right,
            'mentions_count_right': mentions_count_right.tolist()
        }

        return feature_values

    def run(self):

        print('I am diagnostic classifier.')

        # Open the files, which consist the hidden layers of the neural language model
        # The hidden layers of the correct predicted training data
        with open(self.neural_language_model.config.tr_file_of_hid_corr_pred, 'r') as file:
            for line in file:
                tr_corr_pred_fv = json.loads(line)
        with open(self.neural_language_model.config.tr_file_of_hid_wrong_pred, 'r') as file:
            for line in file:
                tr_wrong_pred_fv = json.loads(line)

        with open(self.neural_language_model.config.te_file_of_hid_corr_pred, 'r') as file:
            for line in file:
                te_corr_pred_fv = json.loads(line)
        with open(self.neural_language_model.config.te_file_of_hid_wrong_pred, 'r') as file:
            for line in file:
                te_wrong_pred_fv = json.loads(line)

        configurations = {
            'word_sentiments': DiagnosticClassifierWordSentimentConfig,
            'aspect_sentiments': DiagnosticClassifierAspectSentimentConfig,
            'relations': DiagnosticClassifierRelationConfig,
            'pos_tags': DiagnosticClassifierPOSConfig,
            'mentions': DiagnosticClassifierMentionConfig,
            'full_aspect_sentiment': DiagnosticClassifierFullAspectSentimentConfig
        }

        # diagnostic classifier to test whether the interests are able to predict
        for interest, value in self.diagnostic_classifiers.items():

            if interest == 'full_aspect_sentiment' and value:

                with open(self.neural_language_model.config.file_of_indices, 'r') as file:
                    for line in file:
                        indices = json.loads(line)

                tr_corr_pred = indices['tr_correct_predicted']
                tr_wrong_pred = indices['tr_wrong_predicted']

                te_corr_pred = indices['te_correct_predicted']
                te_wrong_pred = indices['te_wrong_predicted']

                train_x = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
                train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
                train_y = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)

                test_x = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
                test_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
                test_y = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_test)

                self.fit_full_context_diagnostic_classifier(configurations[interest], train_x[tr_corr_pred],
                                                            train_aspects[tr_corr_pred], train_y[tr_corr_pred])
                self.predict_full_context_diagnostic_classifier(configurations[interest],
                                                                train_x[tr_corr_pred], train_aspects[tr_corr_pred],
                                                                train_y[tr_corr_pred], 'tr_corr_pred')
                self.predict_full_context_diagnostic_classifier(configurations[interest],
                                                                train_x[tr_wrong_pred], train_aspects[tr_wrong_pred],
                                                                train_y[tr_wrong_pred], 'tr_wrong_pred')
                self.predict_full_context_diagnostic_classifier(configurations[interest], test_x[te_corr_pred],
                                                                test_aspects[te_corr_pred], test_y[te_corr_pred],
                                                                'te_corr_pred')
                self.predict_full_context_diagnostic_classifier(configurations[interest],
                                                                test_x[te_wrong_pred], test_aspects[te_wrong_pred],
                                                                test_y[te_wrong_pred], 'te_wrong_pred')
            elif value:
                self.fit_diagnostic_classifier(configurations[interest], tr_corr_pred_fv, interest)

                self.predict_diagnostic_classifier(configurations[interest], tr_corr_pred_fv, interest, 'tr_corr_pred')
                self.predict_diagnostic_classifier(configurations[interest], tr_wrong_pred_fv, interest,
                                                   'tr_wrong_pred')

                self.predict_diagnostic_classifier(configurations[interest], te_corr_pred_fv, interest, 'te_corr_pred')
                self.predict_diagnostic_classifier(configurations[interest], te_wrong_pred_fv, interest,
                                                   'te_wrong_pred')

    def fit_diagnostic_classifier(self, diagnostic_config, feature_values, interest):

        left_counts = np.array(feature_values[interest+'_count_left'])
        left_arg_max = np.argmax(left_counts)
        left_items = np.delete(np.arange(left_counts.shape[0]), left_arg_max)
        left_mean_count = int(np.floor(np.mean(left_counts[left_items])))

        right_counts = np.array(feature_values[interest + '_count_right'])
        right_arg_max = np.argmax(right_counts)
        right_items = np.delete(np.arange(right_counts.shape[0]), right_arg_max)
        right_mean_count = int(np.floor(np.mean(right_counts[right_items])))

        tr_left_y = np.array(feature_values[interest+'_per_word_left'])
        left_random_indices = []
        tr_right_y = np.array(feature_values[interest + '_per_word_right'])
        right_random_indices = []

        for i in range(left_counts.shape[0]):

            left_range = np.arange(tr_left_y.shape[0], dtype=int)
            tr_left_y_max = np.argmax(tr_left_y, axis=1)
            left_items = left_range[tr_left_y_max == i]
            count_indices = left_items.shape[0]
            left_random_indices.append(np.random.choice(left_items, min(count_indices, left_mean_count),
                                                        replace=False).tolist())

            right_range = np.arange(tr_right_y.shape[0], dtype=int)
            tr_right_y_max = np.argmax(tr_right_y, axis=1)
            right_items = right_range[tr_right_y_max == i]
            count_indices = right_items.shape[0]
            right_random_indices.append(np.random.choice(right_items, min(count_indices, right_mean_count),
                                                         replace=False).tolist())

        left_random_indices = [y for x in left_random_indices for y in x]
        right_random_indices = [y for x in right_random_indices for y in x]

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_embeddings_')

        if not os.path.isfile(file+".index"):
            diagnostic_config.classifier_embeddings.fit(
                np.array(feature_values['left_word_embedding'])[left_random_indices], tr_left_y[left_random_indices],
                self.neural_language_model, file, '_le')

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_embeddings_')

        if not os.path.isfile(file+".index"):
            diagnostic_config.classifier_embeddings.fit(
                np.array(feature_values['right_word_embedding'])[right_random_indices],
                tr_right_y[right_random_indices], self.neural_language_model, file, '_re')

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_states_')

        if not os.path.isfile(file+".index"):
            diagnostic_config.classifier_states.fit(
                np.array(feature_values['left_hidden_state'])[left_random_indices], tr_left_y[left_random_indices],
                self.neural_language_model, file, '_ls')

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_states_')

        if not os.path.isfile(file+".index"):
            diagnostic_config.classifier_states.fit(
                np.array(feature_values['right_hidden_state'])[right_random_indices], tr_right_y[right_random_indices],
                self.neural_language_model, file, '_rs')

        dict_weighted_hidden_states = feature_values['weighted_hidden_state']

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):

                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")

                if not os.path.isfile(file+".index"):
                    diagnostic_config.classifier_states.fit(
                        np.array(dict_weighted_hidden_states['weighted_left_hidden_state_' + str(i)])
                        [left_random_indices], tr_left_y[left_random_indices], self.neural_language_model, file,
                        '_lw' + str(i))

                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")

                if not os.path.isfile(file+".index"):
                    diagnostic_config.classifier_states.fit(
                        np.array(dict_weighted_hidden_states['weighted_right_hidden_state_' + str(i)])
                        [right_random_indices], tr_right_y[right_random_indices], self.neural_language_model, file,
                        '_rw' + str(i))
        else:
            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_weighted_')

            if not os.path.isfile(file+".index"):
                diagnostic_config.classifier_states.fit(
                np.array(dict_weighted_hidden_states['weighted_left_hidden_state'])[left_random_indices],
                tr_left_y[left_random_indices], self.neural_language_model, file, '_lw')

            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_weighted_')

            if not os.path.isfile(file+".index"):
                diagnostic_config.classifier_states.fit(
                    np.array(dict_weighted_hidden_states['weighted_right_hidden_state'])[right_random_indices],
                    tr_right_y[right_random_indices], self.neural_language_model, file, '_rw')

    def predict_diagnostic_classifier(self, diagnostic_config, feature_values, interest, name):

        left_counts = feature_values[interest+'_count_left']
        right_counts = feature_values[interest + '_count_right']

        left_y = np.array(feature_values[interest+'_per_word_left'])
        argmax_left_y = np.argmax(left_y, axis=1)
        right_y = np.array(feature_values[interest + '_per_word_right'])
        argmax_right_y = np.argmax(right_y, axis=1)

        acc_left_word_embeddings = [0] * left_y.shape[1]
        acc_right_word_embeddings = [0] * right_y.shape[1]
        acc_left_hidden_states =[0] * left_y.shape[1]
        acc_right_hidden_states = [0] * right_y.shape[1]
        acc_weighted_hidden_state = {}

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
            for i in range(self.neural_language_model.config.n_iterations_hop):
                acc_weighted_hidden_state['acc_weighted_left_hidden_state_' + str(i)] = [0] * left_y.shape[1]
                acc_weighted_hidden_state['acc_weighted_right_hidden_state_' + str(i)] = [0] * right_y.shape[1]
        else:
            acc_weighted_hidden_state['acc_weighted_left_hidden_state'] = [0] * left_y.shape[1]
            acc_weighted_hidden_state['acc_weighted_right_hidden_state'] = [0] * right_y.shape[1]

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_embeddings_')

        if os.path.isfile(file+".index"):

            pred_left_word_embeddings =  np.array(diagnostic_config.classifier_embeddings.predict(
                np.array(feature_values['left_word_embedding']), file, '_le'))[0]
            arg_max_pred = np.argmax(pred_left_word_embeddings, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_left_y[index]:
                    acc_left_word_embeddings[arg_max_pred[index]] += 1
            print("acc_left_word_embeddings ", acc_left_word_embeddings)

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_embeddings_')

        if os.path.isfile(file+".index"):
            pred_right_word_embeddings = np.array(diagnostic_config.classifier_embeddings.predict(
                np.array(feature_values['right_word_embedding']), file, '_re'))[0]
            arg_max_pred = np.argmax(pred_right_word_embeddings, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_right_y[index]:
                    acc_right_word_embeddings[arg_max_pred[index]] += 1
            print("acc_right_word_embeddings ", acc_right_word_embeddings)

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_states_')

        if os.path.isfile(file+".index"):
            pred_left_hidden_states = np.array(diagnostic_config.classifier_states.predict(
                np.array(feature_values['left_hidden_state']), file, '_ls'))[0]
            arg_max_pred = np.argmax(pred_left_hidden_states, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_left_y[index]:
                    acc_left_hidden_states[arg_max_pred[index]] += 1
            print("acc_left_hidden_states ", acc_left_hidden_states)

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_states_')

        if os.path.isfile(file+".index"):
            pred_right_hidden_states = np.array(diagnostic_config.classifier_states.predict(
                np.array(feature_values['right_hidden_state']), file, '_rs'))[0]
            arg_max_pred = np.argmax(pred_right_hidden_states, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_right_y[index]:
                    acc_right_hidden_states[arg_max_pred[index]] += 1
            print("acc_right_hidden_states ", acc_right_hidden_states)

        dict_weighted_hidden_states = feature_values['weighted_hidden_state']

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):

                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")

                if os.path.isfile(file+".index"):
                    pred_weighted_left_hidden_state = np.array(diagnostic_config.classifier_states.predict(
                        np.array(dict_weighted_hidden_states['weighted_left_hidden_state_' + str(i)]), file,
                        '_lw' + str(i)))[0]
                    arg_max_pred = np.argmax(pred_weighted_left_hidden_state, axis=1)

                    for index in range(arg_max_pred.shape[0]):
                        if arg_max_pred[index] == argmax_left_y[index]:
                            acc_weighted_hidden_state['acc_weighted_left_hidden_state_' + str(i)][arg_max_pred[index]] \
                                += 1

                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")

                if os.path.isfile(file+".index"):
                    pred_weighted_right_hidden_state = np.array(diagnostic_config.classifier_states.predict(
                        np.array(dict_weighted_hidden_states['weighted_right_hidden_state_' + str(i)]), file,
                        '_rw' + str(i)))[0]
                    arg_max_pred = np.argmax(pred_weighted_right_hidden_state, axis=1)

                    for index in range(arg_max_pred.shape[0]):
                        if arg_max_pred[index] == argmax_right_y[index]:
                            acc_weighted_hidden_state['acc_weighted_right_hidden_state_' + str(i)][arg_max_pred[index]] += 1
        else:
            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_weighted_')

            if os.path.isfile(file+".index"):
                pred_weighted_left_hidden_state = np.array(diagnostic_config.classifier_states.predict(
                    np.array(dict_weighted_hidden_states['weighted_left_hidden_state']), file, '_lw'))[0]
                arg_max_pred = np.argmax(pred_weighted_left_hidden_state, axis=1)

                for index in range(arg_max_pred.shape[0]):
                    if arg_max_pred[index] == argmax_left_y[index]:
                        acc_weighted_hidden_state['acc_weighted_left_hidden_state'][arg_max_pred[index]] += 1

            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_weighted_')

            if os.path.isfile(file+".index"):
                pred_weighted_right_hidden_state = np.array(diagnostic_config.classifier_states.predict(
                    np.array(dict_weighted_hidden_states['weighted_right_hidden_state']), file, '_rw'))[0]
                arg_max_pred = np.argmax(pred_weighted_right_hidden_state, axis=1)

                for index in range(arg_max_pred.shape[0]):
                    if arg_max_pred[index] == argmax_right_y[index]:
                        acc_weighted_hidden_state['acc_weighted_right_hidden_state'][arg_max_pred[index]] += 1

        results = {
            'name': diagnostic_config.name_of_model,
            'neural_language_model': self.neural_language_model.config.name_of_model,
            'n_true_left': left_counts,
            'n_true_right': right_counts,
            'n_acc_left_word_embeddings': acc_left_word_embeddings,
            'n_acc_right_word_embeddings': acc_right_word_embeddings,
            'n_acc_left_hidden_states': acc_left_hidden_states,
            'n_acc_right_hidden_states': acc_right_hidden_states,
            'acc_weighted_hidden_state': acc_weighted_hidden_state
        }

        file = diagnostic_config.get_file_of_results(self.neural_language_model.config.name_of_model, name)
        with open(file, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=0)

    def fit_full_context_diagnostic_classifier(self, diagnostic_config, x, aspects, y):

        true_y = np.zeros(y.shape[1], dtype=int)
        argmax_y = np.argmax(y, axis=1)
        for arg_max in argmax_y:
            true_y[arg_max] += 1

        items = np.delete(np.arange(y.shape[1]), np.argmax(true_y))
        mean = int(np.floor(np.mean(true_y[items])))
        random_indices = []

        for i in range(y.shape[1]):

            item_range = np.arange(y.shape[0], dtype=int)
            selected_indices = item_range[argmax_y == i]
            n_indices = selected_indices.shape[0]
            random_indices.append(np.random.choice(selected_indices, min(n_indices, mean), replace=False).tolist())

        random_indices = [y for x in random_indices for y in x]
        print("random_indices ", random_indices)
        print("len(random_indices) ", len(random_indices))

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.internal_data_loader.split_embeddings(
                x, aspects, self.neural_language_model.config.max_sentence_length,
                self.neural_language_model.config.max_target_length)

        _, information_layer = self.neural_language_model.predict(np.array(x_left_part), np.array(x_target_part),
                                                                  np.array(x_right_part), np.array(x_left_sen_len),
                                                                  np.array(x_tar_len), np.array(x_right_sen_len))

        mcl = diagnostic_config.max_context_length
        # dimension = self.neural_language_model.config.embedding_dimension * diagnostic_config.max_context_length
        dimension = self.neural_language_model.config.embedding_dimension
        n_sentences = x_left_part.shape[0]

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_embeddings_')

        if not os.path.isfile(file+".index"):
            x_train = np.zeros((n_sentences, dimension), dtype=float)
            e = np.array(x_left_part)

            for j in range(n_sentences):
                max_length = x_left_sen_len[j]
                x_train[j] = np.mean(e[j, 0:max_length, :], axis=0)

            # x = np.array(x_left_part)[random_indices][:, 0:mcl, :].reshape(-1, dimension)
            diagnostic_config.classifier_embeddings.fit(x_train[random_indices], y[random_indices], self.neural_language_model, file, '_le')
            # diagnostic_config.classifier_embeddings.fit(x_train, y, self.neural_language_model, file, '_le')


        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_embeddings_')

        if not os.path.isfile(file+".index"):
            x_train = np.zeros((n_sentences, dimension), dtype=float)
            e = np.array(x_right_part)

            for j in range(n_sentences):
                max_length = x_right_sen_len[j]
                x_train[j] = np.mean(e[j, 0:max_length, :], axis=0)

            # x = np.array(x_right_part)[random_indices][:, 0:mcl, :].reshape(-1, dimension)
            diagnostic_config.classifier_embeddings.fit(x_train[random_indices], y[random_indices], self.neural_language_model, file, '_re')
            # diagnostic_config.classifier_embeddings.fit(x_train, y, self.neural_language_model, file, '_re')

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_states_')

        if not os.path.isfile(file+".index"):

            x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
            hs = np.array(information_layer['left_hidden_state'])[0]

            for j in range(n_sentences):
                max_length = x_left_sen_len[j]
                x_train[j] = np.mean(hs[j, 0:max_length, :], axis=0)

            # x = np.array(information_layer['left_hidden_state'])[0][random_indices][:, 0:mcl, :].reshape(-1, dimension*2)
            diagnostic_config.classifier_states.fit(x_train[random_indices], y[random_indices], self.neural_language_model, file, '_ls')
            # diagnostic_config.classifier_states.fit(x_train, y, self.neural_language_model, file, '_ls')

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_states_')

        if not os.path.isfile(file+".index"):
            x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
            hs = np.array(information_layer['right_hidden_state'])[0]

            for j in range(n_sentences):
                max_length = x_right_sen_len[j]
                x_train[j] = np.mean(hs[j, 0:max_length, :], axis=0)

            # x = np.array(information_layer['right_hidden_state'])[0][random_indices][:, 0:mcl, :].reshape(-1, dimension*2)
            diagnostic_config.classifier_states.fit(x_train[random_indices], y[random_indices], self.neural_language_model, file, '_rs')
            # diagnostic_config.classifier_states.fit(x_train, y, self.neural_language_model, file, '_rs')


        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):

                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")

                if not os.path.isfile(file + ".index"):
                    x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
                    ws = np.array(information_layer['weighted_left_hidden_state_' + str(i)])[0]

                    for j in range(n_sentences):
                        max_length = x_left_sen_len[j]
                        x_train[j] = np.sum(ws[j, 0:max_length, :], axis=0)

                    # x = np.array(information_layer['weighted_left_hidden_state_' + str(i)])[0][random_indices][:, 0:mcl, :].reshape(-1, dimension * 2)
                    # diagnostic_config.classifier_states.fit(x, y[random_indices], self.neural_language_model, file,
                    #                                         '_lw' + str(i))
                    diagnostic_config.classifier_states.fit(x_train[random_indices], y[random_indices], self.neural_language_model, file, '_lw' + str(i))
                    # diagnostic_config.classifier_states.fit(x_train, y, self.neural_language_model, file, '_lw' + str(i))


                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")

                if not os.path.isfile(file + ".index"):
                    x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
                    ws = np.array(information_layer['weighted_right_hidden_state_' + str(i)])[0]

                    for j in range(n_sentences):
                        max_length = x_right_sen_len[j]
                        x_train[j] = np.sum(ws[j, 0:max_length, :], axis=0)

                    # x = np.array(information_layer['weighted_right_hidden_state_' + str(i)])[0][random_indices][:, 0:mcl, :].reshape(-1, dimension * 2)
                    diagnostic_config.classifier_states.fit(x_train[random_indices], y[random_indices], self.neural_language_model, file, '_rw' + str(i))
                    # diagnostic_config.classifier_states.fit(x_train, y, self.neural_language_model, file, '_rw' + str(i))

        else:
            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_weighted_')

            if not os.path.isfile(file + ".index"):
                x = np.array(information_layer['weighted_left_hidden_state_'])[0][random_indices][:, 0:mcl, :].\
                    reshape(-1, dimension * 2)
                diagnostic_config.classifier_states.fit(x, y[random_indices], self.neural_language_model, file, '_lw')

            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_weighted_')

            if not os.path.isfile(file + ".index"):
                x = np.array(information_layer['weighted_right_hidden_state'])[0][random_indices][:, 0:mcl, :].\
                    reshape(-1, dimension * 2)
                diagnostic_config.classifier_states.fit(x, y[random_indices], self.neural_language_model, file, '_rw')

    def predict_full_context_diagnostic_classifier(self, diagnostic_config, x, aspects, y, name):

        true_y = [0] * y.shape[1]
        argmax_y = np.argmax(y, axis=1)
        for arg_max in argmax_y:
            true_y[arg_max] += 1

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.internal_data_loader.split_embeddings(
                x, aspects, self.neural_language_model.config.max_sentence_length,
                self.neural_language_model.config.max_target_length)

        _, information_layer = self.neural_language_model.predict(np.array(x_left_part), np.array(x_target_part),
                                                                  np.array(x_right_part), np.array(x_left_sen_len),
                                                                  np.array(x_tar_len), np.array(x_right_sen_len))
        acc_left_word_embeddings = [0] * y.shape[1]
        acc_right_word_embeddings = [0] * y.shape[1]
        acc_left_hidden_states = [0] * y.shape[1]
        acc_right_hidden_states = [0] * y.shape[1]

        acc_weighted_hidden_state = {}
        mcl = diagnostic_config.max_context_length
        dimension = self.neural_language_model.config.embedding_dimension * diagnostic_config.max_context_length
        n_sentences = x_left_part.shape[0]

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
            for i in range(self.neural_language_model.config.n_iterations_hop):
                acc_weighted_hidden_state['acc_weighted_left_hidden_state_' + str(i)] = [0] * y.shape[1]
                acc_weighted_hidden_state['acc_weighted_right_hidden_state_' + str(i)] = [0] * y.shape[1]
        else:
            acc_weighted_hidden_state['acc_weighted_left_hidden_state'] = [0] * y.shape[1]
            acc_weighted_hidden_state['acc_weighted_right_hidden_state'] = [0] * y.shape[1]

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_embeddings_')

        if os.path.isfile(file+".index"):
            # x = np.array(x_left_part)[:, 0:mcl, :].reshape(-1, dimension)
            x_train = np.zeros((n_sentences, dimension), dtype=float)
            e = np.array(x_left_part)

            for j in range(n_sentences):
                max_length = x_left_sen_len[j]
                x_train[j] = np.mean(e[j, 0:max_length, :], axis=0)

            pred_left_word_embeddings = diagnostic_config.classifier_embeddings.predict(
                x_train, file, '_le')[0]
            arg_max_pred = np.argmax(pred_left_word_embeddings, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_y[index]:
                    acc_left_word_embeddings[arg_max_pred[index]] += 1
            print("acc_left_word_embeddings ", acc_left_word_embeddings)

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_embeddings_')

        if os.path.isfile(file+".index"):
            # x = np.array(x_right_part)[:, 0:mcl, :].reshape(-1, dimension)
            x_train = np.zeros((n_sentences, dimension), dtype=float)
            e = np.array(x_right_part)

            for j in range(n_sentences):
                max_length = x_right_sen_len[j]
                x_train[j] = np.mean(e[j, 0:max_length, :], axis=0)

            pred_right_word_embeddings = diagnostic_config.classifier_embeddings.predict(
                x_train, file, '_re')[0]
            arg_max_pred = np.argmax(pred_right_word_embeddings, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_y[index]:
                    acc_right_word_embeddings[arg_max_pred[index]] += 1
            print("acc_right_word_embeddings ", acc_right_word_embeddings)

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'left_states_')

        if os.path.isfile(file+".index"):
            # x = np.array(information_layer['left_hidden_state'])[0][:, 0:mcl, :].reshape(-1, dimension*2)
            x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
            hs = np.array(information_layer['left_hidden_state'])[0]

            for j in range(n_sentences):
                max_length = x_left_sen_len[j]
                x_train[j] = np.mean(hs[j, 0:max_length, :], axis=0)

            pred_left_hidden_states = diagnostic_config.classifier_states.predict(
                x_train, file, '_ls')[0]
            arg_max_pred = np.argmax(pred_left_hidden_states, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_y[index]:
                    acc_left_hidden_states[arg_max_pred[index]] += 1
            print("acc_left_hidden_states ", acc_left_hidden_states)

        file = diagnostic_config.get_file_of_model_savings(
            self.neural_language_model.config.name_of_model, 'right_states_')

        if os.path.isfile(file+".index"):
            # x = np.array(information_layer['right_hidden_state'])[0][:, 0:mcl, :].reshape(-1, dimension * 2)
            x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
            hs = np.array(information_layer['right_hidden_state'])[0]

            for j in range(n_sentences):
                max_length = x_right_sen_len[j]
                x_train[j] = np.mean(hs[j, 0:max_length, :], axis=0)
            pred_right_hidden_states = diagnostic_config.classifier_states.predict(
                x_train, file, '_rs')[0]
            arg_max_pred = np.argmax(pred_right_hidden_states, axis=1)

            for index in range(arg_max_pred.shape[0]):
                if arg_max_pred[index] == argmax_y[index]:
                    acc_right_hidden_states[arg_max_pred[index]] += 1
            print("acc_right_hidden_states ", acc_right_hidden_states)

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):

                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")

                if os.path.isfile(file + ".index"):
                    # x = np.array(information_layer['weighted_left_hidden_state_' + str(i)])[0][:, 0:mcl, :].reshape(-1, dimension*2)
                    x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
                    ws = np.array(information_layer['weighted_left_hidden_state_' + str(i)])[0]

                    for j in range(n_sentences):
                        max_length = x_left_sen_len[j]
                        x_train[j] = np.sum(ws[j, 0:max_length, :], axis=0)

                    pred_weighted_left_hidden_state = diagnostic_config.classifier_states.predict(
                        x_train, file, '_lw' + str(i))[0]
                    arg_max_pred = np.argmax(pred_weighted_left_hidden_state, axis=1)

                    for index in range(arg_max_pred.shape[0]):
                        if arg_max_pred[index] == argmax_y[index]:
                            acc_weighted_hidden_state['acc_weighted_left_hidden_state_' + str(i)][
                                arg_max_pred[index]] += 1

                file = diagnostic_config.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")

                if os.path.isfile(file + ".index"):
                    # x = np.array(information_layer['weighted_right_hidden_state_' + str(i)])[0][:, 0:mcl, :].reshape(-1,
                    #                                                                                                 dimension * 2)
                    x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
                    ws = np.array(information_layer['weighted_right_hidden_state_' + str(i)])[0]

                    for j in range(n_sentences):
                        max_length = x_right_sen_len[j]
                        x_train[j] = np.sum(ws[j, 0:max_length, :], axis=0)

                    pred_weighted_right_hidden_state = diagnostic_config.classifier_states.predict(
                        x_train, file, '_rw' + str(i))[0]
                    arg_max_pred = np.argmax(pred_weighted_right_hidden_state, axis=1)

                    for index in range(arg_max_pred.shape[0]):
                        if arg_max_pred[index] == argmax_y[index]:
                            acc_weighted_hidden_state['acc_weighted_right_hidden_state_' + str(i)][
                                arg_max_pred[index]] += 1
        else:
            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_weighted_')

            if os.path.isfile(file + ".index"):
                # x = np.array(information_layer['weighted_left_hidden_state'])[0][:, 0:mcl, :].reshape(-1, dimension * 2)
                x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
                ws = np.array(information_layer['weighted_left_hidden_state'])[0]

                for j in range(n_sentences):
                    max_length = x_left_sen_len[j]
                    x_train[j] = np.sum(ws[j, 0:max_length, :], axis=0)

                pred_weighted_left_hidden_state = diagnostic_config.classifier_states.predict(x_train, file, '_lw')[0]
                arg_max_pred = np.argmax(pred_weighted_left_hidden_state, axis=1)

                for index in range(arg_max_pred.shape[0]):
                    if arg_max_pred[index] == argmax_y[index]:
                        acc_weighted_hidden_state['acc_weighted_left_hidden_state'][
                            arg_max_pred[index]] += 1

            file = diagnostic_config.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_weighted_')

            if os.path.isfile(file + ".index"):
                # x = np.array(information_layer['weighted_right_hidden_state'])[0][:, 0:mcl, :].reshape(-1, dimension * 2)
                x_train = np.zeros((n_sentences, dimension * 2), dtype=float)
                ws = np.array(information_layer['weighted_right_hidden_state'])[0]

                for j in range(n_sentences):
                    max_length = x_right_sen_len[j]
                    x_train[j] = np.sum(ws[j, 0:max_length, :], axis=0)
                pred_weighted_right_hidden_state = diagnostic_config.classifier_states.predict(
                    x_train, file, '_rw')[0]
                arg_max_pred = np.argmax(pred_weighted_right_hidden_state, axis=1)

                for index in range(arg_max_pred.shape[0]):
                    if arg_max_pred[index] == argmax_y[index]:
                        acc_weighted_hidden_state['acc_weighted_right_hidden_state'][
                            arg_max_pred[index]] += 1

        results = {
            'name': diagnostic_config.name_of_model,
            'neural_language_model': self.neural_language_model.config.name_of_model,
            'n_true': true_y,
            'n_acc_left_word_embeddings': acc_left_word_embeddings,
            'n_acc_right_word_embeddings': acc_right_word_embeddings,
            'n_acc_left_hidden_states': acc_left_hidden_states,
            'n_acc_right_hidden_states': acc_right_hidden_states,
            'acc_weighted_hidden_state': acc_weighted_hidden_state
        }

        file = diagnostic_config.get_file_of_results(self.neural_language_model.config.name_of_model, name)
        print("file ", file)
        with open(file, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=0)

    @staticmethod
    def normalize(x, epsilon=1e-8):
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
        x_normed = (x - mean) / np.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
        return x_normed, mean, variance

    @staticmethod
    def normalize_with_moments(x, mean, variance, epsilon=1e-8):
        x_normed = (x - mean) / np.sqrt(variance + epsilon)  # epsilon to avoid dividing by zero
        return x_normed
