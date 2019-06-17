import numpy as np
import json
import os
from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierPolarityConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierMentionConfig


class DiagnosticClassifier:

    def __init__(self, neural_language_model, diagnostic_classifiers):
        self.neural_language_model = neural_language_model
        self.diagnostic_classifiers = diagnostic_classifiers

        # check whether the hidden layers of the neural language model are already saved.
        if not os.path.isfile(neural_language_model.config.tr_file_of_hidden_layers) or not \
                os.path.isfile(neural_language_model.config.te_file_of_hidden_layers):
            self.create_file_hidden_layers()

    def create_file_hidden_layers(self):

        x_training = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        y_training = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)

        tr_mentions = np.array(self.neural_language_model.internal_data_loader.word_mentions_training)
        tr_polarities = np.array(self.neural_language_model.internal_data_loader.word_polarities_training)
        tr_relations = np.array(self.neural_language_model.internal_data_loader.word_relations_training)
        tr_pos_tags = np.array(self.neural_language_model.internal_data_loader.part_of_speech_training)

        tr_feature_values = self.create_feature_set(x_training, train_aspects, y_training, tr_mentions, tr_polarities,
                                                    tr_relations, tr_pos_tags)

        with open(self.neural_language_model.config.tr_file_of_hidden_layers, 'w') as outfile:
            json.dump(tr_feature_values, outfile, ensure_ascii=False)

        x_test = np.array(self.neural_language_model.internal_data_loader.word_embeddings_test_all)
        test_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_test)
        y_test = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_test)

        te_mentions = np.array(self.neural_language_model.internal_data_loader.word_mentions_test)
        te_polarities = np.array(self.neural_language_model.internal_data_loader.word_polarities_test)
        te_relations = np.array(self.neural_language_model.internal_data_loader.word_relations_test)
        te_pos_tags = np.array(self.neural_language_model.internal_data_loader.part_of_speech_test)

        te_feature_values = self.create_feature_set(x_test, test_aspects, y_test, te_mentions, te_polarities,
                                                    te_relations, te_pos_tags)

        with open(self.neural_language_model.config.te_file_of_hidden_layers, 'w') as outfile:
            json.dump(te_feature_values, outfile, ensure_ascii=False)

    def create_feature_set(self, x, aspects, y, mentions, polarities, relations, pos_tags):

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.config.split_embeddings(x, aspects,
                                                               self.neural_language_model.config.max_sentence_length,
                                                               self.neural_language_model.config.max_target_length)

        left_word_embeddings = []
        right_word_embeddings = []
        left_hidden_states = []
        right_hidden_states = []

        weighted_hidden_state = {}

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):
                weighted_hidden_state['weighted_left_hidden_state' + str(i)] = []
                weighted_hidden_state['weighted_right_hidden_state' + str(i)] = []
        else:
            weighted_hidden_state['weighted_left_hidden_state'] = []
            weighted_hidden_state['weighted_right_hidden_state'] = []

        mentions_per_word_left = []
        mentions_count_left = np.zeros(len(mentions[0][0]), dtype=int)
        mentions_per_word_right = []
        mentions_count_right = np.zeros(len(mentions[0][0]), dtype=int)

        polarities_per_word_left = []
        polarities_count_left = np.zeros(len(polarities[0][0]), dtype=int)
        polarities_per_word_right = []
        polarities_count_right = np.zeros(len(polarities[0][0]), dtype=int)

        relations_per_word_left = []
        relations_count_left = np.zeros(len(relations[0][0]), dtype=int)
        relations_per_word_right = []
        relations_count_right = np.zeros(len(relations[0][0]), dtype=int)

        pos_tags_per_word_left = []
        pos_tags_count_left = np.zeros(len(pos_tags[0][0]), dtype=int)
        pos_tags_per_word_right = []
        pos_tags_count_right = np.zeros(len(pos_tags[0][0]), dtype=int)

        for index in range(x.shape[0]):

            print("index ", index)

            tr_pred, tr_layer_information = self.neural_language_model.predict(np.array([x_left_part[index]]),
                                                                               np.array([x_target_part[index]]),
                                                                               np.array([x_right_part[index]]),
                                                                               np.array([x_left_sen_len[index]]),
                                                                               np.array([x_tar_len[index]]),
                                                                               np.array([x_right_sen_len[index]]))
            if np.argmax(tr_pred) == np.argmax(y[index]):

                n_left_words = x_left_sen_len[index]

                for j in range(n_left_words):
                    left_word_embeddings.append(x_left_part[index][j].tolist())
                    left_hidden_states.append(tr_layer_information['left_hidden_state'][0][j].tolist())

                    if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                        for i in range(self.neural_language_model.config.n_iterations_hop):
                            weighted_hidden_state['weighted_left_hidden_state' + str(i)].append(
                                tr_layer_information['weighted_left_hidden_state' + str(i)][0][j].tolist())
                    else:
                        weighted_hidden_state['weighted_left_hidden_state'].append(
                            tr_layer_information['weighted_left_hidden_state'][0][j].tolist())

                    mentions_per_word_left.append(mentions[index][j])
                    index_of_one = mentions[index][j].index(1)
                    mentions_count_left[index_of_one] += 1

                    polarities_per_word_left.append(polarities[index][j])
                    index_of_one = polarities[index][j].index(1)
                    polarities_count_left[index_of_one] += 1

                    relations_per_word_left.append(relations[index][j])
                    index_of_one = relations[index][j].index(1)
                    relations_count_left[index_of_one] += 1

                    pos_tags_per_word_left.append(pos_tags[index][j])
                    index_of_one = pos_tags[index][j].index(1)
                    pos_tags_count_left[index_of_one] += 1

                n_right_words = x_right_sen_len[index]

                end_index = aspects[index][-1]

                for j in range(n_right_words):
                    right_word_embeddings.append(x_right_part[index][j].tolist())
                    right_hidden_states.append(tr_layer_information['right_hidden_state'][0][j].tolist())

                    if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                        for i in range(self.neural_language_model.config.n_iterations_hop):
                            weighted_hidden_state['weighted_right_hidden_state' + str(i)].append(
                                tr_layer_information['weighted_right_hidden_state' + str(i)][0][j].tolist())
                    else:
                        weighted_hidden_state['weighted_right_hidden_state'].append(
                            tr_layer_information['weighted_right_hidden_state'][0][j].tolist())

                    mentions_per_word_right.append(mentions[index][end_index + 1 + j])
                    index_of_one = mentions[index][end_index + 1 + j].index(1)
                    mentions_count_right[index_of_one] += 1

                    polarities_per_word_right.append(polarities[index][end_index + 1 + j])
                    index_of_one = polarities[index][end_index + 1 + j].index(1)
                    polarities_count_right[index_of_one] += 1

                    relations_per_word_right.append(relations[index][end_index + 1 + j])
                    index_of_one = relations[index][end_index + 1 + j].index(1)
                    relations_count_right[index_of_one] += 1

                    pos_tags_per_word_right.append(pos_tags[index][end_index + 1 + j])
                    index_of_one = pos_tags[index][end_index + 1 + j].index(1)
                    pos_tags_count_right[index_of_one] += 1

        feature_values = {
            'left_word_embeddings': left_word_embeddings,
            'right_word_embeddings': right_word_embeddings,
            'left_hidden_states': left_hidden_states,
            'right_hidden_states': right_hidden_states,
            'weighted_hidden_state': weighted_hidden_state,
            'mentions_per_word_left': mentions_per_word_left,
            'mentions_count_left': mentions_count_left.tolist(),
            'mentions_per_word_right': mentions_per_word_right,
            'mentions_count_right': mentions_count_right.tolist(),
            'polarities_per_word_left': polarities_per_word_left,
            'polarities_count_left': polarities_count_left.tolist(),
            'polarities_per_word_right': polarities_per_word_right,
            'polarities_count_right': polarities_count_right.tolist(),
            'relations_per_word_left': relations_per_word_left,
            'relations_count_left': relations_count_left.tolist(),
            'relations_per_word_right': relations_per_word_right,
            'relations_count_right': relations_count_right.tolist(),
            'pos_tags_per_word_left': pos_tags_per_word_left,
            'pos_tags_count_left': pos_tags_count_left.tolist(),
            'pos_tags_per_word_right': pos_tags_per_word_right,
            'pos_tags_count_right': pos_tags_count_right.tolist()
        }

        return feature_values

    def run(self):

        print('I am diagnostic classifier.')

        # Open the files, which consist the hidden layers of the neural language model
        # The hidden layers of the training data
        with open(self.neural_language_model.config.tr_file_of_hidden_layers, 'r') as file:
            for line in file:
                tr_feature_values = json.loads(line)
        # The hidden layers of the test data
        with open(self.neural_language_model.config.te_file_of_hidden_layers, 'r') as file:
            for line in file:
                te_feature_values = json.loads(line)

        # diagnostic classifier to test whether mentions are able to interpret
        if self.diagnostic_classifiers['mentions']:
            self.run_results_of_diagnostic_classifier(DiagnosticClassifierMentionConfig, tr_feature_values,
                                                      te_feature_values, 'mentions')

        if self.diagnostic_classifiers['polarities']:
            self.run_results_of_diagnostic_classifier(DiagnosticClassifierPolarityConfig, tr_feature_values,
                                                      te_feature_values, 'polarities')

        if self.diagnostic_classifiers['relations']:
            self.run_results_of_diagnostic_classifier(DiagnosticClassifierRelationConfig, tr_feature_values,
                                                      te_feature_values, 'relations')

        if self.diagnostic_classifiers['part_of_speech']:
            self.run_results_of_diagnostic_classifier(DiagnosticClassifierPOSConfig, tr_feature_values,
                                                      te_feature_values, 'pos_tags')

    def run_results_of_diagnostic_classifier(self, diagnostic_config, tr_feature_values, te_feature_values, interest):

        left_counts = np.array(tr_feature_values[interest+'_count_left'])
        left_arg_max = np.argmax(left_counts)
        left_items = np.delete(np.arange(left_counts.shape[0]), left_arg_max)
        left_mean_count = int(np.floor(np.mean(left_counts[left_items])))

        right_counts = np.array(tr_feature_values[interest + '_count_right'])
        right_arg_max = np.argmax(right_counts)
        right_items = np.delete(np.arange(right_counts.shape[0]), right_arg_max)
        right_mean_count = int(np.floor(np.mean(right_counts[right_items])))

        tr_left_y = np.array(tr_feature_values[interest+'_per_word_left'])
        left_random_indices = []
        tr_right_y = np.array(tr_feature_values[interest + '_per_word_right'])
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

        acc_left_word_embeddings = diagnostic_config.classifier.fit(
            np.array(tr_feature_values['left_word_embeddings'])[left_random_indices], tr_left_y[left_random_indices],
            np.array(te_feature_values['left_word_embeddings']), np.array(te_feature_values[interest+'_per_word_left']),
            self.neural_language_model)

        acc_right_word_embeddings = diagnostic_config.classifier.fit(
            np.array(tr_feature_values['right_word_embeddings'])[right_random_indices],
            tr_right_y[right_random_indices],  np.array(te_feature_values['right_word_embeddings']),
            np.array(te_feature_values[interest+'_per_word_right']), self.neural_language_model)

        acc_left_hidden_states = diagnostic_config.classifier.fit(
            np.array(tr_feature_values['left_hidden_states'])[left_random_indices], tr_left_y[left_random_indices],
            np.array(te_feature_values['left_hidden_states']), np.array(te_feature_values[interest+'_per_word_left']),
            self.neural_language_model)

        acc_right_hidden_states = diagnostic_config.classifier.fit(
            np.array(tr_feature_values['right_hidden_states'])[right_random_indices], tr_right_y[right_random_indices],
            np.array(te_feature_values['right_hidden_states']), np.array(te_feature_values[interest+'_per_word_right']),
            self.neural_language_model)

        acc_weighted_hidden_state = {}
        tr_dict_weighted_hidden_states = tr_feature_values['weighted_hidden_state']
        te_dict_weighted_hidden_states = te_feature_values['weighted_hidden_state']

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):
                acc_weighted_left_hidden_state = diagnostic_config.classifier.fit(
                    np.array(tr_dict_weighted_hidden_states['weighted_left_hidden_state' + str(i)])
                    [left_random_indices], tr_left_y[left_random_indices],
                    np.array(te_dict_weighted_hidden_states['weighted_left_hidden_state' + str(i)]),
                    np.array(te_feature_values[interest + '_per_word_left']), self.neural_language_model)

                acc_weighted_right_hidden_state = diagnostic_config.classifier.fit(
                    np.array(tr_dict_weighted_hidden_states['weighted_right_hidden_state' + str(i)])
                    [right_random_indices], tr_right_y[right_random_indices],
                    np.array(te_dict_weighted_hidden_states['weighted_right_hidden_state' + str(i)]),
                    np.array(te_feature_values[interest + '_per_word_right']), self.neural_language_model)

                acc_weighted_hidden_state['in_sample_acc_weighted_left_hidden_state' + str(i)] = \
                    acc_weighted_left_hidden_state[0]
                acc_weighted_hidden_state['out_of_sample_acc_weighted_left_hidden_state' + str(i)] = \
                    acc_weighted_left_hidden_state[1]
                acc_weighted_hidden_state['in_sample_left_n_per_class_weighted' + str(i)] = \
                    acc_weighted_left_hidden_state[2].tolist()
                acc_weighted_hidden_state['out_of_sample_left_n_per_class_weighted' + str(i)] = \
                    acc_weighted_left_hidden_state[3].tolist()

                acc_weighted_hidden_state['in_sample_acc_weighted_right_hidden_state' + str(i)] = \
                    acc_weighted_right_hidden_state[0]
                acc_weighted_hidden_state['out_of_sample_acc_weighted_right_hidden_state' + str(i)] = \
                    acc_weighted_right_hidden_state[1]
                acc_weighted_hidden_state['in_sample_right_n_per_class_weighted' + str(i)] = \
                    acc_weighted_right_hidden_state[2].tolist()
                acc_weighted_hidden_state['out_of_sample_right_n_per_class_weighted' + str(i)] = \
                    acc_weighted_right_hidden_state[3].tolist()
        else:
            acc_weighted_left_hidden_state = diagnostic_config.classifier.fit(
                np.array(tr_dict_weighted_hidden_states['weighted_left_hidden_state'])[left_random_indices],
                tr_left_y[left_random_indices],
                np.array(te_dict_weighted_hidden_states['weighted_left_hidden_state']),
                np.array(te_feature_values[interest + '_per_word_left']), self.neural_language_model)

            acc_weighted_right_hidden_state = diagnostic_config.classifier.fit(
                np.array(tr_dict_weighted_hidden_states['weighted_right_hidden_state'])[right_random_indices],
                tr_right_y[right_random_indices],
                np.array(te_dict_weighted_hidden_states['weighted_right_hidden_state']),
                np.array(te_feature_values[interest + '_per_word_right']), self.neural_language_model)

            acc_weighted_hidden_state['in_sample_acc_weighted_left_hidden_state'] = acc_weighted_left_hidden_state[0]
            acc_weighted_hidden_state['out_of_sample_acc_weighted_left_hidden_state'] = \
                acc_weighted_left_hidden_state[1]
            acc_weighted_hidden_state['in_sample_left_n_per_class_weighted'] = \
                acc_weighted_left_hidden_state[2].tolist()
            acc_weighted_hidden_state['out_of_sample_left_n_per_class_weighted'] = \
                acc_weighted_left_hidden_state[3].tolist()

            acc_weighted_hidden_state['in_sample_acc_weighted_right_hidden_state'] = \
                acc_weighted_right_hidden_state[0]
            acc_weighted_hidden_state['out_of_sample_acc_weighted_right_hidden_state'] = \
                acc_weighted_right_hidden_state[1]
            acc_weighted_hidden_state['in_sample_right_n_per_class_weighted'] = \
                acc_weighted_right_hidden_state[2].tolist()
            acc_weighted_hidden_state['out_of_sample_right_n_per_class_weighted'] = \
                acc_weighted_right_hidden_state[3].tolist()

        results = {
            'name': diagnostic_config.name_of_model,
            'neural_language_model': self.neural_language_model.config.name_of_model,
            'n_left_training_sample': len(left_random_indices),
            interest+'_counts_training_left': tr_feature_values[interest+'_count_left'],
            'n_left_test_sample': np.array(te_feature_values[interest+'_per_word_left']).shape[0],
            interest + '_counts_test_left': te_feature_values[interest + '_count_left'],
            'n_right_training_sample': len(right_random_indices),
            interest + '_counts_training_right': tr_feature_values[interest + '_count_right'],
            'n_right_test_sample': np.array(te_feature_values[interest+'_per_word_right']).shape[0],
            interest + '_counts_test_right': te_feature_values[interest + '_count_right'],
            'in_sample_acc_left_word_embeddings': acc_left_word_embeddings[0],
            'out_of_sample_acc_left_word_embeddings': acc_left_word_embeddings[1],
            'in_sample_left_n_per_class_embeddings': acc_left_word_embeddings[2].tolist(),
            'out_of_sample_left_n_per_class_embeddings': acc_left_word_embeddings[3].tolist(),
            'in_sample_acc_right_word_embeddings': acc_right_word_embeddings[0],
            'out_of_sample_acc_right_word_embeddings': acc_right_word_embeddings[1],
            'in_sample_right_n_per_class_embeddings': acc_right_word_embeddings[2].tolist(),
            'out_of_sample_right_n_per_class_embeddings': acc_right_word_embeddings[3].tolist(),
            'in_sample_acc_left_hidden_states': acc_left_hidden_states[0],
            'out_of_sample_acc_left_hidden_states': acc_left_hidden_states[1],
            'in_sample_left_n_per_class_hidden': acc_left_hidden_states[2].tolist(),
            'out_of_sample_left_n_per_class_hidden': acc_left_hidden_states[3].tolist(),
            'in_sample_acc_right_hidden_states': acc_right_hidden_states[0],
            'out_of_sample_acc_right_hidden_states': acc_right_hidden_states[1],
            'in_sample_right_n_per_class_hidden': acc_right_hidden_states[2].tolist(),
            'out_of_sample_right_n_per_class_hidden': acc_right_hidden_states[3].tolist(),
            'acc_weighted_hidden_state': acc_weighted_hidden_state
        }

        file = diagnostic_config.get_file_of_results(self.neural_language_model.config.name_of_model)
        with open(file, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=0)
