import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierPolarityConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierMentionConfig


class DiagnosticClassifier:

    def __init__(self, neural_language_model):
        self.neural_language_model = neural_language_model

        if not os.path.isfile(neural_language_model.config.tr_file_of_hidden_layers) or not \
                os.path.isfile(neural_language_model.config.te_file_of_hidden_layers):
            self.create_file_hidden_layers()

    def create_file_hidden_layers(self):

        x_training = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        y_training = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)

        tr_feature_values = self.create_feature_set(x_training, train_aspects, y_training)

        with open(self.neural_language_model.config.tr_file_of_hidden_layers, 'w') as outfile:
            json.dump(tr_feature_values, outfile, ensure_ascii=False)

        x_test = np.array(self.neural_language_model.internal_data_loader.word_embeddings_test_all)
        test_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_test)
        y_test = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_test)

        te_feature_values = self.create_feature_set(x_test, test_aspects, y_test)

        with open(self.neural_language_model.config.te_file_of_hidden_layers, 'w') as outfile:
            json.dump(te_feature_values, outfile, ensure_ascii=False)

    def create_feature_set(self, x, aspects, y):

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.config.split_embeddings(x, aspects,
                                                               self.neural_language_model.config.max_sentence_length,
                                                               self.neural_language_model.config.max_target_length)

        left_word_embeddings = []
        right_word_embeddings = []
        left_hidden_states = []
        right_hidden_states = []

        weighted_left_hidden_state = []
        weighted_right_hidden_state = []
        correct_indices = []

        for index in range(x.shape[0]):

            print("index ", index)

            tr_pred, tr_layer_information = self.neural_language_model.predict(np.array([x_left_part[index]]),
                                                                               np.array([x_target_part[index]]),
                                                                               np.array([x_right_part[index]]),
                                                                               np.array([x_left_sen_len[index]]),
                                                                               np.array([x_tar_len[index]]),
                                                                               np.array([x_right_sen_len[index]]))
            if np.argmax(tr_pred) == np.argmax(y[index]):
                correct_indices.append(index)
                left_word_embeddings.append(x_left_part[index].tolist())
                right_word_embeddings.append(x_right_part[index].tolist())
                left_hidden_states.append(tr_layer_information['left_hidden_state'][0].tolist())
                right_hidden_states.append(tr_layer_information['right_hidden_state'][0].tolist())
                weighted_left_hidden_state.append(tr_layer_information['weighted_left_hidden_state'][0].tolist())
                weighted_right_hidden_state.append(tr_layer_information['weighted_right_hidden_state'][0].tolist())

        feature_values = {
            'left_word_embeddings': left_word_embeddings,
            'right_word_embeddings': right_word_embeddings,
            'left_hidden_states': left_hidden_states,
            'right_hidden_states': right_hidden_states,
            'weighted_left_hidden_state': weighted_left_hidden_state,
            'weighted_right_hidden_state': weighted_right_hidden_state,
            'correct_indices': correct_indices,
        }

        return feature_values

    @staticmethod
    def to_one_hot_encoding(values):

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)

        # binary encode
        one_hot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)

        return one_hot_encoded

    def run(self, diagnostic_classifiers):

        print('I am diagnostic classifier.')

        with open(self.neural_language_model.config.tr_file_of_hidden_layers, 'r') as file:
            for line in file:
                tr_feature_values = json.loads(line)

        with open(self.neural_language_model.config.te_file_of_hidden_layers, 'r') as file:
            for line in file:
                te_feature_values = json.loads(line)

        tr_indices = np.array(tr_feature_values['correct_indices'], dtype=int)
        te_indices = np.array(te_feature_values['correct_indices'], dtype=int)

        print("tr_indices ", tr_indices[0:10])

        if diagnostic_classifiers['mentions']:
            y_training = np.array(self.neural_language_model.internal_data_loader.word_polarities_training)[tr_indices]
            y_test = np.array(self.neural_language_model.internal_data_loader.word_polarities_test[te_indices])

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierMentionConfig, y_training, y_test,
                                                      tr_feature_values, te_feature_values)

        if diagnostic_classifiers['polarities']:
            y_training = np.array(self.neural_language_model.internal_data_loader.word_mentions_training)[tr_indices]
            y_test = np.array(self.neural_language_model.internal_data_loader.word_mentions_test)[te_indices]

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierPolarityConfig, y_training, y_test,
                                                      tr_feature_values, te_feature_values)

        if diagnostic_classifiers['relations']:
            y_training = np.array(self.neural_language_model.internal_data_loader.word_relations_training[tr_indices])
            y_test = np.array(self.neural_language_model.internal_data_loader.word_relations_test[te_indices])

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierRelationConfig, y_training, y_test,
                                                      tr_feature_values, te_feature_values)

        if diagnostic_classifiers['part_of_speech']:
            y_training = np.array(self.neural_language_model.internal_data_loader.part_of_speech_training[tr_indices])
            y_test = np.array(self.neural_language_model.internal_data_loader.part_of_speech_test[te_indices])

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierPOSConfig, y_training, y_test,
                                                      tr_feature_values, te_feature_values)

    def run_results_of_diagnostic_classifier(self, diagnostic_config, y_training, y_test, tr_feature_values,
                                             te_feature_values):

        print("y_training: ", y_training.shape)
        print("y_test: ", y_test.shape)

        tr_y_one_hot_encoding = self.to_one_hot_encoding(y_training)
        print("tr_y_one_hot_encoding.shape ", tr_y_one_hot_encoding.shape)
        te_y_one_hot_encoding = self.to_one_hot_encoding(y_test)
        print("te_y_one_hot_encoding. ", te_y_one_hot_encoding.shape)

        acc_left_word_embeddings = diagnostic_config.classifier.fit(np.array(tr_feature_values['left_word_embeddings']),
                                                                    tr_y_one_hot_encoding,
                                                                    np.array(te_feature_values['left_word_embeddings']),
                                                                    te_y_one_hot_encoding,
                                                                    self.neural_language_model)
        acc_right_word_embeddings = diagnostic_config.classifier.fit(
            np.array(tr_feature_values['right_word_embeddings']), tr_y_one_hot_encoding,
            np.array(te_feature_values['right_word_embeddings']), te_y_one_hot_encoding, self.neural_language_model)

        acc_left_hidden_states = diagnostic_config.classifier.fit(np.array(tr_feature_values['left_hidden_states']),
                                                                  tr_y_one_hot_encoding,
                                                                  np.array(te_feature_values['left_hidden_states']),
                                                                  te_y_one_hot_encoding,
                                                                  self.neural_language_model)

        acc_right_hidden_states = diagnostic_config.classifier.fit(np.array(tr_feature_values['right_hidden_states']),
                                                                   tr_y_one_hot_encoding,
                                                                   np.array(te_feature_values['right_hidden_states']),
                                                                   te_y_one_hot_encoding,
                                                                   self.neural_language_model)

        acc_weighted_left_hidden_state = diagnostic_config.classifier.fit(
            np.array(tr_feature_values['weighted_left_hidden_state']), tr_y_one_hot_encoding,
            np.array(te_feature_values['weighted_left_hidden_state']), te_y_one_hot_encoding,
            self.neural_language_model)

        acc_weighted_right_hidden_state = diagnostic_config.classifier.fit(
            np.array(tr_feature_values['weighted_right_hidden_state']), tr_y_one_hot_encoding,
            np.array(te_feature_values['weighted_right_hidden_state']), te_y_one_hot_encoding,
            self.neural_language_model)

        results = {
            'name': diagnostic_config.name_of_model,
            'neural_language_model': self.neural_language_model.name_of_model,
            'n_training_sample': tr_y_one_hot_encoding.shape[0],
            'n_test_sample': te_y_one_hot_encoding.shape[0],
            'in_sample_acc_acc_left_word_embeddings': acc_left_word_embeddings[0],
            'out_of_sample_acc_acc_left_word_embeddings': acc_right_word_embeddings[1],
            'in_sample_acc_acc_right_word_embeddings': acc_right_word_embeddings[0],
            'out_of_sample_acc_acc_right_word_embeddings': acc_left_word_embeddings[1],
            'in_sample_acc_left_hidden_states': acc_left_hidden_states[0],
            'out_of_sample_acc_left_hidden_states': acc_left_hidden_states[1],
            'in_sample_acc_right_hidden_states': acc_right_hidden_states[0],
            'out_of_sample_acc_right_hidden_states': acc_right_hidden_states[1],
            'in_sample_acc_weighted_left_hidden_state': acc_weighted_left_hidden_state[0],
            'out_of_sample_acc_weighted_left_hidden_state': acc_weighted_right_hidden_state[1],
            'in_sample_acc_weighted_right_hidden_state': acc_weighted_left_hidden_state[0],
            'out_of_sample_acc_weighted_right_hidden_state': acc_weighted_right_hidden_state[1],
        }

        with open(diagnostic_config.file_of_results, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False)
