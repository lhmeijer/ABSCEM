import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierPolarityConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierMentionConfig


class DiagnosticClassifier:

    def __init__(self,neural_language_model):
        self.neural_language_model = neural_language_model

        x_training = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        y_training = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)

        self.tr_feature_values = self.create_feature_set(x_training, train_aspects, y_training)
        self.tr_indices = self.tr_feature_values['correct_indices']

        x_test = np.array(self.neural_language_model.internal_data_loader.word_embeddings_test_all)
        test_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_test)
        y_test = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_test)

        self.te_feature_values = self.create_feature_set(x_test, test_aspects, y_test)
        self.te_indices = self.te_feature_values['correct_indices']

    def create_feature_set(self, x, aspects, y):

        left_hidden_states = np.zeros((x.shape[0], self.neural_language_model.config.embedding_dimension), dtype=float)
        right_hidden_states = np.zeros((x.shape[0], self.neural_language_model.config.embedding_dimension), dtype=float)

        left_context_representation = np.zeros((x.shape[0], self.neural_language_model.config.embedding_dimension),
                                               dtype=float)
        right_context_representation = np.zeros((x.shape[0], self.neural_language_model.config.embedding_dimension),
                                                dtype=float)

        correct_indices = []

        for index in range(x.shape[0]):

            tr_pred, tr_layer_information = self.neural_language_model.predict(x[index], aspects[index])

            if np.argmax(tr_pred) == np.argmax(y[index]):
                correct_indices.append(index)
                left_hidden_states[index] = tr_layer_information['left_hidden_state']
                right_hidden_states[index] = tr_layer_information['right_hidden_state']
                left_context_representation[index] = tr_layer_information['left_context_representation']
                right_context_representation[index] = tr_layer_information['right_context_representation']

        feature_values = {
            'left_hidden_states': left_hidden_states,
            'right_hidden_states': right_hidden_states,
            'left_context_representation': left_context_representation,
            'right_context_representation': right_context_representation,
            'correct_indices': np.array(correct_indices),
        }

        return feature_values

    @staticmethod
    def to_one_hot_encoding(data):
        values = np.array(data)

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)

        # binary encode
        one_hot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)

        return one_hot_encoded

    def run(self, diagnostic_classifiers):

        if diagnostic_classifiers['mentions']:
            y_training = self.neural_language_model.internal_data_loader.word_polarities_training[self.tr_indices]
            y_test = self.neural_language_model.internal_data_loader.word_polarities_test[self.te_indices]

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierMentionConfig, y_training, y_test)

        if diagnostic_classifiers['polarities']:
            y_training = self.neural_language_model.internal_data_loader.word_mentions_training[self.tr_indices]
            y_test = self.neural_language_model.internal_data_loader.word_mentions_test[self.te_indices]

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierPolarityConfig, y_training, y_test)

        if diagnostic_classifiers['relations']:
            y_training = self.neural_language_model.internal_data_loader.word_relations_training[self.tr_indices]
            y_test = self.neural_language_model.internal_data_loader.word_relations_test[self.te_indices]

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierRelationConfig, y_training, y_test)

        if diagnostic_classifiers['part_of_speech']:
            y_training = self.neural_language_model.internal_data_loader.part_of_speech_training[self.tr_indices]
            y_test = self.neural_language_model.internal_data_loader.part_of_speech_test[self.te_indices]

            self.run_results_of_diagnostic_classifier(DiagnosticClassifierPOSConfig, y_training, y_test)

    def run_results_of_diagnostic_classifier(self, diagnostic_config, y_training, y_test):

        tr_y_one_hot_encoding = self.to_one_hot_encoding(y_training)
        te_y_one_hot_encoding = self.to_one_hot_encoding(y_test)

        acc_left_hidden_states = diagnostic_config.classifier.fit(self.tr_feature_values['left_hidden_states'],
                                                                  tr_y_one_hot_encoding[self.tr_indices],
                                                                  self.te_feature_values['left_hidden_states'],
                                                                  te_y_one_hot_encoding[self.te_indices])

        acc_right_hidden_states = diagnostic_config.classifier.fit(self.tr_feature_values['right_hidden_states'],
                                                                   tr_y_one_hot_encoding[self.tr_indices],
                                                                   self.te_feature_values['right_hidden_states'],
                                                                   te_y_one_hot_encoding[self.te_indices])

        acc_left_context_representation = diagnostic_config.classifier.fit(
            self.tr_feature_values['left_context_representation'], tr_y_one_hot_encoding[self.tr_indices],
            self.te_feature_values['left_context_representation'], te_y_one_hot_encoding[self.te_indices])

        acc_right_context_representation = diagnostic_config.classifier.fit(
            self.tr_feature_values['right_context_representation'], tr_y_one_hot_encoding[self.tr_indices],
            self.te_feature_values['left_hidden_states'], te_y_one_hot_encoding[self.te_indices])

        results = {
            'name': diagnostic_config.name_of_model,
            'neural_language_model': self.neural_language_model.name_of_model,
            'n_training_sample': self.tr_indices.shape[0],
            'n_test_sample': self.te_indices.shape[0],
            'in_sample_acc_left_hidden_states': acc_left_hidden_states,
            'out_of_sample_acc_left_hidden_states': acc_left_hidden_states,
            'in_sample_acc_right_hidden_states': acc_right_hidden_states,
            'out_of_sample_acc_right_hidden_states': acc_right_hidden_states,
            'in_sample_acc_acc_left_context_representation': acc_left_context_representation,
            'out_of_sample_acc_acc_left_context_representation': acc_left_context_representation,
            'in_sample_acc_acc_right_context_representation': acc_right_context_representation,
            'out_of_sample_acc_acc_right_context_representation': acc_right_context_representation,

        }

        with open(diagnostic_config.file_of_results, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False)
