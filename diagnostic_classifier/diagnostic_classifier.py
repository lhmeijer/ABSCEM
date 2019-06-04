import numpy as np


class DiagnosticClassifier:

    def __init__(self, config, neural_language_model):
        self.config = config
        self.neural_language_model = neural_language_model

    def create_feature_set(self):

        x_training = self.neural_language_model.internal_data_loader.word_embeddings_training_all
        y_training = self.neural_language_model.internal_data_loader.polarity_matrix_training
        x_test = self.neural_language_model.internal_data_loader.word_embeddings_test_all

        training_prediction, training_layer_information = self.neural_language_model.predict(x_training)

        correct_indices = []

        for index in range(len(x_training)):

            if np.argmax(training_prediction[index]) == np.argmax(y_training):
                correct_indices.append(index)

        test_prediction, test_layer_information = self.neural_language_model.predict(x_test)

        return training_layer_information, correct_indices, test_layer_information

    def run(self):

        x_training, training_indices, x_test = self.create_feature_set()

        if self.config.polarity_towards_aspect:
            y_training = self.neural_language_model.internal_data_loader.word_polarities_training[training_indices]
            y_test = self.neural_language_model.internal_data_loader.word_polarities_test

        if self.config.relation_towards_aspect:
            y_training = self.neural_language_model.internal_data_loader.word_polarities_training[training_indices]
            y_test = self.neural_language_model.internal_data_loader.word_polarities_test

        if self.config.ontology_mention:
            y_training = self.neural_language_model.internal_data_loader.word_polarities_training[training_indices]
            y_test = self.neural_language_model.internal_data_loader.word_polarities_test

        if self.config.part_of_speech_tagging:
            y_training = self.neural_language_model.internal_data_loader.word_polarities_training[training_indices]
            y_test = self.neural_language_model.internal_data_loader.word_polarities_test

        if self.config.negation_tagging:
            y_training = self.neural_language_model.internal_data_loader.word_polarities_training[training_indices]
            y_test = self.neural_language_model.internal_data_loader.word_polarities_test