import numpy as np
from local_interpretable_model.contribution_evaluators import LETALinearRegression, PredictionDifference, \
    SingleSetRegression
import json
from local_interpretable_model.linear_model import LinearModel


class LocalInterpretableModel:

    def __init__(self, config, neural_language_model):
        self.config = config
        self.neural_language_model = neural_language_model

    def run(self):

        # Loading all the necessary information
        x = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        aspects_indices = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        lemmatized_sentence = np.array(self.neural_language_model.internal_data_loader.lemmatized_training)
        sentences_id = self.neural_language_model.internal_data_loader.sentence_id_in_training
        aspects = np.array(self.neural_language_model.internal_data_loader.aspects_training)
        aspects_categories = np.array(self.neural_language_model.internal_data_loader.categories_matrix_training)
        aspects_polarities = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)

        # Obtaining the left, right and target contexts
        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.internal_data_loader.split_embeddings(
                x, aspects_indices, self.neural_language_model.config.max_sentence_length,
                self.neural_language_model.config.max_target_length)

        # Getting the prediction by the neural language model
        y_pred, _ = self.neural_language_model.predict(np.array(x_left_part), np.array(x_target_part),
                                                       np.array(x_right_part), np.array(x_left_sen_len),
                                                       np.array(x_tar_len), np.array(x_right_sen_len))

        model_information = {
            'neural_language_model': self.neural_language_model.config.name_of_model,
            'neighbourhood_algorithm': self.config.locality_model_name,
            'rule_based_classifier': self.config.rule_based_classifier_name,
            'relevance_word_algorithm': self.config.attribute_evaluator_name,
        }

        results = [model_information]

        for index in range(x.shape[0]):

            print("index ", index)

            # Getting the training set Z
            x_neighbours, y_neighbours, weights = self.config.locality_model.get_neighbours(x[index],
                                                                                            aspects_indices[index],
                                                                                            self.neural_language_model)
            pred_y = np.argmax(y_pred[index])

            # Obtaining the indices of the word combinations
            attributes_indices, attributes_words = self.config.decision_tree.extract_word_combinations(
                x_neighbours, y_neighbours, lemmatized_sentence[index], pred_y, aspects_indices[index])

            linear_regression = LinearModel(self.config.learning_rate, self.config.n_epochs,
                                                 self.config.batch_size, self.neural_language_model)

            # Obtain the contributions by A-LIME
            word_evaluator_regression = SingleSetRegression()
            word_relevance_linear_regression, intercepts_lr = word_evaluator_regression.evaluate_word_relevance(
                x_neighbours, y_neighbours, lemmatized_sentence[index], aspects_indices[index], weights,
                linear_regression)
            print("word_relevance_linear_regression ", word_relevance_linear_regression)

            # Obtain the contributions by LETA
            attribute_evaluator_regression = LETALinearRegression(self.config.n_of_subset)
            subsets_word_relevance_linear_regression, intercepts_slr = attribute_evaluator_regression.evaluate_attributes(
                attributes_indices, attributes_words, x_neighbours, y_neighbours, lemmatized_sentence[index],
                aspects_indices[index], weights, linear_regression)
            print("subsets_word_relevance_linear_regression ", subsets_word_relevance_linear_regression)

            # Obtain the contribution by A-LACE
            attribute_evaluator_difference = PredictionDifference(self.config.n_of_subset)
            subsets_word_relevance_pred_difference = attribute_evaluator_difference.evaluate_attributes(
                attributes_indices, attributes_words, x[index], aspects_indices[index], y_pred[index],
                self.neural_language_model, lemmatized_sentence[index])
            print("subsets_word_relevance_pred_difference ", subsets_word_relevance_pred_difference)

            result = {
                'sentence_id': sentences_id[index],
                'sentence_index': index,
                'lemmatized_sentence': lemmatized_sentence[index],
                'aspects': aspects[index],
                'prediction': y_pred[index].tolist(),
                'aspect_category_matrix': aspects_categories[index].tolist(),
                'aspect_polarity_matrix': aspects_polarities[index].tolist(),
                'subsets_word_relevance_linear_regression': subsets_word_relevance_linear_regression,
                'intercepts_slr': intercepts_slr,
                'subsets_word_relevance_pred_difference': subsets_word_relevance_pred_difference,
                'word_relevance_linear_regression': word_relevance_linear_regression,
                'intercepts_lr': intercepts_lr
            }

            results.append(result)

        file = self.config.get_file_of_results(self.neural_language_model.config.name_of_model)
        with open(file, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False)

    def single_run(self, x, y_pred, aspects_indices, lemmatized_sentence):

        # Getting the training set Z
        x_neighbours, y_neighbours, weights = self.config.locality_model.get_neighbours(x, aspects_indices,
                                                                               self.neural_language_model)

        pred_y = np.argmax(y_pred)

        # Obtaining the indices of the word combinations
        attributes_indices, attributes_words = self.config.decision_tree.extract_word_combinations(
            x_neighbours, y_neighbours, lemmatized_sentence, pred_y, aspects_indices)

        linear_regression = LinearModel(self.config.learning_rate, self.config.n_epochs,
                                             self.config.batch_size, self.neural_language_model)
        # Obtain the contributions by A-LIME
        word_evaluator_regression = SingleSetRegression()
        relevance_lr, intercept_lr = word_evaluator_regression.evaluate_word_relevance(x_neighbours, y_neighbours,
                                                                         lemmatized_sentence, aspects_indices, weights,
                                                                                       linear_regression)
        # Obtain the contributions by LETA
        attribute_evaluator_regression = LETALinearRegression(self.config.n_of_subset)
        subsets_relevance_lr, intercept_slr = attribute_evaluator_regression.evaluate_attributes(
            attributes_indices, attributes_words, x_neighbours, y_neighbours, lemmatized_sentence, aspects_indices,
            weights, linear_regression)

        # Obtain the contribution by A-LACE
        attribute_evaluator_difference = PredictionDifference(self.config.n_of_subset)
        subsets_relevance_pd = attribute_evaluator_difference.evaluate_attributes(
            attributes_indices, attributes_words, x, aspects_indices, y_pred, self.neural_language_model,
            lemmatized_sentence)

        return relevance_lr, intercept_lr, subsets_relevance_lr, intercept_slr, subsets_relevance_pd
