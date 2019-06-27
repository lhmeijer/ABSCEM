import numpy as np
from local_interpretable_model.attribute_evaluator import MyLinearRegression, PredictionDifference, \
    SingleSetRegression, SingleSetPredictionDifference
import json


class LocalInterpretableModel:

    def __init__(self, config, neural_language_model):
        self.config = config
        self.neural_language_model = neural_language_model

    def run(self):

        x = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        aspects_indices = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        lemmatized_sentence = np.array(self.neural_language_model.internal_data_loader.lemmatized_training)
        sentences_id = self.neural_language_model.internal_data_loader.sentence_id_in_training
        aspects = np.array(self.neural_language_model.internal_data_loader.aspects_training)
        aspects_categories = np.array(self.neural_language_model.internal_data_loader.categories_matrix_training)
        aspects_polarities = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.internal_data_loader.split_embeddings(
                x, aspects_indices, self.neural_language_model.config.max_sentence_length,
                self.neural_language_model.config.max_target_length)

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
        # word_index = sentences_id.index("744478:1")

        # for index in range(x.shape[0]):
        # for index in [word_index]:
        for index in range(1000, 1015):

            print("index ", index)

            x_neighbours, y_neighbours = self.config.locality_model.get_neighbours(x[index], aspects_indices[index],
                                                                                   self.neural_language_model)
            print("lemmatized ", lemmatized_sentence[index])
            print("x_neighbours ", x_neighbours.shape)
            print("y_neighbours ", y_neighbours.shape)

            true_y = np.argmax(aspects_polarities[index])

            attributes_indices, attributes_words = self.config.rule_based_classifier.extract_rules(x_neighbours,
                                                                                                   y_neighbours,
                                                                                                   lemmatized_sentence
                                                                                                   [index], true_y,
                                                                                                   aspects_indices
                                                                                                   [index])
            word_evaluator_regression = SingleSetRegression()
            word_relevance_linear_regression = word_evaluator_regression.evaluate_word_relevance(x_neighbours,
                                                                                                 y_neighbours,
                                                                                                 lemmatized_sentence
                                                                                                 [index])
            print("word_relevance_linear_regression ", word_relevance_linear_regression)

            attribute_evaluator_regression = MyLinearRegression(self.config.n_of_subset)
            subsets_word_relevance_linear_regression = attribute_evaluator_regression.evaluate_attributes(
                attributes_indices, attributes_words, x_neighbours, y_neighbours, lemmatized_sentence[index])
            print("subsets_word_relevance_linear_regression ", subsets_word_relevance_linear_regression)
            attribute_evaluator_difference = PredictionDifference(self.config.n_of_subset)
            subsets_word_relevance_pred_difference = attribute_evaluator_difference.evaluate_attributes(
                attributes_indices, attributes_words, x[index], aspects_indices[index], y_pred[index],
                self.neural_language_model, lemmatized_sentence[index])
            print("subsets_word_relevance_pred_difference ", subsets_word_relevance_pred_difference)
            #
            # result = {
            #     'sentence_id': sentences_id[index],
            #     'lemmatized_sentence': lemmatized_sentence[index],
            #     'aspects': aspects[index],
            #     'aspect_category_matrix': aspects_categories[index].tolist(),
            #     'aspect_polarity_matrix': aspects_polarities[index].tolist(),
            #     'subsets_word_relevance_linear_regression': subsets_word_relevance_linear_regression,
            #     'subsets_word_relevance_pred_difference': subsets_word_relevance_pred_difference,
            #     'word_relevance_linear_regression': word_relevance_linear_regression,
            #     'word_relevance_prediction_difference': word_relevance_prediction_difference
            # }
            # print("result ", result)
            #
            # results.append(result)

        # file = self.config.get_file_of_results(self.neural_language_model.config.name_of_model)
        # with open(file, 'w') as outfile:
        #     json.dump(results, outfile, ensure_ascii=False)

    def single_run(self, x, aspects_polarity, y_pred, aspects_indices, lemmatized_sentence):

        x_neighbours, y_neighbours = self.config.locality_model.get_neighbours(x, aspects_indices,
                                                                               self.neural_language_model)

        true_y = np.argmax(aspects_polarity)

        attributes_indices, attributes_words = self.config.rule_based_classifier.extract_rules(x_neighbours,
                                                                                               y_neighbours,
                                                                                               lemmatized_sentence,
                                                                                               true_y)
        word_evaluator_regression = SingleSetRegression()
        relevance_lr = word_evaluator_regression.evaluate_word_relevance(x_neighbours, y_neighbours,
                                                                         lemmatized_sentence)
        word_evaluator_difference = SingleSetPredictionDifference()
        relevance_pd = word_evaluator_difference.evaluate_word_relevance(
            x, aspects_indices, y_pred, lemmatized_sentence, self.neural_language_model)

        attribute_evaluator_regression = MyLinearRegression(self.config.n_of_subset)
        subsets_relevance_lr = attribute_evaluator_regression.evaluate_attributes(
            attributes_indices, attributes_words, x_neighbours, y_neighbours)

        attribute_evaluator_difference = PredictionDifference(self.config.n_of_subset)
        subsets_relevance_pd = attribute_evaluator_difference.evaluate_attributes(
            attributes_indices, attributes_words, x, aspects_indices, y_pred, self.neural_language_model)

        return relevance_lr, relevance_pd, subsets_relevance_lr, subsets_relevance_pd
