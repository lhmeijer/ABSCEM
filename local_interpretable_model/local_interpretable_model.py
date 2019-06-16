import numpy as np
from local_interpretable_model.attribute_evaluator import LASSORegression, PredictionDifference
import json


class LocalInterpretableModel:

    def __init__(self, config, neural_language_model):
        self.config = config
        self.neural_language_model = neural_language_model

    def run(self):

        x = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        aspects_indices = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        lemmatized_sentence = np.array(self.neural_language_model.internal_data_loader.lemmatized_training)
        aspects = np.array(self.neural_language_model.internal_data_loader.aspects_training)
        aspects_categories = np.array(self.neural_language_model.internal_data_loader.categories_matrix_training)
        aspects_polarities = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.config.split_embeddings(x, aspects_indices,
                                                               self.neural_language_model.config.max_sentence_length,
                                                               self.neural_language_model.config.max_target_length)

        y_pred, _ = self.neural_language_model.predict(np.array(x_left_part), np.array(x_target_part),
                                                       np.array(x_right_part),np.array(x_left_sen_len),
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

            x_neighbours, y_neighbours = self.config.locality_model.get_neighbours(x[index], aspects_indices[index],
                                                                                   self.neural_language_model)

            print("x_neighbours.shape ", x_neighbours.shape)
            print("y_neighbours.shape ", y_neighbours.shape)

            true_y = np.argmax(aspects_polarities[index])

            attributes_indices, attributes_words = self.config.rule_based_classifier.extract_rules(x_neighbours,
                                                                                                   y_neighbours,
                                                                                                   lemmatized_sentence
                                                                                                   [index], true_y)

            word_relevance = []

            if isinstance(self.config.attribute_evaluator, LASSORegression):

                word_relevance = self.config.attribute_evaluator.evaluate_attributes(attributes_indices, x_neighbours,
                                                                                     y_neighbours,
                                                                                     aspects_polarities[index])

            elif isinstance(self.config.attribute_evaluator, PredictionDifference):

                word_relevance = self.config.attribute_evaluator.evaluate_attributes(attributes_indices,
                                                                                     attributes_words, x[index],
                                                                                     aspects_indices[index],
                                                                                     y_pred[index],
                                                                                     self.neural_language_model)

            result = {
                'lemmatized_sentence': lemmatized_sentence[index],
                'aspects': aspects[index],
                'aspect_category_matrix': aspects_categories[index].tolist(),
                'aspect_polarity_matrix': aspects_polarities[index].tolist(),
                'word_relevance_per_set': word_relevance
            }
            print("result ", result)

            results.append(result)

        with open(self.config.file_of_results, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False)





