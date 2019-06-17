import numpy as np
from sklearn.linear_model import LinearRegression


class MyLinearRegression:

    def __init__(self, n_of_subsets):
        self.n_of_subsets = n_of_subsets

    def evaluate_attributes(self, attributes, word_attributes, x, y):

        n_of_attributes = len(attributes)
        correct_attributes = []

        x_training = np.zeros((x.shape[0], n_of_attributes))
        # x_training[:, 0] = np.ones(x.shape[0])
        all_words = []
        all_indices = []

        for attr_index in range(n_of_attributes):

            n_of_words_in_attribute = len(attributes[attr_index][0])

            if n_of_words_in_attribute < self.n_of_subsets + 1:

                words = []
                indices = []
                previous_vector = np.ones(x.shape[0])
                correct_attributes.append(attr_index)

                for tuple_index in range(n_of_words_in_attribute):

                    index = attributes[attr_index][0][tuple_index][0]
                    x_index = x[:, index]

                    x_training[:, attr_index] = previous_vector * x_index
                    previous_vector = x_index

                    words.append(word_attributes[attr_index][0][tuple_index][0])
                    indices.append(attributes[attr_index][0][tuple_index][0])

                all_words.append(words)
                all_indices.append(indices)

        x_training = x_training[:, correct_attributes]

        n_of_classes = y.shape[1]
        coefficients = []

        for c in range(n_of_classes):
            regression = LinearRegression()
            model = regression.fit(x_training, y[:, c])
            abs_sum = np.sum(np.abs(model.coef_), axis=0)
            coefficients.append(model.coef_ / abs_sum)

        word_relevance = []

        for i in correct_attributes:
            dictionary = {
                'word_attribute': all_words[i],
                'indices_attribute': all_indices[i]
            }
            for c in range(n_of_classes):
                dictionary[c] = float(coefficients[c][i])
            word_relevance.append(dictionary)

        return word_relevance


class PredictionDifference:

    def __init__(self, n_of_subsets):
        self.n_of_subsets = n_of_subsets

    def evaluate_attributes(self, attributes, word_attributes, word_embeddings, aspect_indices, y_pred,
                            neural_language_model):

        n_of_attributes = len(attributes)

        begin_aspect_index = aspect_indices[0]
        number_of_aspects = len(aspect_indices)

        neighbour_embeddings = []
        neighbour_aspects = []
        all_words = []
        all_indices = []

        for attr_index in range(n_of_attributes):

            n_of_words_in_attribute = len(attributes[attr_index][0])

            if n_of_words_in_attribute < self.n_of_subsets + 1:

                sentence_representation = np.ones(len(word_embeddings), dtype=int)

                words = []
                indices = []

                for tuple_index in range(n_of_words_in_attribute):
                    a_tuple = attributes[attr_index][0][tuple_index]
                    sentence_representation[a_tuple[0]] = 0
                    words.append(word_attributes[attr_index][0][tuple_index][0])
                    indices.append(attributes[attr_index][0][tuple_index][0])
                all_words.append(words)
                all_indices.append(indices)

                n_of_ones_before_aspect = int(np.sum(sentence_representation[:begin_aspect_index]))
                neighbour_aspect = np.arange(n_of_ones_before_aspect, n_of_ones_before_aspect + number_of_aspects)
                neighbour_aspects.append(neighbour_aspect)

                neighbour_embedding = np.array(word_embeddings)[sentence_representation == 1]
                neighbour_embeddings.append(neighbour_embedding)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            neural_language_model.config.split_embeddings(np.array(neighbour_embeddings),
                                                          np.array(neighbour_aspects),
                                                          neural_language_model.config.max_sentence_length,
                                                          neural_language_model.config.max_target_length)

        pred_neighbour, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                          x_left_sen_len, x_tar_len, x_right_sen_len)

        word_difference = y_pred - pred_neighbour
        word_difference_abs_sum = np.sum(np.abs(word_difference), axis=0)
        word_relevance = []
        n_of_classes = neural_language_model.config.number_of_classes

        for i in range(word_difference.shape[0]):
            dictionary = {
                'word_attribute': all_words[i],
                'indices_attribute': all_indices[i]
            }
            for c in range(n_of_classes):
                dictionary[c] = float(word_difference[i][c] / word_difference_abs_sum[c])
            word_relevance.append(dictionary)

        return word_relevance


class SingleSetRegression:

    @staticmethod
    def evaluate_word_relevance(x, y, lemmas):

        n_of_variables = x.shape[1]

        assert n_of_variables == len(lemmas)

        n_of_classes = y.shape[1]
        coefficients = []

        word_relevance = []

        for c in range(n_of_classes):
            regression = LinearRegression()
            model = regression.fit(x, y[:, c])
            abs_sum = np.sum(np.abs(model.coef_), axis=0)
            coefficients.append(model.coef_ / abs_sum)

        for i in range(n_of_variables):
            dictionary = {
                'word_attribute': lemmas[i],
            }
            for c in range(n_of_classes):
                dictionary[c] = float(coefficients[c][i])
            word_relevance.append(dictionary)

        return word_relevance


class SingleSetPredictionDifference:

    @staticmethod
    def evaluate_word_relevance(word_embeddings, aspect_indices, y_pred, lemmas, neural_language_model):

        n_of_variables = len(word_embeddings)

        assert n_of_variables == len(lemmas)

        begin_aspect_index = aspect_indices[0]
        number_of_aspects = len(aspect_indices)

        neighbour_embeddings = []
        neighbour_aspects = []
        all_words = []

        for var_index in range(n_of_variables):

            if var_index not in aspect_indices:

                sentence_representation = np.ones(n_of_variables, dtype=int)
                sentence_representation[var_index] = 0
                all_words.append(lemmas[var_index])

                n_of_ones_before_aspect = int(np.sum(sentence_representation[:begin_aspect_index]))
                neighbour_aspect = np.arange(n_of_ones_before_aspect, n_of_ones_before_aspect + number_of_aspects)
                neighbour_aspects.append(neighbour_aspect)

                neighbour_embedding = np.array(word_embeddings)[sentence_representation == 1]
                neighbour_embeddings.append(neighbour_embedding)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            neural_language_model.config.split_embeddings(np.array(neighbour_embeddings),
                                                          np.array(neighbour_aspects),
                                                          neural_language_model.config.max_sentence_length,
                                                          neural_language_model.config.max_target_length)

        pred_neighbour, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                          x_left_sen_len, x_tar_len, x_right_sen_len)

        word_difference = y_pred - pred_neighbour
        word_difference_abs_sum = np.sum(np.abs(word_difference), axis=0)
        word_relevance = []
        n_of_classes = neural_language_model.config.number_of_classes

        for i in range(word_difference.shape[0]):
            dictionary = {
                'word_attribute': all_words[i],
            }
            for c in range(n_of_classes):
                dictionary[c] = float(word_difference[i][c] / word_difference_abs_sum[c])
            print("dictionary ", dictionary)
            word_relevance.append(dictionary)

        return word_relevance
