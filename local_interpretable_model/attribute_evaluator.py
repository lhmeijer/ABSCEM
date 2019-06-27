import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


class MyLinearRegression:

    def __init__(self, n_of_subsets):
        self.n_of_subsets = n_of_subsets

    def evaluate_attributes(self, attributes, word_attributes, x, y, lemmas):

        n_of_attributes = len(attributes)
        n_of_variables = x.shape[1]

        total_variables = n_of_attributes + n_of_variables

        correct_attributes = []

        x_training = np.zeros((x.shape[0], total_variables))
        all_words = []
        all_indices = []

        for index in range(total_variables):

            if index < n_of_variables:
                x_training[:, index] = x[:, index]
                correct_attributes.append(index)
                all_words.append(lemmas[index])
                all_indices.append([index])
            else:
                attr_index = index - n_of_variables
                n_of_words_in_attribute = len(attributes[attr_index][0])
                if n_of_words_in_attribute < self.n_of_subsets + 1 and n_of_words_in_attribute != 1:

                    words = []
                    indices = []
                    previous_vector = np.ones(x.shape[0])
                    correct_attributes.append(index)

                    for tuple_index in range(n_of_words_in_attribute):
                        at_index = attributes[attr_index][0][tuple_index][0]
                        x_index = x[:, at_index]

                        x_training[:, index] = previous_vector * x_index
                        previous_vector = x_training[:, index]

                        words.append(word_attributes[attr_index][0][tuple_index][0])
                        indices.append(attributes[attr_index][0][tuple_index][0])
                    all_words.append(words)
                    all_indices.append(indices)

        x_training = x_training[:, correct_attributes]

        n_of_classes = y.shape[1]
        coefficients = []

        for c in range(n_of_classes):
            lasso_model = linear_model.Lasso(alpha=0.001, fit_intercept=True)
            model = lasso_model.fit(x_training, y[:, c])
            abs_sum = np.nansum(np.abs(model.coef_), axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                # coefficients.append(np.nan_to_num(model.coef_ / abs_sum))
                coefficients.append(model.coef_)

        word_relevance = []

        for i in range(len(correct_attributes)):
            dictionary = {
                'word_attribute': all_words[i],
                'indices_attribute': all_indices[i]
            }
            for c in range(n_of_classes):
                # dictionary[c] = float(coefficients[c][i])
                sum_of_coefficients = float(coefficients[c][i])
                if i >= n_of_variables:
                    for word_index in range(len(all_indices)):
                        if word_index == i:
                            continue
                        if list(set(all_indices[word_index]).intersection(all_indices[i])) == all_indices[word_index]:
                            sum_of_coefficients += coefficients[c][word_index]
                dictionary[c] = sum_of_coefficients

            word_relevance.append(dictionary)

        return word_relevance


class PredictionDifference:

    def __init__(self, n_of_subsets):
        self.n_of_subsets = n_of_subsets

    def evaluate_attributes(self, attributes, word_attributes, word_embeddings, aspect_indices, y_pred,
                            neural_language_model, lemmas):

        n_of_attributes = len(attributes)
        n_of_variables = len(lemmas)

        total_variables = n_of_attributes + n_of_variables

        begin_aspect_index = aspect_indices[0]
        number_of_aspects = len(aspect_indices)

        neighbour_embeddings = []
        neighbour_aspects = []
        all_words = []
        all_indices = []

        for index in range(total_variables):

            if index in aspect_indices:
                continue

            sentence_representation = None

            if index < n_of_variables:
                sentence_representation = np.ones(len(word_embeddings), dtype=int)
                sentence_representation[index] = 0
                all_words.append(lemmas[index])
                all_indices.append(index)
            else:
                attr_index = index - n_of_variables
                n_of_words_in_attribute = len(attributes[attr_index][0])

                if n_of_words_in_attribute < self.n_of_subsets + 1 and n_of_words_in_attribute != 1:

                    words = []
                    indices = []
                    sentence_representation = np.ones(len(word_embeddings), dtype=int)

                    for tuple_index in range(n_of_words_in_attribute):
                        a_tuple = attributes[attr_index][0][tuple_index]
                        sentence_representation[a_tuple[0]] = 0
                        words.append(word_attributes[attr_index][0][tuple_index][0])
                        indices.append(attributes[attr_index][0][tuple_index][0])
                    all_words.append(words)
                    all_indices.append(indices)

            if sentence_representation is not None:
                n_of_ones_before_aspect = int(np.sum(sentence_representation[:begin_aspect_index]))
                neighbour_aspect = np.arange(n_of_ones_before_aspect, n_of_ones_before_aspect + number_of_aspects)
                neighbour_aspects.append(neighbour_aspect)

                neighbour_embedding = np.array(word_embeddings)[sentence_representation == 1]
                neighbour_embeddings.append(neighbour_embedding)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            neural_language_model.internal_data_loader.split_embeddings(np.array(neighbour_embeddings),
                                                          np.array(neighbour_aspects),
                                                          neural_language_model.config.max_sentence_length,
                                                          neural_language_model.config.max_target_length)

        pred_neighbour, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                          x_left_sen_len, x_tar_len, x_right_sen_len)

        word_difference = y_pred - pred_neighbour
        word_difference_abs_sum = np.nansum(np.abs(word_difference), axis=0)
        word_relevance = []
        n_of_classes = neural_language_model.config.number_of_classes

        for i in range(word_difference.shape[0]):
            dictionary = {
                'word_attribute': all_words[i],
                'indices_attribute': all_indices[i]
            }
            for c in range(n_of_classes):
                with np.errstate(divide='ignore', invalid='ignore'):
                    # dictionary[c] = float(np.nan_to_num(word_difference[i][c] / word_difference_abs_sum[c]))
                    dictionary[c] = word_difference[i][c]
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
            # regression = LinearRegression()
            # model = regression.fit(x, y[:, c])
            lasso_model = linear_model.Lasso(alpha=0.001, fit_intercept=True)
            model = lasso_model.fit(x, y[:, c])
            abs_sum = np.nansum(np.abs(model.coef_), axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                # coefficients.append(np.nan_to_num(model.coef_ / abs_sum))
                coefficients.append(model.coef_)

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
            neural_language_model.internal_data_loader.split_embeddings(np.array(neighbour_embeddings),
                                                          np.array(neighbour_aspects),
                                                          neural_language_model.config.max_sentence_length,
                                                          neural_language_model.config.max_target_length)

        pred_neighbour, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                          x_left_sen_len, x_tar_len, x_right_sen_len)

        word_difference = y_pred - pred_neighbour
        word_difference_abs_sum = np.nansum(np.abs(word_difference), axis=0)
        word_relevance = []
        n_of_classes = neural_language_model.config.number_of_classes

        for i in range(word_difference.shape[0]):
            dictionary = {
                'word_attribute': all_words[i],
            }
            for c in range(n_of_classes):
                with np.errstate(divide='ignore', invalid='ignore'):
                    dictionary[c] = float(np.nan_to_num(word_difference[i][c] / word_difference_abs_sum[c]))
            word_relevance.append(dictionary)

        return word_relevance
