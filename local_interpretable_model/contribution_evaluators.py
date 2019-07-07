import numpy as np


class LETALinearRegression:

    def __init__(self, n_of_subsets):
        self.n_of_subsets = n_of_subsets

    def evaluate_attributes(self, attributes, word_attributes, x, y, lemmas, aspect_indices, weights, linear_regression):

        n_of_attributes = len(attributes)
        n_of_variables = x.shape[1]

        total_variables = n_of_attributes + n_of_variables

        correct_attributes = []

        x_training = np.zeros((x.shape[0], total_variables))
        all_words = []
        all_indices = []

        for index in range(total_variables):

            if index in aspect_indices:
                continue

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
        intercepts = []

        for c in range(n_of_classes):
            coefficient, intercept = linear_regression.fit(x_training, np.transpose(np.array([y[:, c]])),
                                                np.transpose(np.array([weights])))
            coefficients.append(coefficient)
            intercepts.append(float(intercept[0]))

        word_relevance = []

        for i in range(len(correct_attributes)):
            dictionary = {
                'word_attribute': all_words[i],
                'indices_attribute': all_indices[i]
            }
            for c in range(n_of_classes):
                coef = float(coefficients[c][i])
                intc = intercepts[c]
                sum_of_coefficients = coef + intc
                if i >= n_of_variables:
                    for word_index in range(len(all_indices)):
                        if word_index == i:
                            continue
                        if list(set(all_indices[word_index]).intersection(all_indices[i])) == all_indices[word_index]:
                            sum_of_coefficients += float(coefficients[c][word_index])
                dictionary[c] = sum_of_coefficients

            word_relevance.append(dictionary)

        return word_relevance, intercepts


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
                all_indices.append([index])
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
        word_relevance = []
        n_of_classes = neural_language_model.config.number_of_classes

        for i in range(word_difference.shape[0]):
            dictionary = {
                'word_attribute': all_words[i],
                'indices_attribute': all_indices[i]
            }
            for c in range(n_of_classes):
                dictionary[c] = float(word_difference[i][c])
            word_relevance.append(dictionary)

        return word_relevance


class SingleSetRegression:

    @staticmethod
    def evaluate_word_relevance(x, y, lemmas, aspect_indices, weights, linear_regression):

        n_of_variables = x.shape[1]

        assert n_of_variables == len(lemmas)

        n_of_classes = y.shape[1]
        coefficients = []
        intercepts = []

        word_relevance = []

        variable_indices = np.arange(n_of_variables, dtype=int)
        variable_indices = np.delete(variable_indices, aspect_indices)
        train_x = np.delete(x, aspect_indices, axis=1)

        for c in range(n_of_classes):
            coefficient, intercept = linear_regression.fit(train_x, np.transpose(np.array([y[:, c]])),
                                        np.transpose(np.array([weights])))
            coefficients.append(coefficient)
            intercepts.append(float(intercept[0]))

        for i in range(len(variable_indices)):

            dictionary = {
                'word_attribute': lemmas[variable_indices[i]],
                'indices_attribute': [int(variable_indices[i])]
            }
            for c in range(n_of_classes):
                coef = float(coefficients[c][i])
                intc = intercepts[c]
                dictionary[c] = coef + intc
            word_relevance.append(dictionary)

        return word_relevance, intercepts
