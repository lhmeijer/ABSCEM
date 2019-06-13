import numpy as np
from sklearn.linear_model import Lasso, LinearRegression


class LASSORegression:

    def __init__(self, alpha):
        self.alpha = alpha

    def evaluate_attributes(self, attributes, x, y, true_pred):

        print("len(attributes)", len(attributes))
        print("x.shape ", x.shape)
        print("true_pred ", true_pred)

        x_training = np.zeros((x.shape[0], len(attributes) + 1))
        x_training[:, 0] = np.ones(x.shape[0])

        correct_attributes = []

        for index_attribute in range(len(attributes)):

            print("attributes[index_attribute] ", attributes[index_attribute])
            print("np.argmax(true_pred) ", np.argmax(true_pred))
            if attributes[index_attribute][1] == np.argmax(true_pred):

                correct_attributes.append(index_attribute)

                previous_vector = np.ones(x.shape[0])

                for a_tuple in attributes[index_attribute][0]:

                    index = a_tuple[0]
                    print("index ", index)
                    x_index = x[:, index]
                    print("x_index ", x_index)

                    if a_tuple[1] == 0:
                        x_index[x_index == 0] = 2
                        x_index[x_index == 1] = 0
                        x_index[x_index == 2] = 1

                    x_training[:, index_attribute+1] = previous_vector * x_index
                    previous_vector = x_index

        x_training = x_training[:, correct_attributes]
        print("x_training ", x_training.shape)
        print("y ", y.shape)
        print("np.max(y) ", y[:, np.argmax(true_pred)])
        # lasso_regression = Lasso(alpha=self.alpha, normalize=True, max_iter=1e5)
        regression = LinearRegression()

        # model = lasso_regression.fit(x_training, np.max(y, axis=1))
        model = regression.fit(x_training, y[:, np.argmax(true_pred)])
        print("model.coef_ ", model.coef_)

        return model.coef_


class PredictionDifference:

    def evaluate_attributes(self, attributes, word_embeddings, aspect_indices, neural_language_model):

        # attributes are indices of relevant words

        # x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
        #     neural_language_model.config.split_embeddings(np.array([word_embeddings]),
        #                                                   np.array([aspect_indices]),
        #                                                   neural_language_model.config.max_sentence_length,
        #                                                   neural_language_model.config.max_target_length)
        #
        # complete_pred = neural_language_model.predict(x_left_part, x_target_part, x_right_part, x_left_sen_len,
        #                                               x_tar_len, x_right_sen_len)
        word_relevance = []

        for attribute in attributes:

            print("attribute ", attribute)

            sentence_representation1 = np.ones(len(word_embeddings), dtype=int)
            sentence_representation2 = np.ones(len(word_embeddings), dtype=int)

            for a_tuple in attribute[0]:

                if a_tuple[1] == 1:
                    sentence_representation2[a_tuple[0]] = 0
                elif a_tuple[1] == 0:
                    sentence_representation1[a_tuple[0]] = 0

            print("sentence_representation1 ", sentence_representation1)
            print("sentence_representation2 ", sentence_representation2)

            begin_aspect_index = aspect_indices[0]
            number_of_aspects = len(aspect_indices)

            n_of_ones_before_aspect1 = int(np.sum(sentence_representation1[:begin_aspect_index]))
            neighbour_aspects1 = np.arange(n_of_ones_before_aspect1, n_of_ones_before_aspect1 + number_of_aspects)
            neighbour_embeddings1 = np.array(word_embeddings)[sentence_representation1 == 1]

            x_left_part1, x_target_part1, x_right_part1, x_left_sen_len1, x_tar_len1, x_right_sen_len1 = \
                neural_language_model.config.split_embeddings(np.array([neighbour_embeddings1]),
                                                              np.array([neighbour_aspects1]),
                                                              neural_language_model.config.max_sentence_length,
                                                              neural_language_model.config.max_target_length)

            pred_neighbour1, _ = neural_language_model.predict(x_left_part1, x_target_part1, x_right_part1,
                                                            x_left_sen_len1, x_tar_len1, x_right_sen_len1)
            print("pred_neighbour1 ", pred_neighbour1)

            n_of_ones_before_aspect2 = int(np.sum(sentence_representation2[:begin_aspect_index]))
            neighbour_aspects2 = np.arange(n_of_ones_before_aspect2, n_of_ones_before_aspect2 + number_of_aspects)
            neighbour_embeddings2 = np.array(word_embeddings)[sentence_representation2 == 1]

            x_left_part2, x_target_part2, x_right_part2, x_left_sen_len2, x_tar_len2, x_right_sen_len2 = \
                neural_language_model.config.split_embeddings(np.array([neighbour_embeddings2]),
                                                              np.array([neighbour_aspects2]),
                                                              neural_language_model.config.max_sentence_length,
                                                              neural_language_model.config.max_target_length)

            pred_neighbour2, _ = neural_language_model.predict(x_left_part2, x_target_part2, x_right_part2,
                                                            x_left_sen_len2, x_tar_len2, x_right_sen_len2)
            print("pred_neighbour2 ", pred_neighbour2)

            word_relevance.append((pred_neighbour1[0] - pred_neighbour2[0]))
            print(word_relevance)

        return word_relevance
