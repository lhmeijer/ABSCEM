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
            # if attributes[index_attribute][1] == np.argmax(true_pred):

            # correct_attributes.append(index_attribute)

            previous_vector = np.ones(x.shape[0])

            for a_tuple in attributes[index_attribute][0]:

                index = a_tuple[0]
                print("index ", index)
                x_index = x[:, index]
                print("x_index ", x_index)

                # if a_tuple[1] == 0:
                #     x_index[x_index == 0] = 2
                #     x_index[x_index == 1] = 0
                #     x_index[x_index == 2] = 1

                x_training[:, index_attribute+1] = previous_vector * x_index
                previous_vector = x_index

        # x_training = x_training[:, correct_attributes]
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

    def __init__(self, n_of_subsets):
        self.n_of_subsets = n_of_subsets

    def evaluate_attributes(self, attributes, word_attributes, word_embeddings, aspect_indices, y_pred,
                            neural_language_model):

        # attributes are indices of relevant words

        # x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
        #     neural_language_model.config.split_embeddings(np.array([word_embeddings]),
        #                                                   np.array([aspect_indices]),
        #                                                   neural_language_model.config.max_sentence_length,
        #                                                   neural_language_model.config.max_target_length)
        #
        # complete_pred, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part, x_left_sen_len,
        #                                                  x_tar_len, x_right_sen_len)
        # print("complete_pred ", complete_pred)
        n_of_attributes = len(attributes)

        begin_aspect_index = aspect_indices[0]
        number_of_aspects = len(aspect_indices)

        neighbour_embeddings = []
        neighbour_aspects = []
        all_words = []

        for attr_index in range(n_of_attributes):

            n_of_words_in_attribute = len(attributes[attr_index][0])

            if n_of_words_in_attribute < self.n_of_subsets + 1:

                sentence_representation = np.ones(len(word_embeddings), dtype=int)

                words = []

                for tuple_index in range(n_of_words_in_attribute):
                    a_tuple = attributes[attr_index][0][tuple_index]
                    sentence_representation[a_tuple[0]] = 0
                    words.append(word_attributes[attr_index][0][tuple_index][0])
                all_words.append(words)

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
        word_relevance = []
        n_of_classes = neural_language_model.config.number_of_classes

        for i in range(word_difference.shape[0]):
            list_of_triples = []
            for c in range(n_of_classes):
                list_of_triples.append([all_words[i], c, word_difference[i][c]])
            word_relevance.append(list_of_triples)

        return word_relevance
