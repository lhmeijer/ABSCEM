import numpy as np
from sklearn.linear_model import Lasso


class LASSORegression:

    def __init__(self, alpha):
        self.alpha = alpha

    def evaluate_attributes(self, attributes, x, y):

        x_training = np.zeros((len(x), len(attributes) + 1))
        x_training[:, 0] = np.ones(len(x))

        for index_attribute in range(len(attributes)):

            previous_vector = np.ones(len(x))

            for index in attributes[index_attribute]:
                x_training[:, index_attribute+1] = previous_vector * x[:, index]
                previous_vector = x[:, index]

        lasso_regression = Lasso(alpha=self.alpha, normalize=True, max_iter=1e5)

        model = lasso_regression.fit(x_training, y)

        return model.coef_


class PredictionDifference:

    def evaluate_attributes(self, attributes, word_embeddings, aspect_indices, neural_language_model):

        # attributes are indices of relevant words

        complete_pred = neural_language_model.predict(word_embeddings, aspect_indices)
        word_relevance = []

        for attribute in attributes:

            sentence_representation = np.ones(len(word_embeddings))

            sentence_representation[attribute] = 0

            begin_aspect_index = aspect_indices[0]
            number_of_aspects = len(aspect_indices)

            n_of_ones_before_aspect = int(np.sum(sentence_representation[:begin_aspect_index]))
            neighbour_aspect = np.arange(n_of_ones_before_aspect, n_of_ones_before_aspect + number_of_aspects)
            neighbour_embedding = word_embeddings[sentence_representation == 1]

            pred_neighbour = neural_language_model.predict(neighbour_embedding, neighbour_aspect)
            word_relevance.append((complete_pred - pred_neighbour))

        return word_relevance
