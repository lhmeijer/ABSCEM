import numpy as np


class Perturbing:

    def __init__(self, n_of_neighbours, sigma):

        self.k = n_of_neighbours
        self.sigma = sigma

    def get_neighbours(self, x, aspects, neural_language_model):

        sentence_length = len(x)

        # Determining how many data points should be in the sample
        n_of_possibilities = np.power(2, sentence_length - len(aspects))

        if n_of_possibilities < self.k and n_of_possibilities != 0:
            used_k = n_of_possibilities
        else:
            used_k = self.k

        x_neighbours = []
        begin_aspect_index = aspects[0]
        number_of_aspects = len(aspects)

        neighbour_embeddings = []
        neighbour_aspects = []
        weights = np.zeros(used_k, dtype=float)

        for k in range(used_k):

            # Getting a random sentence representation zi'
            sentence_representation = np.random.randint(2, size=sentence_length)
                
            # The aspect(s) always must be in the representation
            sentence_representation[aspects] = 1

            n_of_ones_before_aspect = int(np.sum(sentence_representation[:begin_aspect_index]))
            neighbour_aspect = np.arange(n_of_ones_before_aspect, n_of_ones_before_aspect + number_of_aspects)
            neighbour_aspects.append(neighbour_aspect)

            # Getting a random sentence representation zi
            neighbour_embedding = np.array(x)[sentence_representation == 1]
            neighbour_embeddings.append(neighbour_embedding)

            embedding = np.zeros((sentence_length, len(x[0])), dtype=float)
            embedding[sentence_representation == 1] = np.array(x)[sentence_representation == 1]

            # Computing procedure for the weights by a kernel function
            flatten_x = np.array(x).reshape(-1)
            flatten_embedding = embedding.reshape(-1)

            cos_similarity = np.dot(flatten_x, flatten_embedding)/(np.linalg.norm(flatten_x)*np.linalg.norm(flatten_embedding))
            cos_distance = 1 - cos_similarity
            weights[k] = np.exp((-1*cos_distance*cos_distance) / (self.sigma * self.sigma))

            sentence_representation[aspects] = 0

            x_neighbours.append(sentence_representation)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            neural_language_model.internal_data_loader.split_embeddings(
                np.array(neighbour_embeddings),np.array(neighbour_aspects),
                neural_language_model.config.max_sentence_length, neural_language_model.config.max_target_length)

        # Get the prediction by the neural language model of the artificial dataset Z
        pred_neighbours, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part, x_left_sen_len,
                                                           x_tar_len, x_right_sen_len)

        return np.array(x_neighbours), np.array(pred_neighbours), weights
