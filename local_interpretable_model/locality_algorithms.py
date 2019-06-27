import numpy as np


class Perturbing:

    def __init__(self, n_of_neighbours):

        self.k = n_of_neighbours

    def get_neighbours(self, x, aspects, neural_language_model):

        sentence_length = len(x)
        print("sentence_length ", sentence_length)

        n_of_possibilities = np.power(2, sentence_length - len(aspects))
        print("n_of_possibilities ", n_of_possibilities)
        if n_of_possibilities < self.k and n_of_possibilities != 0:
            used_k = n_of_possibilities
        else:
            used_k = self.k
        print("used_k ", used_k)
        x_neighbours = []

        begin_aspect_index = aspects[0]
        print("begin_aspect_index ", begin_aspect_index)
        number_of_aspects = len(aspects)

        neighbour_embeddings = []
        neighbour_aspects = []
        weights = np.zeros(used_k,  dtype=float)

        for k in range(used_k):
            # proportion_of_one = np.random.random(1)[0]
            # sentence_representation = np.random.choice([0, 1], size=sentence_length,
            #                                            p=[1 - proportion_of_one, proportion_of_one])

            sentence_representation = np.random.randint(2, size=sentence_length)

            # number_of_zeros = np.where(sentence_representation == 0)[0].shape[0]
            # if number_of_zeros == sentence_length:
            #     rand_number = np.random.randint(sentence_length, size=1)
            #     while rand_number in aspects:
            #         rand_number = np.random.randint(sentence_length, size=1)
            #     sentence_representation[rand_number] = 1
                
            # The aspect(s) always must be in the representation
            sentence_representation[aspects] = 1

            n_of_ones_before_aspect = int(np.sum(sentence_representation[:begin_aspect_index]))
            neighbour_aspect = np.arange(n_of_ones_before_aspect, n_of_ones_before_aspect + number_of_aspects)
            neighbour_aspects.append(neighbour_aspect)

            neighbour_embedding = np.array(x)[sentence_representation == 1]
            neighbour_embeddings.append(neighbour_embedding)

            sentence_representation[aspects] = 0

            # proximity function
            # number_of_zeros = np.where(sentence_representation == 0)[0].shape[0]
            # weights[k] = np.exp(-1*(number_of_zeros / sentence_length))
            number_of_ones = np.where(sentence_representation == 1)[0].shape[0]
            weights[k] = np.exp((number_of_ones / sentence_length) - 1)

            x_neighbours.append(sentence_representation)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            neural_language_model.internal_data_loader.split_embeddings(np.array(neighbour_embeddings),
                                                          np.array(neighbour_aspects),
                                                          neural_language_model.config.max_sentence_length,
                                                          neural_language_model.config.max_target_length)

        pred_neighbours, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part, x_left_sen_len,
                                                           x_tar_len, x_right_sen_len)

        weighted_predictions = [weights[i] * pred_neighbours[i] for i in range(used_k)]

        return np.array(x_neighbours), np.array(weighted_predictions)