import numpy as np


class Perturbing:

    def __init__(self, n_of_neighbours):

        self.k = n_of_neighbours

    def get_neighbours(self, x, aspects, neural_language_model):

        sentence_length = len(x)

        x_neighbours = []
        y_neighbours = []

        for k in range(self.k):
            proportion_of_one = np.random.random(1)[0]
            sentence_representation = np.random.choice([0, 1], size=(sentence_length,),
                                                       p=[1 - proportion_of_one, proportion_of_one])
            # The aspect(s) always must be in the representation
            sentence_representation[aspects] = 1

            print("sentence_representation ", sentence_representation)

            x_neighbours.append(sentence_representation)

            begin_aspect_index = aspects[0]
            number_of_aspects = len(aspects)

            n_of_ones_before_aspect = int(np.sum(sentence_representation[:begin_aspect_index]))
            neighbour_aspect = np.arange(n_of_ones_before_aspect, n_of_ones_before_aspect + number_of_aspects)
            neighbour_embedding = np.array(x)[sentence_representation == 1]

            x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
                neural_language_model.config.split_embeddings(np.array([neighbour_embedding]),
                                                              np.array([neighbour_aspect]),
                                                              neural_language_model.config.max_sentence_length,
                                                              neural_language_model.config.max_target_length)

            pred_neighbour, _ = neural_language_model.predict(x_left_part, x_target_part, x_right_part, x_left_sen_len,
                                                           x_tar_len, x_right_sen_len)
            print("pred_neighbour ", pred_neighbour)
            # proximity function

            y_neighbours.append(pred_neighbour[0])

        return np.array(x_neighbours), np.array(y_neighbours)