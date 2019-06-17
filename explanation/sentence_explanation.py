import numpy as np


class SentenceExplaining:

    def __init__(self, neural_language_model, diagnostic_classifiers, local_interpretable_model):
        self.neural_language_model = neural_language_model
        self.diagnostic_classifiers = diagnostic_classifiers
        self.local_interpretable_model = local_interpretable_model

    def explain_sentence(self, sentence_id):

        sentences_id = self.neural_language_model.internal_data_loader.sentence_id_in_training

        sentence_index = None
        try:
            sentence_index = sentences_id.index(sentence_id)
        except ValueError:
            print(sentence_id + "is not in sentences_id")

        x_training = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        y_training = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)
        lemmatized_sentence = self.neural_language_model.internal_data_loader.lemmatized_training

        mentions = np.array(self.neural_language_model.internal_data_loader.word_mentions_training)
        polarities = np.array(self.neural_language_model.internal_data_loader.word_polarities_training)
        relations = np.array(self.neural_language_model.internal_data_loader.word_relations_training)
        pos_tags = np.array(self.neural_language_model.internal_data_loader.part_of_speech_training)

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.config.split_embeddings(np.array([x_training[sentence_index]]),
                                                               np.array([train_aspects[sentence_index]]),
                                                               self.neural_language_model.config.max_sentence_length,
                                                               self.neural_language_model.config.max_target_length)

        pred, layer_information = self.neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                                     x_left_sen_len, x_tar_len, x_right_sen_len)

        sentence_explanation = {
            'neural_language_model': self.neural_language_model.config.name_of_model
        }

        left_word_embeddings = []
        right_word_embeddings = []
        left_hidden_states = []
        right_hidden_states = []

        weighted_hidden_state = {}

        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

            for i in range(self.neural_language_model.config.n_iterations_hop):
                weighted_hidden_state['weighted_left_hidden_state' + str(i)] = []
                weighted_hidden_state['weighted_right_hidden_state' + str(i)] = []
        else:
            weighted_hidden_state['weighted_left_hidden_state'] = []
            weighted_hidden_state['weighted_right_hidden_state'] = []

        mentions_per_word_left = []
        mentions_count_left = np.zeros(len(mentions[0][0]), dtype=int)
        mentions_per_word_right = []
        mentions_count_right = np.zeros(len(mentions[0][0]), dtype=int)

        polarities_per_word_left = []
        polarities_count_left = np.zeros(len(polarities[0][0]), dtype=int)
        polarities_per_word_right = []
        polarities_count_right = np.zeros(len(polarities[0][0]), dtype=int)

        relations_per_word_left = []
        relations_count_left = np.zeros(len(relations[0][0]), dtype=int)
        relations_per_word_right = []
        relations_count_right = np.zeros(len(relations[0][0]), dtype=int)

        pos_tags_per_word_left = []
        pos_tags_count_left = np.zeros(len(pos_tags[0][0]), dtype=int)
        pos_tags_per_word_right = []
        pos_tags_count_right = np.zeros(len(pos_tags[0][0]), dtype=int)

        n_left_words = x_left_sen_len[0]

        for j in range(n_left_words):

            left_word_embeddings = x_left_part[0][j].tolist()
            left_hidden_states = layer_information['left_hidden_state'][0][j].tolist()

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                for i in range(self.neural_language_model.config.n_iterations_hop):
                    weighted_hidden_state['left_attention_score' + str(i)] = \
                        layer_information['left_attention_score' + str(i)][0][j].tolist()
                    weighted_hidden_state['weighted_left_hidden_state' + str(i)] = \
                        layer_information['weighted_left_hidden_state' + str(i)][0][j].tolist()
            else:
                weighted_hidden_state['left_attention_score'] = \
                    layer_information['left_attention_score'][0][j].tolist()
                weighted_hidden_state['weighted_left_hidden_state'] = \
                    layer_information['weighted_left_hidden_state'][0][j].tolist()

            mentions_per_word_left = mentions[sentence_index][j]
            polarities_per_word_left = polarities[sentence_index][j]
            relations_per_word_left = relations[sentence_index][j]
            pos_tags_per_word_left = pos_tags[sentence_index][j]

            lemmatized_sentence[sentence_index][j]

        n_right_words = x_right_sen_len[index]

        end_index = aspects[index][-1]

        for j in range(n_right_words):
            right_word_embeddings.append(x_right_part[index][j].tolist())
            right_hidden_states.append(tr_layer_information['right_hidden_state'][0][j].tolist())

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                for i in range(self.neural_language_model.config.n_iterations_hop):
                    weighted_hidden_state['weighted_right_hidden_state' + str(i)].append(
                        tr_layer_information['weighted_right_hidden_state' + str(i)][0][j].tolist())
            else:
                weighted_hidden_state['weighted_right_hidden_state'].append(
                    tr_layer_information['weighted_right_hidden_state'][0][j].tolist())

            mentions_per_word_right.append(mentions[index][end_index + 1 + j])
            index_of_one = mentions[index][end_index + 1 + j].index(1)
            mentions_count_right[index_of_one] += 1

            polarities_per_word_right.append(polarities[index][end_index + 1 + j])
            index_of_one = polarities[index][end_index + 1 + j].index(1)
            polarities_count_right[index_of_one] += 1

            relations_per_word_right.append(relations[index][end_index + 1 + j])
            index_of_one = relations[index][end_index + 1 + j].index(1)
            relations_count_right[index_of_one] += 1

            pos_tags_per_word_right.append(pos_tags[index][end_index + 1 + j])
            index_of_one = pos_tags[index][end_index + 1 + j].index(1)
            pos_tags_count_right[index_of_one] += 1
