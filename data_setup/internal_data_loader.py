import os
import json
import numpy as np


class InternalDataLoader:

    def __init__(self, config):
        self.config = config
        self.word_list = []

        self.sentence_id_in_training = []
        self.total_word_in_training = []
        self.original_sentence_training = []
        self.lemmatized_training = []
        self.word_embeddings_training_all = []
        self.sentiment_distribution_training = []
        self.part_of_speech_training = []
        self.negation_in_training = []
        self.word_sentiments_in_training = []
        self.aspect_sentiments_in_training = []
        self.mentions_in_training = []
        self.word_polarities_training = []
        self.word_relations_training = []
        self.aspects_training = []
        self.aspect_indices_training = []
        self.polarity_matrix_training = []
        self.categories_matrix_training = []

        self.sentence_id_in_test = []
        self.total_word_in_test = []
        self.original_sentence_test = []
        self.lemmatized_test = []
        self.word_embeddings_test_all = []
        self.sentiment_distribution_test = []
        self.part_of_speech_test = []
        self.negation_in_test = []
        self.word_sentiments_in_test = []
        self.aspect_sentiments_in_test = []
        self.mentions_in_test = []
        self.word_polarities_test = []
        self.word_relations_test = []
        self.aspects_test = []
        self.aspect_indices_test = []
        self.polarity_matrix_test = []
        self.categories_matrix_test = []

    def load_internal_training_data(self, load_internal_file_name):

        if not os.path.isfile(load_internal_file_name):
            raise ("[!] Data %s not found" % load_internal_file_name)

        with open(load_internal_file_name, 'r') as file:
            for line in file:
                sentences = json.loads(line)

                for sentence in sentences:

                    for n_aspects in range(len(sentence['aspects'])):

                        number_of_words = sentence['lemmatized_sentence']
                        self.total_word_in_training.append(number_of_words)

                        self.original_sentence_training.append(sentence['original_sentence'])
                        lemmatized_sentence = sentence['lemmatized_sentence']
                        self.lemmatized_training.append(lemmatized_sentence)

                        for lemma in lemmatized_sentence:
                            self.word_list.append(lemma)

                        self.mentions_in_training.append(sentence['word_mentions'])
                        self.sentence_id_in_training.append(sentence['sentence_id'])
                        self.word_embeddings_training_all.append(sentence['word_embeddings'])
                        self.sentiment_distribution_training.append(sentence['sentiment_distribution'])
                        self.negation_in_training.append(sentence['negation_in_sentence'])
                        self.word_sentiments_in_training.append(sentence['word_sentiments'][n_aspects])
                        self.aspect_sentiments_in_training.append(sentence['aspect_sentiments'][n_aspects])
                        self.word_polarities_training.append(sentence['word_polarities'][n_aspects])
                        self.aspects_training.append(sentence['aspects'][n_aspects])
                        self.word_relations_training.append(sentence['aspect_relations'][n_aspects])
                        self.aspect_indices_training.append(sentence['aspect_indices'][n_aspects])
                        self.polarity_matrix_training.append(sentence['polarity_matrix'][n_aspects])
                        self.categories_matrix_training.append(sentence['category_matrix'][n_aspects])
                        self.part_of_speech_training.append(sentence['part_of_speech_tags'])

        self.setup_cross_val_indices(len(self.word_embeddings_training_all))

    def load_internal_test_data(self, load_internal_file_name):

        if not os.path.isfile(load_internal_file_name):
            raise ("[!] Data %s not found" % load_internal_file_name)

        with open(load_internal_file_name, 'r') as file:
            for line in file:
                sentences = json.loads(line)
                for sentence in sentences:

                    for n_aspects in range(len(sentence['aspects'])):

                        number_of_words = sentence['lemmatized_sentence']
                        self.total_word_in_test.append(number_of_words)

                        self.original_sentence_test.append(sentence['original_sentence'])
                        lemmatized_sentence = sentence['lemmatized_sentence']
                        self.lemmatized_test.append(lemmatized_sentence)

                        for lemma in lemmatized_sentence:
                            self.word_list.append(lemma)

                        self.mentions_in_test.append(sentence['word_mentions'])
                        self.sentence_id_in_test.append(sentence['sentence_id'])
                        self.word_embeddings_test_all.append(sentence['word_embeddings'])
                        self.sentiment_distribution_test.append(sentence['sentiment_distribution'])
                        self.negation_in_test.append(sentence['negation_in_sentence'])
                        self.word_sentiments_in_test.append(sentence['word_sentiments'][n_aspects])
                        self.aspect_sentiments_in_test.append(sentence['aspect_sentiments'][n_aspects])
                        self.word_polarities_test.append(sentence['word_polarities'][n_aspects])
                        self.aspects_test.append(sentence['aspects'][n_aspects])
                        self.word_relations_test.append(sentence['aspect_relations'][n_aspects])
                        self.aspect_indices_test.append(sentence['aspect_indices'][n_aspects])
                        self.polarity_matrix_test.append(sentence['polarity_matrix'][n_aspects])
                        self.categories_matrix_test.append(sentence['category_matrix'][n_aspects])
                        self.part_of_speech_test.append(sentence['part_of_speech_tags'])

    def setup_cross_val_indices(self, size):

        if not os.path.isfile(self.config.cross_validation_indices_training) \
                and not os.path.isfile(self.config.cross_validation_indices_validation):

            np.random.seed(self.config.seed)
            number_training_data = np.int(self.config.cross_validation_percentage * size)
            number_test_data = size - number_training_data

            training_indices = np.zeros((self.config.cross_validation_rounds, number_training_data), dtype=int)
            validation_indices = np.zeros((self.config.cross_validation_rounds, number_test_data), dtype=int)

            for i in range(self.config.cross_validation_rounds):
                a_range = np.arange(0, size, dtype=int)
                np.random.shuffle(a_range)

                training_indices[i] = a_range[:number_training_data]
                validation_indices[i] = a_range[number_training_data:size]

            cross_val_training_indices = {
                'cross_val_train': training_indices.tolist()
            }

            with open(self.config.cross_validation_indices_training, 'w') as outfile:
                json.dump(cross_val_training_indices, outfile, ensure_ascii=False)

            cross_val_validation_indices = {
                'cross_val_validation': validation_indices.tolist()
            }

            with open(self.config.cross_validation_indices_validation, 'w') as outfile:
                json.dump(cross_val_validation_indices, outfile, ensure_ascii=False)

    def get_indices_cross_validation(self):

        if not os.path.isfile(self.config.cross_validation_indices_training):
            raise ("[!] cross validation training set is not found, please run the internal data loader")

        if not os.path.isfile(self.config.cross_validation_indices_validation):
            raise ("[!] cross validation test set is not found, please run the internal data loader")

        with open(self.config.cross_validation_indices_training, 'r') as file:
            for line in file:
                line = json.loads(line)
                training_indices = line['cross_val_train']

        with open(self.config.cross_validation_indices_validation, 'r') as file:
            for line in file:
                line = json.loads(line)
                test_indices = line['cross_val_validation']

        return np.array(training_indices), np.array(test_indices)

    @staticmethod
    def batch_index(length, batch_size, is_shuffle=True):
        index = list(range(length))
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]

    def read_remaining_data(self, is_cross_val):

        if is_cross_val:

            if not os.path.isfile(self.config.remaining_data_cross_val):
                raise ("[!] Remaining cross val data set is not found, please run the ontology beforehand")

            with open(self.config.remaining_data_cross_val, 'r') as file:
                for line in file:
                    line = json.loads(line)
                    return np.array(line['cross_val_indices'])
        else:

            if not os.path.isfile(self.config.remaining_data):
                raise ("Remaining dataset is not found, please run the ontology beforehand")

            with open(self.config.remaining_data, 'r') as file:
                for line in file:
                    line = json.loads(line)
                    return np.array(line['test_set_indices'])

    def split_embeddings(self, word_embeddings, aspect_indices, max_sentence_length, max_target_length):

        number_of_sentences = np.shape(word_embeddings)

        left_part = np.zeros((number_of_sentences[0], max_sentence_length, self.config.embedding_dimension),
                             dtype=float)
        right_part = np.zeros((number_of_sentences[0], max_sentence_length, self.config.embedding_dimension),
                              dtype=float)
        target_part = np.zeros((number_of_sentences[0], max_target_length, self.config.embedding_dimension),
                               dtype=float)

        words_in_left_context = np.zeros(number_of_sentences[0], dtype=int)
        words_in_target = np.zeros(number_of_sentences[0], dtype=int)
        words_in_right_context = np.zeros(number_of_sentences[0], dtype=int)

        for index in range(number_of_sentences[0]):

            begin_index_aspect = aspect_indices[index][0]
            end_index_aspect = aspect_indices[index][-1]

            np_word_embeddings = np.array(word_embeddings[index])
            max_embeddings = np_word_embeddings.shape[0]

            words_in_left_context[index] = begin_index_aspect
            words_in_target[index] = (end_index_aspect - begin_index_aspect) + 1
            words_in_right_context[index] = max_embeddings - (end_index_aspect + 1)

            left_part[index][:words_in_left_context[index]] = np_word_embeddings[0:begin_index_aspect]
            target_part[index][:words_in_target[index]] = np_word_embeddings[begin_index_aspect:end_index_aspect + 1]
            right_part[index][:words_in_right_context[index]] = np_word_embeddings[end_index_aspect + 1:]

        return left_part, target_part, right_part, words_in_left_context, words_in_target, words_in_right_context
