import os
import json
import numpy as np


class InternalDataLoader:

    def __init__(self, config):
        self.config = config

        self.total_word_in_training = []
        self.lemmatized_training = []
        self.word_embeddings_training_all = []
        self.word_embeddings_training_only = []
        self.bag_of_words_training = []
        self.part_of_speech_training = []
        self.negation_in_training = []
        self.word_mentions_training = []
        self.word_polarities_training = []
        self.word_relations_training = []
        self.aspect_indices_training = []
        self.polarity_matrix_training = []
        self.categories_matrix_training = []

        self.total_word_in_test = []
        self.lemmatized_test = []
        self.word_embeddings_test_all = []
        self.word_embeddings_test_only = []
        self.bag_of_words_test = []
        self.part_of_speech_test = []
        self.negation_in_test = []
        self.word_mentions_test = []
        self.word_polarities_test = []
        self.word_relations_test = []
        self.aspect_indices_test = []
        self.polarity_matrix_test = []
        self.categories_matrix_test = []

    def load_internal_training_data(self, load_internal_file_name):

        if not os.path.isfile(load_internal_file_name):
            raise ("[!] Data %s not found" % load_internal_file_name)

        with open(load_internal_file_name, 'r') as file:
            for line in file:
                sentence = json.loads(line)

                self.word_embeddings_training_only.append(sentence['word_embeddings'])
                self.part_of_speech_training.append(sentence['part_of_speech_tags'])
                self.negation_in_training.append(sentence['word_negations'])

                for n_aspects in range(len(sentence['aspects'])):
                    number_of_words = sentence['lemmatized_sentence']
                    self.total_word_in_training.append(number_of_words)
                    self.lemmatized_training.append(sentence['lemmatized_sentence'])
                    self.word_embeddings_training_all.append(sentence['word_embeddings'])
                    self.bag_of_words_training.append(sentence['bag_of_words'])
                    self.word_mentions_training.append(sentence['word_mentions'][n_aspects])
                    self.word_polarities_training.append(sentence['word_polarities'][n_aspects])
                    self.word_relations_training.append(sentence['aspect_relations'][n_aspects])
                    self.aspect_indices_training.append(sentence['aspect_indices'][n_aspects])
                    self.polarity_matrix_training.append(sentence['polarity_matrix'][n_aspects])
                    self.categories_matrix_training.append(sentence['category_matrix'][n_aspects])

    def load_internal_test_data(self, load_internal_file_name):

        if not os.path.isfile(load_internal_file_name):
            raise ("[!] Data %s not found" % load_internal_file_name)

        with open(load_internal_file_name, 'r') as file:
            for line in file:
                sentence = json.loads(line)

                self.word_embeddings_test_only.append(sentence['word_embeddings'])
                self.part_of_speech_test.append(sentence['part_of_speech_tags'])
                self.negation_in_test.append(sentence['word_negations'])

                for n_aspects in range(len(sentence['aspects'])):
                    number_of_words = sentence['lemmatized_sentence']
                    self.total_word_in_test.append(number_of_words)
                    self.lemmatized_test.append(sentence['lemmatized_sentence'])
                    self.word_embeddings_test_all.append(sentence['word_embeddings'])
                    self.bag_of_words_test.append(sentence['bag_of_words'])
                    self.word_mentions_test.append(sentence['word_mentions'][n_aspects])
                    self.word_polarities_test.append(sentence['word_polarities'][n_aspects])
                    self.word_relations_test.append(sentence['aspect_relations'][n_aspects])
                    self.aspect_indices_test.append(sentence['aspect_indices'][n_aspects])
                    self.polarity_matrix_test.append(sentence['polarity_matrix'][n_aspects])
                    self.categories_matrix_test.append(sentence['category_matrix'][n_aspects])

    def get_random_indices_for_cross_validation(self, cross_validation_rounds, size_of_dataset):

        np.random.seed(self.config.seed)

        number_training_data = np.int(self.config.cross_validation_percentage * size_of_dataset)
        number_test_data = size_of_dataset - number_training_data

        training_indices = np.zeros(cross_validation_rounds, number_training_data)
        test_indices = np.zeros(cross_validation_rounds, number_test_data)

        for i in range(cross_validation_rounds):

            a_range = np.arange(0, size_of_dataset)
            np.random.shuffle(a_range)

            training_indices[i] = a_range[:number_training_data]
            test_indices[i] = a_range[number_training_data:size_of_dataset]

        return training_indices, test_indices

    @staticmethod
    def split_embeddings_in_left_target_right(word_embeddings, aspect_indices, max_sentence_length, max_target_length):

        number_of_sentences = np.shape(word_embeddings)

        left_part = np.zeros((number_of_sentences[0], max_sentence_length))
        right_part = np.zeros((number_of_sentences[0], max_sentence_length))
        target_part = np.zeros((number_of_sentences[0], max_target_length))

        words_in_left_context = np.zeros(number_of_sentences[0])
        words_in_target = np.zeros(number_of_sentences[0])
        words_in_right_context = np.zeros(number_of_sentences[0])

        for index in range(number_of_sentences[0]):

            begin_index_aspect = aspect_indices[index][0]
            end_index_aspect = aspect_indices[index][-1]

            words_in_left_context[index] = begin_index_aspect
            words_in_target[index] = end_index_aspect - begin_index_aspect + 1
            words_in_right_context[index] = max_sentence_length - end_index_aspect + 1

            left_part[index][:words_in_left_context] = word_embeddings[index][0:begin_index_aspect]
            target_part[index][:words_in_target] = word_embeddings[index][begin_index_aspect:end_index_aspect + 1]
            right_part[index][:words_in_right_context] = word_embeddings[index][end_index_aspect + 1:]

        return left_part, target_part, right_part, words_in_left_context, words_in_target, words_in_right_context

    def batch_index(self, length, batch_size):

        index = np.arange(length)
        np.random.seed(self.config.seed)
        np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]
