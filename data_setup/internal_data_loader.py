import os
import json
import numpy as np


class InternalDataLoader:

    def __init__(self, config):
        self.config = config

        self.lemmatized_training = []
        self.word_embeddings_training_all = []
        self.word_embeddings_training_only = []
        self.bag_of_words_training = []
        self.part_of_speech_training = []
        self.word_mentions_training = []
        self.word_polarities_training = []
        self.word_relations_training = []
        self.aspect_indices_training = []
        self.polarity_matrix_training = []
        self.categories_matrix_training = []

        self.lemmatized_test = []
        self.word_embeddings_test_all = []
        self.word_embeddings_test_only = []
        self.bag_of_words_test= []
        self.part_of_speech_test = []
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

                for n_aspects in range(len(sentence['aspects'])):
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

                for n_aspects in range(len(sentence['aspects'])):
                    self.lemmatized_test.append(sentence['lemmatized_sentence'])
                    self.word_embeddings_test_all.append(sentence['word_embeddings'])
                    self.bag_of_words_test.append(sentence['bag_of_words'])
                    self.word_mentions_test.append(sentence['word_mentions'][n_aspects])
                    self.word_polarities_test.append(sentence['word_polarities'][n_aspects])
                    self.word_relations_test.append(sentence['aspect_relations'][n_aspects])
                    self.aspect_indices_test.append(sentence['aspect_indices'][n_aspects])
                    self.polarity_matrix_test.append(sentence['polarity_matrix'][n_aspects])
                    self.categories_matrix_test.append(sentence['category_matrix'][n_aspects])

    def get_random_indices_for_cross_validation(self, cross_validation_rounds):

        np.random.seed(self.config.seed)

        number_entire_data = len(self.word_embeddings_training_all)
        number_training_data = self.config.cross_validation_percentage_training * number_entire_data
        number_test_data = number_entire_data - number_training_data

        training_indices = np.zeros(cross_validation_rounds, number_training_data)
        test_indices = np.zeros(cross_validation_rounds, number_test_data)

        for i in range(cross_validation_rounds):

            a_range = np.arange(0, number_entire_data)
            np.random.shuffle(a_range)

            training_indices[i] = a_range[:number_training_data]
            test_indices[i] = a_range[number_training_data:number_entire_data]

        return training_indices, test_indices