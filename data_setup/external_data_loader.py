import os
import xml.etree.ElementTree as elementTree
import nltk
import numpy as np
from data_setup.ontology_tagging import OntologyTagging
from autocorrect import spell
import json


class ExternalDataLoader:

    def __init__(self, config):
        self.word_dictionary = {}
        self.ontology_tagging = OntologyTagging()
        self.config = config

    def load_external_data(self, load_external_file_name, write_internal_file_name):

        if not os.path.isfile(load_external_file_name):
            raise ("[!] Data %s not found" % load_external_file_name)

        xml_tree = elementTree.parse(load_external_file_name)
        root = xml_tree.getroot()

        opinion_counter = 0
        with open(write_internal_file_name, 'w') as outfile:
            for sentence in root.iter('sentence'):

                sentence_id = sentence.get('id')
                original_sentence = sentence.find('text').text

                print("original_sentence ", original_sentence)
                tokenized_sentence = nltk.word_tokenize(original_sentence)

                print("tokenized_sentence ", tokenized_sentence)

                aspects = []
                aspect_indices = []
                polarities = []
                polarity_matrix = []
                categories = []
                category_matrix = []

                for opinions in sentence.iter('Opinions'):

                    for opinion in opinions.findall('Opinion'):

                        aspect = opinion.get('target')
                        if aspect != "NULL":
                            opinion_counter += 1

                            aspects.append(aspect)
                            category = opinion.get('category')
                            polarity = opinion.get('polarity')
                            categories.append(category)
                            polarities.append(polarity)

                            tokenized_aspect = nltk.word_tokenize(aspect)
                            aspect_indices.append(self.get_aspect_indices(tokenized_aspect, tokenized_sentence))
                            polarity_matrix.append(self.get_polarity_number(polarity))
                            category_matrix.append(self.get_category_number(category))

                if len(aspects) != 0:

                    lemmatized_sentence, part_of_speech_sentence = self.lemmatize_and_pos_tagging(tokenized_sentence)
                    # ontology_classes_sentence = self.ontology_tagging.ontology_classes_tagging(lemmatized_sentence)
                    # word_mention_sentence = self.ontology_tagging.mention_tagging(ontology_classes_sentence)
                    # word_polarity_sentence, aspect_relation_sentence = self.ontology_tagging.\
                    #     polarity_and_aspect_relation_tagging(ontology_classes_sentence, aspect_indices, categories)
                    #
                    # word_embedding_sentence = self.compute_word_embeddings(lemmatized_sentence)

                    dict_sentence = {
                        'sentence_id': sentence_id,
                        'original_sentence': original_sentence,
                        'lemmatized_sentence': lemmatized_sentence,
                        # 'word_embeddings': word_embedding_sentence,
                        # 'bag_of_words': [],
                        # 'part_of_speech_tags': part_of_speech_sentence,
                        # 'word_polarities': word_polarity_sentence,
                        # 'word_mentions': word_mention_sentence,
                        # 'aspect_relations': aspect_relation_sentence,
                        # 'aspects': aspects,
                        # 'aspect_indices': aspect_indices,
                        # 'polarities': polarities,
                        # 'polarity_matrix': polarity_matrix,
                        # 'categories': categories,
                        # 'category_matrix': category_matrix
                    }

                    json.dump(dict_sentence, outfile, ensure_ascii=False)
                    outfile.write('\n')

    @staticmethod
    def get_polarity_number(polarity):

        if polarity == "positive":
            return [1, 0, 0]
        elif polarity == "neutral":
            return [0, 1, 0]
        elif polarity == "negative":
            return [0, 0, 1]
        else:
            raise Exception("Polarity ", polarity, " is not in the sentence.")

    @staticmethod
    def get_category_number(category):

        if category == "AMBIENCE#GENERAL":
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "DRINKS#PRICES":
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "DRINKS#QUALITY":
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "DRINKS#STYLE_OPTIONS":
            return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#GENERAL":
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#PRICES":
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#QUALITY":
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#STYLE_OPTIONS":
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif category == "LOCATION#GENERAL":
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif category == "RESTAURANT#GENERAL":
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif category == "RESTAURANT#MISCELLANEOUS":
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif category == "RESTAURANT#PRICES":
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif category == "SERVICE#GENERAL":
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
            raise Exception("Category ", category, " is not in the sentence.")

    @staticmethod
    def get_aspect_indices(aspect, sentence):

        number_words_in_aspect = len(aspect)
        number_words_in_sentence = len(sentence)

        for i in range(number_words_in_sentence):

            if aspect[0] == sentence[i]:
                return list(range(i, i+number_words_in_aspect))

        raise Exception("Aspect ", aspect, " is not in the sentence ", sentence)

    def compute_word_embeddings(self, sentence):

        number_words_in_sentence = len(sentence)
        word_embeddings = np.random.normal(0, 0.05, [number_words_in_sentence, 300])

        with open(self.config.glove_embeddings, 'r', encoding="utf8") as f:
            for line in f:
                word_embedding = line.strip().split()

                if word_embedding[0] in sentence:
                    word_index = sentence.index(word_embedding[0])
                    word_embeddings[word_index] = list(map(float, word_embedding[1:]))

        return word_embeddings.tolist()

    @staticmethod
    def lemmatize_and_pos_tagging(sentence):

        number_words_in_sentence = len(sentence)
        processed_sentence = []
        part_of_speech_sentence = []
        lemmatized_sententce = []

        punctuation_and_numbers = ['–', '(', ')', '?', ':', ';', ',', '.', '!', '/', '"', '’', '*', '$',
                                   '&', '%', '@', '#', '^', '!', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        punctuations = ['–', '(', ')', '?', ':', ';', ',', '.', '!', '/', '"', '’', '*', '$', '&', '%', '@', '#', '^',
                        '!', '\'', '-']
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z', '']

        for word_index in range(number_words_in_sentence):

            list_of_word = list(sentence[word_index].lower())

            if len(sentence[word_index]) > 1:

                for char_index in range(len(list_of_word)-1):

                    if list_of_word[char_index] in alphabet and list_of_word[char_index + 1] in punctuation_and_numbers:
                        list_of_word[char_index+1] = ''
                    elif list_of_word[char_index] in punctuation_and_numbers and list_of_word[char_index + 1] in alphabet:
                        list_of_word[char_index] = ''

            word = "".join(list_of_word)
            if word not in punctuations:
                word = spell(word)
            processed_sentence.append(word)

        pos_tags = nltk.pos_tag(processed_sentence)

        wordnet_lemmatizer = nltk.WordNetLemmatizer()

        for word_index in range(number_words_in_sentence):
            pos_tag = pos_tags[word_index][1]
            part_of_speech_sentence.append(pos_tag)

            if pos_tag.startswith('V'):     # Verb
                lemma_of_word = wordnet_lemmatizer.lemmatize(processed_sentence[word_index], 'v')
            elif pos_tag.startswith('J'):   # Adjective
                lemma_of_word = wordnet_lemmatizer.lemmatize(processed_sentence[word_index], 'a')
            elif pos_tag.startswith('R'):   # Adverb
                lemma_of_word = wordnet_lemmatizer.lemmatize(processed_sentence[word_index], 'r')
            elif pos_tag.startswith('N'):   # Noun
                lemma_of_word = wordnet_lemmatizer.lemmatize(processed_sentence[word_index], 'n')
            else:                           # Otherwise
                lemma_of_word = wordnet_lemmatizer.lemmatize(processed_sentence[word_index])
            lemmatized_sententce.append(lemma_of_word)

        return lemmatized_sententce, part_of_speech_sentence
