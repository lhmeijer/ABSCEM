import os
import xml.etree.ElementTree as elementTree
import nltk
import numpy as np
import requests
from data_setup.ontology_tagging import OntologyTagging
from autocorrect import spell
import json
from nltk.parse import CoreNLPParser
from nltk.corpus import wordnet
from nltk.parse.corenlp import CoreNLPDependencyParser


class ExternalDataLoader:

    def __init__(self, config):
        self.ontology_tagging = OntologyTagging()
        self.config = config
        self.word_dictionary = self.compute_all_embeddings()
        self.server_url = 'http://localhost:9000'
        self.parser = CoreNLPParser(url=self.server_url)
        self.core_nlp_dependency_parser = CoreNLPDependencyParser(url=self.server_url)

        self.positive_counter = 0
        self.negative_counter = 0
        self.neutral_counter = 0

        self.ambience_general_counter = 0
        self.drinks_prices_counter = 0
        self.drinks_quality_counter = 0
        self.drinks_style_options_counter = 0
        self.food_general = 0
        self.food_prices = 0
        self.food_quality = 0
        self.food_category = 0
        self.location_general = 0
        self.location_general = 0
        self.restaurant_general = 0
        self.restaurant_miscellaneous = 0
        self.restaurant_prices = 0
        self.service_general = 0

    def load_external_data(self, load_external_file_name, write_internal_file_name):

        if not os.path.isfile(load_external_file_name):
            raise ("[!] Data %s not found" % load_external_file_name)

        xml_tree = elementTree.parse(load_external_file_name)
        root = xml_tree.getroot()

        opinion_counter = 0

        all_sentences = []

        for sentence in root.iter('sentence'):

            sentence_id = sentence.get('id')

            original_sentence = sentence.find('text').text

            tokenized_sentence = list(self.parser.tokenize(original_sentence))

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

                        tokenized_aspect = list(self.parser.tokenize(aspect))
                        aspect_indices.append(self.get_aspect_indices(tokenized_aspect, tokenized_sentence))
                        polarity_matrix.append(self.get_polarity_number(polarity))
                        category_matrix.append(self.get_category_number(category))

            if len(aspects) != 0:

                sentiment_distribution = self.annotate(original_sentence, properties={"annotators": "sentiment",
                                                                                      "outputFormat": "json", })

                processed_sentence = self.process_characters(tokenized_sentence)

                lemmatized_sentence, part_of_speech_sentence, aspect_dependencies, sentence_negation = \
                    self.lemmatize_and_pos_tagging(processed_sentence, aspect_indices)

                ontology_classes_sentence = self.ontology_tagging.ontology_classes_tagging(lemmatized_sentence)

                word_mention_sentence = self.ontology_tagging.mention_tagging(ontology_classes_sentence)

                word_polarity_sentence, aspect_relation_sentence = self.ontology_tagging.\
                    polarity_and_aspect_relation_tagging(ontology_classes_sentence, aspect_indices, categories,
                                                         aspect_dependencies)

                word_embedding_sentence = self.compute_word_embeddings(lemmatized_sentence)

                dict_sentence = {
                    'sentence_id': sentence_id,
                    'original_sentence': original_sentence,
                    'lemmatized_sentence': lemmatized_sentence,
                    'sentiment_distribution': sentiment_distribution,
                    'part_of_speech_tags': part_of_speech_sentence,
                    'negation_in_sentence': sentence_negation,
                    'word_polarities': word_polarity_sentence,
                    'word_mentions': word_mention_sentence,
                    'aspect_relations': aspect_relation_sentence,
                    'aspects': aspects,
                    'aspect_indices': aspect_indices,
                    'polarities': polarities,
                    'polarity_matrix': polarity_matrix,
                    'categories': categories,
                    'category_matrix': category_matrix,
                    'word_embeddings': word_embedding_sentence
                }
                all_sentences.append(dict_sentence)

        print("Number of data points in the sample ", opinion_counter)
        print("Number of positive data points ", self.positive_counter)
        print("Number of neutral data points ", self.neutral_counter)
        print("Number of negative data points ", self.negative_counter)

        with open(write_internal_file_name, 'w') as outfile:
            json.dump(all_sentences, outfile, ensure_ascii=False)

    def get_polarity_number(self, polarity):

        if polarity == "positive":
            self.positive_counter += 1
            return [1, 0, 0]
        elif polarity == "neutral":
            self.neutral_counter += 1
            return [0, 1, 0]
        elif polarity == "negative":
            self.negative_counter += 1
            return [0, 0, 1]
        else:
            raise Exception("Polarity ", polarity, " is not in the sentence.")

    def get_category_number(self, category):

        if category == "AMBIENCE#GENERAL":
            self.ambience_general_counter += 1
            return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "DRINKS#PRICES":
            self.drinks_prices_counter += 1
            return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "DRINKS#QUALITY":
            self.drinks_quality_counter += 1
            return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "DRINKS#STYLE_OPTIONS":
            self.drinks_style_options_counter += 1
            return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#GENERAL":
            self.food_general += 1
            return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#PRICES":
            self.food_prices += 1
            return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#QUALITY":
            self.food_quality += 1
            return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif category == "FOOD#STYLE_OPTIONS":
            self.food_category += 1
            return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif category == "LOCATION#GENERAL":
            self.location_general += 1
            return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif category == "RESTAURANT#GENERAL":
            self.restaurant_general += 1
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif category == "RESTAURANT#MISCELLANEOUS":
            self.restaurant_miscellaneous += 1
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif category == "RESTAURANT#PRICES":
            self.restaurant_prices += 1
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif category == "SERVICE#GENERAL":
            self.service_general += 1
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

    def compute_all_embeddings(self):

        word_dictionary = {}

        with open(self.config.glove_embeddings, 'r', encoding="utf8") as f:
            for line in f:
                word_embedding = line.strip().split()
                word_dictionary[word_embedding[0]] = list(map(float, word_embedding[1:]))

        return word_dictionary

    def compute_word_embeddings(self, sentence):

        number_words_in_sentence = len(sentence)
        word_embeddings = np.random.normal(0, 0.05, [number_words_in_sentence, 300])

        for word_index in range(number_words_in_sentence):

            if sentence[word_index] in self.word_dictionary:
                word_embeddings[word_index] = self.word_dictionary[sentence[word_index]]

        return word_embeddings.tolist()

    @staticmethod
    def process_characters(sentence):

        number_words_in_sentence = len(sentence)
        processed_sentence = []

        punctuation_and_numbers = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', '*', '$',
                                   '&', '%', '@', '#', '^', '!', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z', '']
        punctuation_to_be_replaced = {'–': '-', '’': '\''}

        for word_index in range(number_words_in_sentence):

            list_of_word = list(sentence[word_index].lower())

            for char_index in range(len(list_of_word)-1):

                if list_of_word[char_index] in punctuation_to_be_replaced:
                    list_of_word[char_index] = punctuation_to_be_replaced[list_of_word[char_index]]

                if list_of_word[char_index] in alphabet and list_of_word[char_index + 1] in punctuation_and_numbers:
                    list_of_word[char_index+1] = ''
                elif list_of_word[char_index] in punctuation_and_numbers and list_of_word[char_index + 1] in alphabet:
                    list_of_word[char_index] = ''

            word = "".join(list_of_word)
            if word == '.' and sentence[word_index-1] == '.':
                pass
            else:
                if word == '.......' or word == '....' or word == '.....' or word == '......' or word == '..':
                    word = '...'
                processed_sentence.append(word)
        return processed_sentence

    def lemmatize_and_pos_tagging(self, sentence, aspect_indices):

        punctuations = ['–', '(', ')', '?', ':', ';', ',', '.', '!', '/', '"', '’', '*', '$', '&', '%', '@', '#', '^',
                        '!', '\'', '-']

        parses = self.core_nlp_dependency_parser.parse(sentence)
        dependencies = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()]
                        for parse in parses][0]

        wordnet_lemmatizer = nltk.WordNetLemmatizer()
        part_of_speech_sentence = list(range(len(sentence)))
        lemmatized_sentence = list(range(len(sentence)))
        aspects_dependencies = [['no'] * len(sentence) for i in range(len(aspect_indices))]

        backup_sentence = sentence.copy()
        interesting_translates = {'-LRB-': '(', '-RRB-': ')', '2\xa01/2': '2 1/2', "''": '"', ':-RRB-': ':)'}

        sentence_negations = []

        for dependency in dependencies:

            words = [dependency[0][0], dependency[2][0]]
            part_of_speech = [dependency[0][1], dependency[2][1]]

            if words[0] in interesting_translates:
                words[0] = interesting_translates[words[0]]
            if words[1] in interesting_translates:
                words[1] = interesting_translates[words[1]]

            range_list = [0, 1]
            if words[0] in sentence:
                index_of_word1 = sentence.index(words[0])
                sentence[index_of_word1] = ''
            else:
                index_of_word1 = backup_sentence.index(words[0])
                range_list = [1]

            if words[1] in sentence:
                index_of_word2 = sentence.index(words[1])
                sentence[index_of_word2] = ''
            else:
                index_of_word2 = backup_sentence.index(words[1])
                range_list = [0]

            word_indices = [index_of_word1, index_of_word2]

            if dependency[1] == 'neg':
                sentence_negations.append(word_indices)

            for aspect_index in range(len(aspect_indices)):

                if index_of_word1 in aspect_indices[aspect_index] and index_of_word2 not in \
                        aspect_indices[aspect_index]:
                    aspects_dependencies[aspect_index][index_of_word2] = dependency[1]
                elif index_of_word1 not in aspect_indices[aspect_index] and index_of_word2 in \
                        aspect_indices[aspect_index]:
                    aspects_dependencies[aspect_index][index_of_word1] = dependency[1]
                elif index_of_word1 in aspect_indices[aspect_index] and index_of_word2 in aspect_indices[aspect_index]:
                    if aspects_dependencies[aspect_index][index_of_word1] == 'no':
                        aspects_dependencies[aspect_index][index_of_word1] = dependency[1]
                    else:
                        aspects_dependencies[aspect_index][index_of_word2] = dependency[1]

            for i in range_list:

                if part_of_speech[i].startswith('V'):        # Verb
                    part_of_speech_sentence[word_indices[i]] = [1, 0, 0, 0, 0]
                    word = spell(words[i])
                    lemma = wordnet_lemmatizer.lemmatize(word, wordnet.VERB)
                    lemmatized_sentence[word_indices[i]] = lemma
                elif part_of_speech[i].startswith('J'):      # Adjective
                    part_of_speech_sentence[word_indices[i]] = [0, 1, 0, 0, 0]
                    word = spell(words[i])
                    lemma = wordnet_lemmatizer.lemmatize(word, wordnet.ADJ)
                    lemmatized_sentence[word_indices[i]] = lemma
                elif part_of_speech[i].startswith('R'):      # Adverb
                    part_of_speech_sentence[word_indices[i]] = [0, 0, 1, 0, 0]
                    word = spell(words[i])
                    lemma = wordnet_lemmatizer.lemmatize(word, wordnet.ADV)
                    lemmatized_sentence[word_indices[i]] = lemma
                elif part_of_speech[i].startswith('N'):      # Noun
                    part_of_speech_sentence[word_indices[i]] = [0, 0, 0, 1, 0]
                    word = spell(words[i])
                    lemma = wordnet_lemmatizer.lemmatize(word, wordnet.NOUN)
                    lemmatized_sentence[word_indices[i]] = lemma
                else:                                       # Otherwise
                    part_of_speech_sentence[word_indices[i]] = [0, 0, 0, 0, 1]
                    if words[i] not in punctuations:
                        words[i] = spell(words[i])
                    lemma = wordnet_lemmatizer.lemmatize(words[i])
                    lemmatized_sentence[word_indices[i]] = lemma

        return lemmatized_sentence, part_of_speech_sentence, aspects_dependencies, sentence_negations

    def annotate(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n' 
                            '$ cd stanford-corenlp-full-2018-02-27/ \n'
                            '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')
        data = text.encode()
        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data=data, headers={'Connection': 'close'})
        output = r.text

        char_index1 = output.index("sentimentDistribution")
        char_index2 = output.index("sentimentTree")
        distribution = output[(char_index1 - 1):(char_index2 - 2)]

        new_distribution = []
        word = []

        for char_index in range(len(distribution)):

            if distribution[char_index].isnumeric():
                word.append(distribution[char_index])
            elif distribution[char_index] == ',' and len(word) == 1:
                word.append('.')
            elif (distribution[char_index] == ',' or distribution[char_index] == ']') and len(word) != 1:
                number = float("".join(word))
                new_distribution.append(number)
                word = []

        return new_distribution
