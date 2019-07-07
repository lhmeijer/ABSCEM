import os
import json
import numpy as np
import matplotlib.pyplot as plt
from nltk.parse import CoreNLPParser


class SentencePlot:

    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_polarity(polarity_number):

        if polarity_number == 0:
            return "positive"
        elif polarity_number == 1:
            return "neutral"
        elif polarity_number == 2:
            return "negative"
        else:
            raise Exception("Polarity ", polarity_number, " is not in the sentence.")

    def plot_final_results(self, x, y, polarity_index, name_method, index, id, name_nlm, intercept):


        max_value = np.max(x[polarity_index]) + np.max(x[polarity_index]) * 0.1
        min_value = np.min(x[polarity_index]) - np.max(x[polarity_index]) * 0.1
        n_of_relevance_words = x.shape[1]

        y_pos = np.arange(n_of_relevance_words)

        fig = plt.figure(figsize=(12, 6.5))
        fig.set_size_inches(13.5, 9.5)
        ax = fig.add_subplot(111)

        ax.tick_params(length=6, axis='y', width=1, labelsize=45)
        ax.tick_params(length=6, axis='x', width=1, labelsize=25)

        ax.set_xlim([min_value, max_value])
        ax.axvline(intercept[polarity_index], color='grey', alpha=0.40)

        title = 'Contribution towards the ' + str(self.get_polarity(polarity_index)) + ' sentiment value'
        ax.set_xlabel(title, fontsize=20)

        bar = ax.barh(y_pos, x[polarity_index], align='center', height=0.6, alpha=0.5,
                               tick_label=y[polarity_index])

        plt.subplots_adjust(left=0.54)
        file = self.config.get_sentence_results(name_method, index, id, name_nlm, self.get_polarity(polarity_index))

        plt.savefig(file)


class SingleSentencePlot(SentencePlot):

    def __init__(self, config, nl_model):
        super().__init__(config)
        self.nl_model = nl_model
        self.server_url = 'http://localhost:9000'
        self.parser = CoreNLPParser(url=self.server_url)

    def plot(self, max_relevance_words_in_plot, sentence_id="1004293:0"):

        results_file = self.config.get_file_of_results(self.nl_model.config.name_of_model)

        if not os.path.isfile(results_file):
            raise ("[!] Data %s not found" % results_file)

        index = -1
        original_sentences = self.nl_model.internal_data_loader.original_sentence_training

        with open(results_file, 'r') as file:
            for line in file:
                sentences = json.loads(line)
                sentences.pop(0)
                for sentence in sentences:

                    index += 1

                    if sentence['sentence_id'] == sentence_id:

                        sentence_index = sentence['sentence_index']

                        tokenized_sentence = list(self.parser.tokenize(original_sentences[sentence_index]))
                        print("tokenized_sentence ", tokenized_sentence)

                        max_word_relevance = np.full((3, max_relevance_words_in_plot), -1.)
                        relevant_words = np.empty([3, max_relevance_words_in_plot], dtype=object)

                        for attribute_dict in sentence['subsets_word_relevance_linear_regression']:

                            for i in range(3):

                                min_value = max_word_relevance[i].min()
                                min_index = max_word_relevance[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance[i][min_index] = attribute_dict[str(i)]

                                    n_indices = len(attribute_dict['indices_attribute'])
                                    word = str(tokenized_sentence[attribute_dict['indices_attribute'][0]])

                                    for w in range(1, n_indices):
                                        word = word + ' ' + \
                                               str(tokenized_sentence[attribute_dict['indices_attribute'][w]])

                                    relevant_words[i][min_index] = word

                        intercept = sentence['intercepts_slr']

                        self.plot_final_results(max_word_relevance, relevant_words,
                                                np.argmax(sentence['prediction']), "OWN", sentence_index,
                                                sentence_id, self.nl_model.config.name_of_model, intercept)
                        self.plot_final_results(max_word_relevance, relevant_words,
                                                np.argmax(sentence['aspect_polarity_matrix']), "OWN", sentence_index,
                                                sentence_id, self.nl_model.config.name_of_model, intercept)

                        max_word_relevance = np.full((3, max_relevance_words_in_plot), -1.)
                        relevant_words = np.empty([3, max_relevance_words_in_plot], dtype=object)

                        for attribute_dict in sentence['subsets_word_relevance_pred_difference']:

                            for i in range(3):

                                min_value = max_word_relevance[i].min()
                                min_index = max_word_relevance[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance[i][min_index] = attribute_dict[str(i)]

                                    indices = attribute_dict['indices_attribute']
                                    if type(indices) == list:

                                        n_indices = len(attribute_dict['indices_attribute'])
                                        word = str(tokenized_sentence[attribute_dict['indices_attribute'][0]])

                                        for w in range(1, n_indices):
                                            word = word + ' ' + \
                                                   str(tokenized_sentence[attribute_dict['indices_attribute'][w]])
                                    else:
                                        word = str(tokenized_sentence[indices])

                                    relevant_words[i][min_index] = word

                        intercept = np.zeros(len(sentence['prediction']))

                        self.plot_final_results(max_word_relevance, relevant_words,
                                                np.argmax(sentence['prediction']), "LACE", sentence_index,
                                                sentence_id, self.nl_model.config.name_of_model, intercept)
                        self.plot_final_results(max_word_relevance, relevant_words,
                                                np.argmax(sentence['aspect_polarity_matrix']), "LACE", sentence_index,
                                                sentence_id, self.nl_model.config.name_of_model, intercept)

                        max_word_relevance = np.full((3, max_relevance_words_in_plot), -1.)
                        relevant_words = np.empty([3, max_relevance_words_in_plot], dtype=object)

                        for attribute_dict in sentence['word_relevance_linear_regression']:

                            for i in range(3):

                                min_value = max_word_relevance[i].min()
                                min_index = max_word_relevance[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance[i][min_index] = attribute_dict[str(i)]
                                    relevant_words[i][min_index] = str(tokenized_sentence[attribute_dict['indices_attribute'][0]])

                        intercept = sentence['intercepts_slr']

                        self.plot_final_results(max_word_relevance, relevant_words,
                                                np.argmax(sentence['prediction']), "LIME", sentence_index,
                                                sentence_id, self.nl_model.config.name_of_model, intercept)
                        self.plot_final_results(max_word_relevance, relevant_words,
                                                np.argmax(sentence['aspect_polarity_matrix']), "LIME", sentence_index,
                                                sentence_id, self.nl_model.config.name_of_model, intercept)
