import os
import json
import numpy as np
import matplotlib.pyplot as plt


class SentencePlot:

    @staticmethod
    def get_polarity_number(polarity):

        if polarity == "positive":
            return 0
        elif polarity == "neutral":
            return 1
        elif polarity == "negative":
            return 2
        else:
            raise Exception("Polarity ", polarity, " is not in the sentence.")

    @staticmethod
    def get_category_number(category):

        if category == "AMBIENCE#GENERAL":
            return 0
        elif category == "DRINKS#PRICES":
            return 1
        elif category == "DRINKS#QUALITY":
            return 2
        elif category == "DRINKS#STYLE_OPTIONS":
            return 3
        elif category == "FOOD#GENERAL":
            return 4
        elif category == "FOOD#PRICES":
            return 5
        elif category == "FOOD#QUALITY":
            return 6
        elif category == "FOOD#STYLE_OPTIONS":
            return 7
        elif category == "LOCATION#GENERAL":
            return 8
        elif category == "RESTAURANT#GENERAL":
            return 9
        elif category == "RESTAURANT#MISCELLANEOUS":
            return 10
        elif category == "RESTAURANT#PRICES":
            return 11
        elif category == "SERVICE#GENERAL":
            return 12
        else:
            raise Exception("Category ", category, " is not in the sentence.")

    @staticmethod
    def plot_final_results(x, y, title):

        max_value = np.max(x) + 0.02
        min_value = np.min(x) - 0.02
        n_of_relevance_words = x.shape[1]

        y_pos = np.arange(n_of_relevance_words)

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        # fig.subplots_adjust(left=0.125, right=0.5)
        # plt.yticks(y_pos, y[0], fontsize=15, fontweight=100, fontstretch=500)
        # ax.set_yticklabels(y[0], fontsize=15, fontweight=100, fontstretch=500)

        ax.tick_params(length=6, width=1)

        ax.set_title("title van de plot")

        ax.set_xlim([min_value, max_value])
        ax.axvline(0, color='grey', alpha=0.50)

        ax.set_xlabel(title + ": positive")

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # positive plot
        bar_positive = ax.barh(y_pos, x[0], align='center', height=0.80, alpha=0.5, tick_label=y[0])
        plt.show()

        # neutral plot
        fig2 = plt.figure(figsize=(9, 6))
        ax = fig2.add_subplot(111)
        ax.tick_params(length=6, width=1)

        ax.set_title("title van de plot")

        ax.set_xlim([min_value, max_value])
        ax.axvline(0, color='grey', alpha=0.50)

        ax.set_xlabel(title + ": neutral")
        bar_neutral= ax.barh(y_pos, x[1], align='center', height=0.80, alpha=0.5, tick_label=y[1])
        plt.show()
        # performance = x[1]
        #
        # plt.barh(y_pos, performance, align='center', alpha=0.5)
        # plt.yticks(y_pos, y[1])
        # plt.xlabel('Relative word relevance')
        # plt.title(title + ": neutral")

        # plt.show()

        # negative plot
        fig3 = plt.figure(figsize=(9, 6))
        ax = fig3.add_subplot(111)
        ax.tick_params(length=6, width=1)

        ax.set_title("title van de plot")

        ax.set_xlim([min_value, max_value])
        ax.axvline(0, color='grey', alpha=0.50)

        ax.set_xlabel(title + ": negative")
        bar_negative = ax.barh(y_pos, x[2], align='center', height=0.80, alpha=0.5, tick_label=y[2])
        plt.show()
        # performance = x[2]
        #
        # plt.barh(y_pos, performance, align='center', alpha=0.5)
        # plt.yticks(y_pos, y[2])
        # plt.xlabel('Relative word relevance')
        # plt.title(title + ": negative")
        #
        # plt.show()


class SingleSentencePlot(SentencePlot):

    def __init__(self, config, nlm_name):
        self.config = config
        self.nlm_name = nlm_name

    def plot(self, max_relevance_words_in_plot, sentence_id="1004293:0"):

        results_file = self.config.get_file_of_results(self.nlm_name)

        if not os.path.isfile(results_file):
            raise ("[!] Data %s not found" % results_file)

        with open(results_file, 'r') as file:
            for line in file:
                sentences = json.loads(line)
                sentences.pop(0)
                for sentence in sentences:

                    if sentence['sentence_id'] == sentence_id:

                        max_word_relevance = np.full((3, max_relevance_words_in_plot), -1.)
                        relevant_words = np.empty([3, max_relevance_words_in_plot], dtype=object)

                        for attribute_dict in sentence['word_relevance_linear_regression']:

                            for i in range(3):

                                min_value = max_word_relevance[i].min()
                                min_index = max_word_relevance[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance[i][min_index] = attribute_dict[str(i)]

                                    n_of_words = len(attribute_dict['word_attribute'])
                                    word = str(attribute_dict['word_attribute'][0])

                                    for w in range(1, n_of_words):
                                        word = word + ' ' + str(attribute_dict['word_attribute'][w])

                                    relevant_words[i][min_index] = word

                        title = "Most relevance words for sentence id " + str(sentence_id) + \
                                " and neural language model " + self.nlm_name
                        self.plot_final_results(max_word_relevance, relevant_words, title)

                        max_word_relevance = np.full((3, max_relevance_words_in_plot), -1.)
                        relevant_words = np.empty([3, max_relevance_words_in_plot], dtype=object)

                        for attribute_dict in sentence['word_relevance_pred_difference']:

                            for i in range(3):

                                min_value = max_word_relevance[i].min()
                                min_index = max_word_relevance[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance[i][min_index] = attribute_dict[str(i)]

                                    n_of_words = len(attribute_dict['word_attribute'])
                                    word = str(attribute_dict['word_attribute'][0])

                                    for w in range(1, n_of_words):
                                        word = word + ' ' + str(attribute_dict['word_attribute'][w])

                                    relevant_words[i][min_index] = word

                        title = "Most relevance words for sentence id " + str(sentence_id) + \
                                " and neural language model " + self.nlm_name
                        self.plot_final_results(max_word_relevance, relevant_words, title)


class PolaritySentencePlot(SentencePlot):

    def __init__(self, config, nlm_name):
        self.config = config
        self.nlm_name = nlm_name

    def plot(self, max_relevance_words_in_plot, indices, polarity="positive"):

        print("indices ", indices)

        number_of_polarity = self.get_polarity_number(polarity)

        results_file = self.config.get_file_of_results(self.nlm_name)

        if not os.path.isfile(results_file):
            raise ("[!] Data %s not found" % results_file)

        max_word_relevance_slr = np.zeros((3, max_relevance_words_in_plot), dtype=float)
        relevant_words_slr = np.empty([3, max_relevance_words_in_plot], dtype=object)

        max_word_relevance_spd = np.zeros((3, max_relevance_words_in_plot), dtype=float)
        relevant_words_spd = np.empty([3, max_relevance_words_in_plot], dtype=object)

        max_word_relevance_lr = np.zeros((3, max_relevance_words_in_plot), dtype=float)
        relevant_words_lr = np.empty([3, max_relevance_words_in_plot], dtype=object)

        max_word_relevance_pd = np.zeros((3, max_relevance_words_in_plot), dtype=float)
        relevant_words_pd = np.empty([3, max_relevance_words_in_plot], dtype=object)

        index_counter = 0
        index_indices = 0

        with open(results_file, 'r') as file:
            for line in file:
                sentences = json.loads(line)
                sentences.pop(0)
                for sentence in sentences:

                    if index_indices == len(indices):
                        break

                    print("np.argmax(sentence['aspect_polarity_matrix']) ", np.argmax(sentence['aspect_polarity_matrix']))
                    print("number_of_polarity ", number_of_polarity)
                    print("index_counter ", index_counter)
                    print("indices[index_indices] ", indices[index_indices])
                    if index_counter == indices[index_indices]:
                        index_indices += 1

                    if np.argmax(sentence['aspect_polarity_matrix']) == number_of_polarity and \
                            index_counter == indices[index_indices - 1]:

                        print("sentence ", sentence)
                        for attribute_dict in sentence['subsets_word_relevance_linear_regression']:

                            print("attribute_dict ", attribute_dict)

                            for i in range(3):

                                min_value = max_word_relevance_slr[i].min()
                                min_index = max_word_relevance_slr[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance_slr[i][min_index] = attribute_dict[str(i)]

                                    n_of_words = len(attribute_dict['word_attribute'])
                                    word = attribute_dict['word_attribute'][0]

                                    for w in range(1, n_of_words):
                                        word = word + " " + attribute_dict['word_attribute'][w]

                                    relevant_words_slr[i][min_index] = word

                        for attribute_dict in sentence['subsets_word_relevance_pred_difference']:

                            print("attribute_dict ", attribute_dict)

                            for i in range(3):

                                min_value = max_word_relevance_spd[i].min()
                                min_index = max_word_relevance_spd[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance_spd[i][min_index] = attribute_dict[str(i)]

                                    n_of_words = len(attribute_dict['word_attribute'])
                                    word = attribute_dict['word_attribute'][0]

                                    for w in range(1, n_of_words):
                                        word = word + " " + attribute_dict['word_attribute'][w]

                                    relevant_words_spd[i][min_index] = word

                        for attribute_dict in sentence['word_relevance_linear_regression']:

                            for i in range(3):

                                min_value = max_word_relevance_lr[i].min()
                                min_index = max_word_relevance_lr[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance_lr[i][min_index] = attribute_dict[str(i)]

                                    n_of_words = len(attribute_dict['word_attribute'])
                                    word = attribute_dict['word_attribute'][0]

                                    for w in range(1, n_of_words):
                                        word = word + " " + attribute_dict['word_attribute'][w]

                                    relevant_words_lr[i][min_index] = word

                        for attribute_dict in sentence['word_relevance_prediction_difference']:

                            for i in range(3):

                                min_value = max_word_relevance_pd[i].min()
                                min_index = max_word_relevance_pd[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance_pd[i][min_index] = attribute_dict[str(i)]

                                    n_of_words = len(attribute_dict['word_attribute'])
                                    word = attribute_dict['word_attribute'][0]

                                    for w in range(1, n_of_words):
                                        word = word + " " + attribute_dict['word_attribute'][w]

                                    relevant_words_pd[i][min_index] = word
                    index_counter += 1

        title = "Most relevance words for polarity " + str(polarity) + " subset linear regression"
        self.plot_final_results(max_word_relevance_slr, relevant_words_slr, title)

        title = "Most relevance words for polarity " + str(polarity) + " subset prediction difference"
        self.plot_final_results(max_word_relevance_spd, relevant_words_spd, title)

        title = "Most relevance words for polarity " + str(polarity) + " single linear regression"
        self.plot_final_results(max_word_relevance_lr, relevant_words_lr, title)

        title = "Most relevance words for polarity " + str(polarity) + " single prediction difference"
        self.plot_final_results(max_word_relevance_pd, relevant_words_pd, title)


class CategorySentencePlot(SentencePlot):

    def __init__(self, config, nlm_name):
        self.config = config
        self.nlm_name = nlm_name

    def plot(self, max_relevance_words_in_plot, category="AMBIENCE#GENERAL"):

        number_of_category = self.get_category_number(category)

        results_file = self.config.get_file_of_results(self.nlm_name)

        if not os.path.isfile(results_file):
            raise ("[!] Data %s not found" % results_file)

        max_word_relevance = np.zeros((3, max_relevance_words_in_plot), dtype=float)
        relevant_words = np.empty([3, max_relevance_words_in_plot], dtype=object)

        with open(results_file, 'r') as file:
            for line in file:
                sentences = json.loads(line)
                sentences.pop(0)
                for sentence in sentences:

                    if np.argmax(sentence['aspect_category_matrix']) == number_of_category:

                        for attribute_dict in sentence['word_relevance_per_set']:

                            for i in range(3):

                                min_value = max_word_relevance[i].min()
                                min_index = max_word_relevance[i].argmin()

                                if attribute_dict[str(i)] > min_value:
                                    max_word_relevance[i][min_index] = attribute_dict[str(i)]

                                    n_of_words = len(attribute_dict['word_attribute'])
                                    word = attribute_dict['word_attribute'][0]

                                    for w in range(1, n_of_words):
                                        word = " " + attribute_dict['word_attribute'][w]

                                    relevant_words[i][min_index] = word

        title = "Most relevance words for category " + str(category) + " and neural language model " + self.nlm_name
        self.plot_final_results(max_word_relevance, relevant_words, title)