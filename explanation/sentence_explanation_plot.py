import os, json
import numpy as np
import matplotlib.pyplot as plt
from nltk.parse import CoreNLPParser


class SentenceExplanationPlot:

    def __init__(self, neural_language_model):
        self.neural_language_model = neural_language_model
        self.server_url = 'http://localhost:9000'
        self.parser = CoreNLPParser(url=self.server_url)

    def run(self, sentence_id):

        file = self.neural_language_model.config.get_explanation_file(self.neural_language_model.config.name_of_model,
                                                                      sentence_id)

        with open(file, 'r') as file:
            for line in file:
                sentences = json.loads(line)

                for sentence in sentences:

                    lemmatized_sentence = sentence['lemmatized_sentence']
                    original_sentence = sentence['original_sentence']
                    tokenized_sentence = list(self.parser.tokenize(original_sentence))
                    aspect_indices = sentence['aspects']

                    sentence_id = sentence['sentence_id']
                    sentence_index = sentence['sentence_index']

                    argmax_pred = np.argmax(sentence['prediction'])

                    lace = []
                    lime = []
                    own = []

                    relation_yes = {}

                    aspect_sentiment_positive = {}
                    aspect_sentiment_negative = {}

                    word_sentiment_positive = {}
                    word_sentiment_negative = {}

                    attention_score = {}

                    x = []

                    if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                        for i in range(self.neural_language_model.config.n_iterations_hop):
                            relation_yes[i] = []
                            aspect_sentiment_positive[i] = []
                            aspect_sentiment_negative[i] = []
                            word_sentiment_positive[i] = []
                            word_sentiment_negative[i] = []
                            attention_score[i] = []
                    else:
                        relation_yes[0] = []
                        aspect_sentiment_positive[0] = []
                        aspect_sentiment_negative[0] = []
                        word_sentiment_positive[0] = []
                        word_sentiment_negative[0] = []
                        attention_score[0] = []

                    for index in range(len(lemmatized_sentence)):

                        if index in aspect_indices:
                            continue

                        original_word = tokenized_sentence[index]
                        x.append(original_word)

                        lemma = lemmatized_sentence[index]
                        word_info = sentence[lemma]

                        lime.append(word_info['relevance_linear_regression'][argmax_pred])
                        lace.append(word_info['subset_pred_dif'][argmax_pred])
                        own.append(word_info['subset_linear_reg'][argmax_pred])

                        if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                            for i in range(self.neural_language_model.config.n_iterations_hop):
                                attention_score[i].append(word_info['attention_score_' + str(i)])

                                aspect_sentiment_positive[i].append(
                                    word_info['weighted_states_pred_aspect_sentiments_' + str(i)][0])
                                aspect_sentiment_negative[i].append(
                                    word_info['weighted_states_pred_aspect_sentiments_' + str(i)][1])

                                relation_yes[i].append(word_info['weighted_states_pred_relations_' + str(i)][0])

                                word_sentiment_positive[i].append(
                                    word_info['weighted_states_pred_word_sentiments_' + str(i)][0])
                                word_sentiment_negative[i].append(
                                    word_info['weighted_states_pred_word_sentiments_' + str(i)][1])
                        else:
                            attention_score[0].append(word_info['attention_score'])

                            aspect_sentiment_positive[0].append(
                                word_info['weighted_states_pred_aspect_sentiments'][0])
                            aspect_sentiment_negative[0].append(
                                word_info['weighted_states_pred_aspect_sentiments'][1])

                            relation_yes[0].append(word_info['weighted_states_pred_relations'][0])

                            word_sentiment_positive[0].append(
                                word_info['weighted_states_pred_word_sentiments'][0])
                            word_sentiment_negative[0].append(
                                word_info['weighted_states_pred_word_sentiments'][1])

                    sum_lace = np.sum(np.abs(lace))
                    sum_lime = np.sum(np.abs(lime))
                    sum_own = np.sum(np.abs(own))

                    average_lace = np.array(lace) / sum_lace
                    print("average_lace ", average_lace)
                    average_lime = np.array(lime) / sum_lime
                    print("average_lime ", average_lime)
                    average_own = np.array(own) / sum_own
                    print("own ", own)
                    print("average_own ", average_own)

                    for i in range(self.neural_language_model.config.n_iterations_hop):

                        self.plot(x, average_lime, average_lace, average_own, attention_score[i], aspect_sentiment_positive[i],
                                  aspect_sentiment_negative[i], word_sentiment_positive[i],
                                  word_sentiment_negative[i], relation_yes[i], sentence_id, sentence_index, i)

    def plot(self, x, lime, lace, own, attention_score, aspect_sentiment_positive, aspect_sentiment_negative,
             word_sentiment_positive, word_sentiment_negative, relation_yes, sentence_id, index_number, weight_number):

        fig, ax = plt.subplots()
        fig.set_size_inches(40.5, 14.5)
        ax.tick_params(length=15, axis='x', width=3, labelsize=58)
        ax.tick_params(length=15, axis='y', width=3, labelsize=40)

        plt.subplots_adjust(bottom=0.55)
        ax.set_ylim([-0.1, 1.1])

        ax.axhline(0, color='grey', alpha=0.50)

        index = np.array([2+x*1.25 for x in range(len(x))])
        print(index)
        bar_width = 0.10
        opacity = 0.8

        rects1 = plt.bar(index, relation_yes, bar_width, alpha=opacity, align='center', color='C0', label='ARC', edgecolor='black')
        rects2 = plt.bar(index + bar_width, aspect_sentiment_positive, bar_width, alpha=opacity, align='center', color='C1', label='ARWSC positive', edgecolor='black')
        rects3 = plt.bar(index + bar_width * 2, aspect_sentiment_negative, bar_width, alpha=opacity, align='center', color='C2', label='ARWSC negative', edgecolor='black')
        rects4 = plt.bar(index + bar_width * 3, word_sentiment_positive, bar_width, alpha=opacity, align='center', color='C3', label='WSC positive', edgecolor='black')
        rects5 = plt.bar(index + bar_width * 4, word_sentiment_negative, bar_width, alpha=opacity, align='center', color='C4', label='WSC negative', edgecolor='black')
        rects6 = plt.bar(index + bar_width * 5, attention_score, bar_width, alpha=opacity, align='center', color='C9', label='Attention score', edgecolor='black')
        rects7 = plt.bar(index + bar_width * 6, lime, bar_width, alpha=opacity, align='center', color='C6', label='A-LIME', edgecolor='black')
        rects8 = plt.bar(index + bar_width * 7, lace, bar_width, alpha=opacity, align='center', color='C7', label='A-LACE', edgecolor='black')
        rects9 = plt.bar(index + bar_width * 8, own, bar_width, alpha=opacity, align='center', color='C8', label='LETA', edgecolor='black')

        plt.xticks(index + bar_width, x)
        plt.xticks(index + bar_width * 2, x)
        plt.xticks(index + bar_width * 3, x)
        plt.xticks(index + bar_width * 4, x)

        plt.legend(loc='upper left', prop={'size': 36})

        plt.tight_layout()
        # plt.show()

        model_name = self.neural_language_model.config.name_of_model
        file = self.neural_language_model.config.get_plot_entire_sentence(model_name, sentence_id,
                                                                          index_number, weight_number)

        plt.savefig(file)



