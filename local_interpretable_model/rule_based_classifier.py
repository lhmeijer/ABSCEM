import numpy as np


class RuleBasedClassifier:

    def __init__(self, tree_depth):
        self.tree_depth = tree_depth
        self.rules_indices = []
        self.rules_lemmas = []

    def extract_rules(self, x_train, y_train, lemmas, true_y):

        self.rules_indices = []
        self.rules_lemmas = []

        n_of_words = x_train.shape[1]

        tree_depth = int(np.around(np.sqrt(n_of_words - 1)))
        if tree_depth > self.tree_depth:
            self.tree_depth = tree_depth

        def go_through_tree(x, y, previous_indices, in_sentence, lemma_sentence, true_y):
            # print("in_sentence ", in_sentence)
            # print("previous_indices ", previous_indices)

            index, no_indices, yes_indices = self.compute_word_scores_and_rules(x, y, previous_indices, in_sentence,
                                                                                lemma_sentence, true_y)
            # print("index ", index)
            previous_indices.append(index)
            if index == -1 or len(previous_indices) == self.tree_depth:
                return

            # if len(previous_indices) == (n_of_words - 1) or len(previous_indices) == self.tree_depth:
            #     return
            #
            # print("previous_indices ", previous_indices)
            # print("in_sentence ", in_sentence)
            # print("index ", index)
            # print("no_indices ", no_indices)
            # print("yes_indices ", yes_indices)

            in_sentence.append(0)
            go_through_tree(x[no_indices], y[no_indices], previous_indices, in_sentence, lemma_sentence, true_y)
            previous_indices.pop()
            in_sentence.pop()

            in_sentence.append(1)
            go_through_tree(x[yes_indices], y[yes_indices], previous_indices, in_sentence, lemma_sentence, true_y)
            previous_indices.pop()
            in_sentence.pop()

        go_through_tree(x_train, y_train, [], [], lemmas, true_y)
        return self.rules_indices, self.rules_lemmas

    def compute_word_scores_and_rules(self, x, y, previous_indices, previous_in_sentence, lemma_sentence, true_y):

        n_of_words = x.shape[1]
        # print("n_of_words ", n_of_words)
        n_of_neighbours = x.shape[0]
        max_goodness_of_fit_score = 0
        max_word_index = -1
        sentence_indices = np.arange(x.shape[0])
        no_indices = None
        yes_indices = None

        # print("previous_indices ", previous_indices)
        # print("previous_in_sentence ", previous_in_sentence)
        # print("lemma_sentence ", lemma_sentence)
        # print("x ", x)
        # print("y ", y)

        for word_index in range(n_of_words):
            # print("word_index ", word_index)

            if word_index not in previous_indices:

                in_sentence = y[x[:, word_index] == 1]
                n_in_sentence = in_sentence.shape[0]
                # print("n_in_sentence ", n_in_sentence)

                not_in_sentence = y[x[:, word_index] == 0]
                n_not_in_sentence = not_in_sentence.shape[0]
                # print("n_not_in_sentence ", n_not_in_sentence)

                if n_in_sentence != 0 and n_not_in_sentence != 0:

                    arg_max_in_sentence = np.argmax(in_sentence, axis=1)
                    mean_in_sentence = in_sentence.mean(axis=0)
                    # print("mean_in_sentence ", mean_in_sentence)
                    correct_classified_in_sentence = in_sentence[arg_max_in_sentence == true_y]
                    n_correct_classified_in_sentence = correct_classified_in_sentence.shape[0]
                    # print("n_correct_classified_in_sentence ", n_correct_classified_in_sentence)
                    acc_in_sentence = n_correct_classified_in_sentence / n_in_sentence
                    # print("acc_in_sentence ", acc_in_sentence)
                    score_in_sentence = mean_in_sentence[true_y] * acc_in_sentence

                    # print("score_in_sentence", score_in_sentence)
                    arg_max_not_in_sentence = np.argmax(not_in_sentence, axis=1)
                    mean_not_in_sentence = not_in_sentence.mean(axis=0)
                    # print("mean_not_in_sentence ", mean_not_in_sentence)

                    correct_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence == true_y]
                    n_correct_classified_not_in_sentence = correct_classified_not_in_sentence.shape[0]
                    # print("n_correct_classified_not_in_sentence ", n_correct_classified_not_in_sentence)
                    acc_not_in_sentence = n_correct_classified_not_in_sentence / n_not_in_sentence
                    # print("acc_not_in_sentence ", acc_not_in_sentence)
                    score_not_in_sentence = mean_not_in_sentence[true_y] * acc_not_in_sentence
                    # print("score_not_in_sentence ", score_not_in_sentence)

                    goodness_difference = score_in_sentence - score_not_in_sentence
                    goodness_score_per_word = np.abs(goodness_difference)
                    # print("goodness_score_per_word ", goodness_score_per_word)

                    # # positive
                    # positive_classified_in_sentence = in_sentence[arg_max_in_sentence == 0].shape[0]
                    # print("positive_classified_in_sentence ", positive_classified_in_sentence)
                    # negative_classified_in_sentence = in_sentence[arg_max_in_sentence == 2].shape[0]
                    # # print("negative_classified_in_sentence ", negative_classified_in_sentence)
                    #
                    # # negative
                    # positive_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence == 0].shape[0]
                    # print("positive_classified_not_in_sentence ", positive_classified_not_in_sentence)
                    # negative_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence == 2].shape[0]
                    # # print("negative_classified_not_in_sentence ", negative_classified_not_in_sentence)
                    #
                    # positive_difference = np.abs((positive_classified_in_sentence / n_in_sentence) -
                    #                              (positive_classified_not_in_sentence / n_not_in_sentence))
                    # print("positive_difference ", positive_difference)
                    # negative_difference = np.abs((negative_classified_in_sentence / n_in_sentence) -
                    #                              (negative_classified_not_in_sentence / n_not_in_sentence))
                    # print("negative_difference ", negative_difference)

                    # if positive_difference > self.gamma:
                    #     previous_rules_indices, previous_rules_lemmas = [], []
                    #
                    #     for index in range(len(previous_in_sentence)):
                    #
                    #         if previous_in_sentence[index] == 1:
                    #             previous_index = previous_indices[index]
                    #             previous_bit = previous_in_sentence[index]
                    #             previous_rules_indices.append((previous_index, previous_bit))
                    #             # print("previous_rules_indices ", previous_rules_indices)
                    #             previous_rules_lemmas.append((lemma_sentence[previous_index], previous_bit))
                    #             # print("previous_rules_lemmas ", previous_rules_lemmas)
                    #
                    #     previous_rules_indices.append((word_index, 1))
                    #     previous_rules_lemmas.append((lemma_sentence[word_index], 1))
                    #     rule_indices = [previous_rules_indices, polarity_in_sentence]
                    #     rule_names = [previous_rules_lemmas, polarity_in_sentence]
                    #     self.rules_indices.append(rule_indices)
                    #     print("rules_indices ", self.rules_indices)
                    #     self.rules_lemmas.append(rule_names)
                    #     print("rules_lemmas ", self.rules_lemmas)

                    if goodness_score_per_word > max_goodness_of_fit_score:
                        max_goodness_of_fit_score = goodness_score_per_word
                        max_word_index = word_index
                        no_indices = sentence_indices[x[:, word_index] == 0]
                        yes_indices = sentence_indices[x[:, word_index] == 1]

        previous_rules_indices, previous_rules_lemmas = [], []

        if max_word_index != -1:

            for index in range(len(previous_in_sentence)):

                if previous_in_sentence[index] == 1:
                    previous_index = previous_indices[index]
                    previous_bit = previous_in_sentence[index]
                    previous_rules_indices.append((previous_index, previous_bit))
                    # print("previous_rules_indices ", previous_rules_indices)
                    previous_rules_lemmas.append((lemma_sentence[previous_index], previous_bit))
                    # print("previous_rules_lemmas ", previous_rules_lemmas)

            previous_rules_indices.append((max_word_index, 1))
            previous_rules_lemmas.append((lemma_sentence[max_word_index], 1))
            rule_indices = [previous_rules_indices, true_y]
            rule_names = [previous_rules_lemmas, true_y]
            self.rules_indices.append(rule_indices)
            # print("rules_indices ", self.rules_indices)
            self.rules_lemmas.append(rule_names)
            # print("rules_lemmas ", self.rules_lemmas)

        return max_word_index, no_indices, yes_indices
