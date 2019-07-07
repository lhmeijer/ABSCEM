import numpy as np


class DecisionTree:

    def __init__(self):
        self.rules_indices = []
        self.rules_lemmas = []

    def extract_word_combinations(self, x_train, y_train, lemmas, pred_y, aspects):

        self.rules_indices = []
        self.rules_lemmas = []

        n_of_words = x_train.shape[1]

        tree_depth = int(np.floor(np.log2(n_of_words - len(aspects) + 1)))

        def go_through_tree(x, y, previous_indices, in_sentence, lemma_sentence, pred_y):

            index, no_indices, yes_indices = self.compute_word_scores_and_rules(x, y, previous_indices, in_sentence,
                                                                                lemma_sentence, pred_y)
            previous_indices.append(index)
            if index == -1 or len(previous_indices) == tree_depth:
                return

            in_sentence.append(0)
            go_through_tree(x[no_indices], y[no_indices], previous_indices, in_sentence, lemma_sentence, pred_y)
            previous_indices.pop()
            in_sentence.pop()

            in_sentence.append(1)
            go_through_tree(x[yes_indices], y[yes_indices], previous_indices, in_sentence, lemma_sentence, pred_y)
            previous_indices.pop()
            in_sentence.pop()

        go_through_tree(x_train, y_train, [], [], lemmas, pred_y)
        return self.rules_indices, self.rules_lemmas

    def compute_word_scores_and_rules(self, x, y, previous_indices, previous_in_sentence, lemma_sentence, pred_y):

        n_of_words = x.shape[1]
        max_goodness_of_fit_score = 0
        max_goodness_difference = 0
        max_word_index = -1
        sentence_indices = np.arange(x.shape[0])
        no_indices = None
        yes_indices = None

        for word_index in range(n_of_words):

            if word_index not in previous_indices:

                in_sentence = y[x[:, word_index] == 1]
                n_in_sentence = in_sentence.shape[0]

                not_in_sentence = y[x[:, word_index] == 0]
                n_not_in_sentence = not_in_sentence.shape[0]

                if n_in_sentence != 0 and n_not_in_sentence != 0:

                    arg_max_in_sentence = np.argmax(in_sentence, axis=1)
                    mean_in_sentence = in_sentence.mean(axis=0)
                    correct_classified_in_sentence = in_sentence[arg_max_in_sentence == pred_y]
                    n_correct_classified_in_sentence = correct_classified_in_sentence.shape[0]
                    acc_in_sentence = n_correct_classified_in_sentence / n_in_sentence
                    score_in_sentence = mean_in_sentence[pred_y] * acc_in_sentence

                    arg_max_not_in_sentence = np.argmax(not_in_sentence, axis=1)
                    mean_not_in_sentence = not_in_sentence.mean(axis=0)
                    correct_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence == pred_y]
                    n_correct_classified_not_in_sentence = correct_classified_not_in_sentence.shape[0]
                    acc_not_in_sentence = n_correct_classified_not_in_sentence / n_not_in_sentence
                    score_not_in_sentence = mean_not_in_sentence[pred_y] * acc_not_in_sentence

                    goodness_difference = score_in_sentence - score_not_in_sentence
                    goodness_score_per_word = np.abs(goodness_difference)

                    if goodness_score_per_word > max_goodness_of_fit_score:
                        max_goodness_of_fit_score = goodness_score_per_word
                        max_goodness_difference = goodness_difference
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
                    previous_rules_lemmas.append((lemma_sentence[previous_index], previous_bit))

            if max_goodness_difference < 0:
                previous_rules_indices.append((max_word_index, 0))
                previous_rules_lemmas.append((lemma_sentence[max_word_index], 0))
            else:
                previous_rules_indices.append((max_word_index, 1))
                previous_rules_lemmas.append((lemma_sentence[max_word_index], 1))

            rule_indices = [previous_rules_indices, pred_y]
            rule_names = [previous_rules_lemmas, pred_y]
            self.rules_indices.append(rule_indices)
            self.rules_lemmas.append(rule_names)

        return max_word_index, no_indices, yes_indices
