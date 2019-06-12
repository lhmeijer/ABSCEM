import numpy as np


class RuleBasedClassifier:

    def __init__(self, gamma, tree_depth):
        self.gamma = gamma
        self.tree_depth = tree_depth
        self.rules_indices = []
        self.rules_lemmas = []

    def extract_rules(self, x_train, y_train, lemmas):

        n_of_words = x_train.shape[1]

        def go_through_tree(x, y, previous_indices, in_sentence, lemma_sentence):

            index, no_indices, yes_indices, no_rule, yes_rule = self.compute_word_scores_and_rules(x, y,
                                                                                                   previous_indices,
                                                                                                   in_sentence,
                                                                                                   lemma_sentence)
            previous_indices = previous_indices.append(index)

            if len(previous_indices) == (n_of_words - 1) or len(previous_indices) == self.tree_depth:
                return

            if not no_rule:
                go_through_tree(x[no_indices], y[no_indices], previous_indices, in_sentence.append(0), lemma_sentence)
            if not yes_rule:
                go_through_tree(x[yes_indices], y[yes_indices], previous_indices, in_sentence.append(1), lemma_sentence)

        go_through_tree(x_train, y_train, [], [], lemmas)
        return self.rules_indices, self.rules_lemmas

    def compute_word_scores_and_rules(self, x, y, previous_indices, previous_in_sentence, lemma_sentence):

        n_of_words = x.shape[1]
        max_goodness_of_fit_score = 0
        max_word_index = 0
        sentence_indices = np.arange(x.shape[0])
        no_indices = None
        yes_indices = None
        no_rule = False
        yes_rule = False

        for word_index in range(n_of_words):

            if word_index not in previous_indices:

                in_sentence = y[x[:, word_index] == 1]
                n_in_sentence = in_sentence.shape[0]
                arg_max_in_sentence = np.argmax(in_sentence, axis=1)
                sum_in_sentence = in_sentence.sum(axis=0)

                polarity_in_sentence = np.argmax(sum_in_sentence)
                correct_classified_in_sentence = in_sentence[arg_max_in_sentence == polarity_in_sentence]
                n_correct_classified_in_sentence = correct_classified_in_sentence.shape[0]
                score_in_sentence = correct_classified_in_sentence.mean(axis=0)[polarity_in_sentence]

                if n_correct_classified_in_sentence / n_in_sentence > self.gamma:
                    previous_rules_indices, previous_rules_lemmas = [], []
                    for previous_index in range(len(previous_in_sentence)):
                        previous_rules_indices.append([(previous_indices[previous_index],
                                                      previous_in_sentence[previous_index])])
                        previous_rules_lemmas.append([(lemma_sentence[previous_index],
                                                      previous_in_sentence[previous_index])])

                    previous_rules_indices.append([(word_index, 1)])
                    previous_rules_lemmas.append([(lemma_sentence[word_index], 1)])
                    rule_indices = [previous_rules_indices, [polarity_in_sentence]]
                    rule_names = [previous_rules_lemmas, [polarity_in_sentence]]
                    self.rules_indices.append(rule_indices)
                    self.rules_lemmas.append(rule_names)

                not_in_sentence = y[x[:, word_index] == 0]
                n_not_in_sentence = not_in_sentence.shape[0]
                arg_max_not_in_sentence = np.argmax(not_in_sentence, axis=1)
                sum_not_in_sentence = not_in_sentence.sum(axis=0)

                polarity_not_in_sentence = np.argmax(sum_not_in_sentence)
                correct_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence ==
                                                                     polarity_not_in_sentence]
                score_not_in_sentence = correct_classified_not_in_sentence.mean(axis=0)[polarity_not_in_sentence]

                if correct_classified_not_in_sentence / n_not_in_sentence > self.gamma:
                    previous_rules_indices, previous_rules_lemmas = [], []
                    for previous_index in range(len(previous_in_sentence)):
                        previous_rules_indices.append([(previous_indices[previous_index],
                                                        previous_in_sentence[previous_index])])
                        previous_rules_lemmas.append([(lemma_sentence[previous_index],
                                                       previous_in_sentence[previous_index])])

                    previous_rules_indices.append([(word_index, 0)])
                    previous_rules_lemmas.append([(lemma_sentence[word_index], 0)])
                    rule_indices = [previous_rules_indices, [polarity_not_in_sentence]]
                    rule_names = [previous_rules_lemmas, [polarity_not_in_sentence]]
                    self.rules_indices.append(rule_indices)
                    self.rules_lemmas.append(rule_names)

                goodness_score_per_word = score_in_sentence + score_not_in_sentence

                if goodness_score_per_word > max_goodness_of_fit_score:
                    max_goodness_of_fit_score = goodness_score_per_word
                    max_word_index = word_index
                    no_indices = sentence_indices[x[:, word_index] == 0]
                    yes_indices = sentence_indices[x[:, word_index] == 1]

        return max_word_index, no_indices, yes_indices, no_rule, yes_rule
