import numpy as np


class RuleBasedClassifier:

    def __init__(self, gamma, tree_depth):
        self.gamma = gamma
        self.tree_depth = tree_depth
        self.rules_indices = []
        self.rules_lemmas = []

    def extract_rules(self, x_train, y_train, lemmas):

        n_of_words = x_train.shape[1]
        print("n_of_words ", n_of_words)

        def go_through_tree(x, y, previous_indices, in_sentence, lemma_sentence):
            print("in_sentence ", in_sentence)
            print("previous_indices ", previous_indices)

            index, no_indices, yes_indices = self.compute_word_scores_and_rules(x, y, previous_indices, in_sentence,
                                                                                lemma_sentence)
            print("index ", index)
            previous_indices.append(index)
            if index == -1:
                return

            # if len(previous_indices) == (n_of_words - 1) or len(previous_indices) == self.tree_depth:
            #     return

            print("previous_indices ", previous_indices)
            print("in_sentence ", in_sentence)
            print("index ", index)
            print("no_indices ", no_indices)
            print("yes_indices ", yes_indices)

            in_sentence.append(0)
            go_through_tree(x[no_indices], y[no_indices], previous_indices, in_sentence, lemma_sentence)
            previous_indices.pop()
            in_sentence.pop()

            in_sentence.append(1)
            go_through_tree(x[yes_indices], y[yes_indices], previous_indices, in_sentence, lemma_sentence)
            previous_indices.pop()
            in_sentence.pop()

        go_through_tree(x_train, y_train, [], [], lemmas)
        return self.rules_indices, self.rules_lemmas

    def compute_word_scores_and_rules(self, x, y, previous_indices, previous_in_sentence, lemma_sentence):

        n_of_words = x.shape[1]
        n_of_neighbours = x.shape[0]
        max_goodness_of_fit_score = 0
        max_word_index = -1
        sentence_indices = np.arange(x.shape[0])
        no_indices = None
        yes_indices = None

        print("previous_indices ", previous_indices)
        print("previous_in_sentence ", previous_in_sentence)
        print("lemma_sentence ", lemma_sentence)
        print("x ", x)
        print("y ", y)

        for word_index in range(n_of_words):
            print("word_index ", word_index)

            if word_index not in previous_indices:

                in_sentence = y[x[:, word_index] == 1]
                n_in_sentence = in_sentence.shape[0]

                not_in_sentence = y[x[:, word_index] == 0]
                n_not_in_sentence = not_in_sentence.shape[0]

                if n_in_sentence != 0 and n_not_in_sentence != 0:

                    arg_max_in_sentence = np.argmax(in_sentence, axis=1)
                    sum_in_sentence = in_sentence.sum(axis=0)

                    polarity_in_sentence = np.argmax(sum_in_sentence)
                    print("polarity_in_sentence ", polarity_in_sentence)
                    correct_classified_in_sentence = in_sentence[arg_max_in_sentence == polarity_in_sentence]
                    print("correct_classified_in_sentence ", correct_classified_in_sentence)
                    n_correct_classified_in_sentence = correct_classified_in_sentence.shape[0]
                    print("n_correct_classified_in_sentence ", n_correct_classified_in_sentence)
                    score_in_sentence = correct_classified_in_sentence.sum(axis=0)[polarity_in_sentence]

                    print("n_not_in_sentence ", n_not_in_sentence)
                    arg_max_not_in_sentence = np.argmax(not_in_sentence, axis=1)
                    print("arg_max_not_in_sentence ", arg_max_not_in_sentence)
                    sum_not_in_sentence = not_in_sentence.sum(axis=0)
                    print("sum_not_in_sentence ", sum_not_in_sentence)

                    polarity_not_in_sentence = np.argmax(sum_not_in_sentence)
                    print("polarity_not_in_sentence ", polarity_not_in_sentence)
                    correct_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence ==
                                                                         polarity_not_in_sentence]
                    print("correct_classified_not_in_sentence ", correct_classified_not_in_sentence)
                    n_correct_classified_not_in_sentence = correct_classified_not_in_sentence.shape[0]
                    print("n_correct_classified_not_in_sentence ", n_correct_classified_not_in_sentence)
                    score_not_in_sentence = correct_classified_not_in_sentence.sum(axis=0)[polarity_not_in_sentence]

                    # positive
                    positive_classified_in_sentence = in_sentence[arg_max_in_sentence == 0].shape[0]
                    print("positive_classified_in_sentence ", positive_classified_in_sentence)
                    negative_classified_in_sentence = in_sentence[arg_max_in_sentence == 2].shape[0]
                    print("negative_classified_in_sentence ", negative_classified_in_sentence)

                    # negative
                    positive_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence == 0].shape[0]
                    print("positive_classified_not_in_sentence ", positive_classified_not_in_sentence)
                    negative_classified_not_in_sentence = not_in_sentence[arg_max_not_in_sentence == 2].shape[0]
                    print("negative_classified_not_in_sentence ", negative_classified_not_in_sentence)

                    positive_difference = np.abs((positive_classified_in_sentence / n_in_sentence) -
                                                 (positive_classified_not_in_sentence / n_not_in_sentence))
                    print("positive_difference ", positive_difference)
                    # negative_difference = np.abs((negative_classified_in_sentence / n_in_sentence) -
                    #                              (negative_classified_not_in_sentence / n_not_in_sentence))
                    # print("negative_difference ", negative_difference)

                    if positive_difference > self.gamma:
                        previous_rules_indices, previous_rules_lemmas = [], []

                        for previous_index in range(len(previous_in_sentence)):

                            if previous_in_sentence[previous_index] == 1:
                                previous_rules_indices.append((previous_indices[previous_index],
                                                               previous_in_sentence[previous_index]))
                                print("previous_rules_indices ", previous_rules_indices)
                                previous_rules_lemmas.append((lemma_sentence[previous_index],
                                                              previous_in_sentence[previous_index]))
                                print("previous_rules_lemmas ", previous_rules_lemmas)

                        previous_rules_indices.append((word_index, 1))
                        previous_rules_lemmas.append((lemma_sentence[word_index], 1))
                        rule_indices = [previous_rules_indices, polarity_in_sentence]
                        rule_names = [previous_rules_lemmas, polarity_in_sentence]
                        self.rules_indices.append(rule_indices)
                        print("rules_indices ", self.rules_indices)
                        self.rules_lemmas.append(rule_names)
                        print("rules_lemmas ", self.rules_lemmas)

                    goodness_score_per_word = score_in_sentence + score_not_in_sentence
                    print("goodness_score_per_word ", goodness_score_per_word)

                    if goodness_score_per_word > max_goodness_of_fit_score:
                        max_goodness_of_fit_score = goodness_score_per_word
                        max_word_index = word_index
                        no_indices = sentence_indices[x[:, word_index] == 0]
                        yes_indices = sentence_indices[x[:, word_index] == 1]

        return max_word_index, no_indices, yes_indices
