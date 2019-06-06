import numpy as np
import json


class OntologyReasoner:

    def __init__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    @staticmethod
    def predict_sentiment_of_sentences(sentence_polarities, aspects_dependencies, sentence_negations, polarity_matrix):

        number_of_sentences = len(sentence_polarities)

        polarity_count = np.array([0, 0, 0])
        majority_count = np.array([0, 0, 0])
        with_backup_count = np.array([0, 0, 0])
        indices_not_able_to_predict = []

        for index_sentence in range(number_of_sentences):

            index_true_polarity = polarity_matrix[index_sentence].index(1)
            polarity_count[index_true_polarity] += 1

            number_of_words_in_sentence = len(sentence_polarities[index_sentence])
            positive_polarities = []
            negative_polarities = []

            negation = False

            if len(sentence_negations[index_sentence]) == 2:

                aspect_relation_1 = aspects_dependencies[index_sentence][sentence_negations[index_sentence][0]]
                aspect_relation_2 = aspects_dependencies[index_sentence][sentence_negations[index_sentence][1]]

                if aspect_relation_1 != 'no' or aspect_relation_2 != 'no':
                    negation = True

            for index_word in range(number_of_words_in_sentence):

                if sentence_polarities[index_sentence][index_word] == "Positive":

                    if negation:
                        negative_polarities.append(True)
                    else:
                        positive_polarities.append(True)

                elif sentence_polarities[index_sentence][index_word] == "Negative":

                    if negation:
                        positive_polarities.append(True)
                    else:
                        negative_polarities.append(True)

            if True in positive_polarities and True not in negative_polarities:

                if polarity_matrix[index_sentence][0] == 1:
                    majority_count[0] += 1
                    with_backup_count[0] += 1
            elif True not in positive_polarities and True in negative_polarities:

                if polarity_matrix[index_sentence][2] == 1:
                    majority_count[2] += 1
                    with_backup_count[2] += 1

            elif True in positive_polarities and True in negative_polarities:

                indices_not_able_to_predict.append(index_sentence)
                if polarity_matrix[index_sentence][0] == 1:
                    majority_count[0] += 1

            elif True not in positive_polarities and True not in negative_polarities:

                indices_not_able_to_predict.append(index_sentence)
                if polarity_matrix[index_sentence][0] == 1:
                    majority_count[0] += 1

        return majority_count, with_backup_count, polarity_count, indices_not_able_to_predict

    def run(self):

        x_train = self.internal_data_loader.word_polarities_training
        train_aspects_dependencies = self.internal_data_loader.aspect_dependencies_training
        train_negations = self.internal_data_loader.negation_in_training
        y_train = self.internal_data_loader.polarity_matrix_training

        x_test = self.internal_data_loader.word_polarities_test
        test_aspects_dependencies = self.internal_data_loader.aspect_dependencies_test
        test_negations = self.internal_data_loader.negation_in_test
        y_test = self.internal_data_loader.polarity_matrix_test

        if self.config.cross_validation:

            training_indices, validation_indices = self.internal_data_loader.get_indices_cross_validation()

            results = {}
            remaining_indices = []

            for i in range(self.config.cross_validation_rounds):

                x_train_cross = x_train[training_indices[i]]
                y_train_cross = y_train[training_indices[i]]
                train_aspects_dependencies_cross = train_aspects_dependencies[training_indices[i]]
                train_negations_cross = train_negations[training_indices[i]]
                x_validation_cross = x_train[validation_indices[i]]
                y_validation_cross = y_train[validation_indices[i]]
                validation_aspects_dependencies_cross = train_aspects_dependencies[validation_indices[i]]
                validation_negations_cross = train_negations[validation_indices[i]]

                majority_count_training, with_backup_count_training, count_training, _ = \
                    self.predict_sentiment_of_sentences(
                        sentence_polarities=x_train_cross,
                        aspects_dependencies=train_aspects_dependencies_cross,
                        sentence_negations=train_negations_cross,
                        polarity_matrix=y_train_cross
                    )

                accuracy_majority_training = majority_count_training / count_training
                accuracy_with_back_training = with_backup_count_training / count_training

                majority_count_validation, with_backup_count_validation, count_validation, remaining_validation_indices = \
                    self.predict_sentiment_of_sentences(
                        sentence_polarities=x_validation_cross,
                        aspects_dependencies=validation_aspects_dependencies_cross,
                        sentence_negations=validation_negations_cross,
                        polarity_matrix=y_validation_cross
                    )

                remaining_indices.append(remaining_validation_indices)

                accuracy_majority_validation = majority_count_validation / count_validation
                accuracy_with_back_validation = with_backup_count_validation / count_validation

                result = {
                    'classification model': self.config.name_of_model,
                    'n_in_training_sample': count_training,
                    'n_in_validation_sample': count_validation,
                    'train_acc_majority': accuracy_majority_training,
                    'train_acc_with_backup': accuracy_with_back_training,
                    'validation_acc_majority': accuracy_majority_validation,
                    'validation_acc_with_backup': accuracy_with_back_validation,
                    'train_pred_y_majority': majority_count_training,
                    'train_pred_y_with_backup': with_backup_count_training,
                    'validation_pred_y_majority': majority_count_validation,
                    'validation_pred_y_with_backup': with_backup_count_validation
                }

                results['cross_validation_' + str(i)] = result

            remaining_sentences = {
                'cross_val_indices': remaining_indices
            }

            with open(self.config.remaining_data_cross_val, 'w') as outfile:
                json.dump(remaining_sentences, outfile, ensure_ascii=False)

            with open(self.config.file_of_cross_val_results, 'w') as outfile:
                json.dump(results, outfile, ensure_ascii=False)

        else:

            majority_count_training, with_backup_count_training, count_training, _ = \
                self.predict_sentiment_of_sentences(
                        sentence_polarities=x_train,
                        aspects_dependencies=train_aspects_dependencies,
                        sentence_negations=train_negations,
                        polarity_matrix=y_train
                    )

            accuracy_majority_training = majority_count_training / count_training
            accuracy_with_back_training = with_backup_count_training / count_training

            majority_count_test, with_backup_count_test, count_test, remaining_test_indices = \
                self.predict_sentiment_of_sentences(
                    sentence_polarities=x_test,
                    aspects_dependencies=test_aspects_dependencies,
                    sentence_negations=test_negations,
                    polarity_matrix=y_test
                )

            remaining_sentences = {
                'test_set_indices': remaining_test_indices
            }

            with open(self.config.remaining_data, 'w') as outfile:
                json.dump(remaining_sentences, outfile, ensure_ascii=False)

            accuracy_majority_test = majority_count_test / count_test
            accuracy_with_back_test = with_backup_count_test / count_test

            results = {
                'classification model': self.config.name_of_model,
                'n_in_training_sample': count_training,
                'n_in_test_sample': count_test,
                'train_acc_majority': accuracy_majority_training,
                'train_acc_with_backup': accuracy_with_back_training,
                'test_acc_majority': accuracy_majority_test,
                'test_acc_with_backup': accuracy_with_back_test,
                'train_pred_y_majority': majority_count_training,
                'train_pred_y_with_backup': with_backup_count_training,
                'test_pred_y_majority': majority_count_test,
                'test_pred_y_with_backup': with_backup_count_test
            }

            with open(self.config.file_of_results, 'w') as outfile:
                json.dump(results, outfile, ensure_ascii=False)
