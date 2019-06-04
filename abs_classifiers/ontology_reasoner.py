import numpy as np
import json


class OntologyReasoner:

    def __init__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    @staticmethod
    def predict_sentiment_of_sentences(sentence_polarities, sentence_negation, polarity_matrix):

        number_of_polarities = len(sentence_polarities)

        majority_count = 0
        with_backup_count = 0
        indices_not_able_to_predict = []

        for index_polarity in range(number_of_polarities):

            number_of_words_in_sentence = len(sentence_polarities[index_polarity])
            positive_polarities = []
            negative_polarities = []

            for index_word in range(number_of_words_in_sentence):

                left_index = max(0, index_word - 3)
                right_index = min(number_of_words_in_sentence, index_word + 3)

                if sentence_polarities[index_polarity][index_word] == "Positive":

                    if True in sentence_negation[left_index:right_index]:
                        negative_polarities.append(True)
                    else:
                        positive_polarities.append(True)
                elif sentence_polarities[index_polarity][index_word] == "Negative":

                    if True in sentence_negation[left_index:right_index]:
                        positive_polarities.append(True)
                    else:
                        negative_polarities.append(True)

            if True in positive_polarities and True not in negative_polarities:

                if polarity_matrix[index_polarity][0] == 1:
                    majority_count += 1
                    with_backup_count += 1
            elif True not in positive_polarities and True in negative_polarities:

                if polarity_matrix[index_polarity][2] == 1:
                    majority_count += 1
                    with_backup_count += 1

            elif True in positive_polarities and True in negative_polarities:

                indices_not_able_to_predict.append(index_polarity)
                if polarity_matrix[index_polarity][0] == 1:
                    majority_count += 1

            elif True not in positive_polarities and True not in negative_polarities:

                indices_not_able_to_predict.append(index_polarity)
                if polarity_matrix[index_polarity][0] == 1:
                    majority_count += 1

        return majority_count, with_backup_count, indices_not_able_to_predict

    def run(self):

        results = {
            'classification model': self.config.name_of_model,
            'size_of_training_set': len(self.internal_data_loader.lemmatized_training),
            'size_of_test_set': len(self.internal_data_loader.lemmatized_test),
            'size_of_cross_validation_sets': 0,
            'out_of_sample_accuracy_majority': 0,
            'in_sample_accuracy_majority': 0,
            'cross_val_accuracy_majority': [],
            'cross_val_mean_accuracy_majority': "cross validation is switched off",
            'cross_val_stdeviation_majority': "cross validation is switched off",
            'out_of_sample_accuracy_with_backup': "No backup model is used for ontology reasoner",
            'in_sample_accuracy_with_backup': "No backup model is used for ontology reasoner",
            'cross_val_accuracy_with_backup': [],
            'cross_val_mean_accuracy_with_backup': "cross validation is switched off",
            'cross_val_stdeviation_with_backup': "cross validation is switched off"
        }

        remaining_sentences = {
            'cross_val_indices': [],
            'test_set_indices': []
        }

        if self.config.cross_validation:

            training_indices, test_indices = self.internal_data_loader.get_random_indices_for_cross_validation(
                self.config.cross_validation_rounds)

            results['size_of_validation_sets'] = training_indices.size
            remaining_indices = []

            for i in range(self.config.cross_validation_rounds):

                majority_count_training, with_backup_count_training, _ = self.predict_sentiment_of_sentences(
                    sentence_polarities=self.internal_data_loader.word_polarities_training[training_indices[i]],
                    sentence_negation=self.internal_data_loader.negation_in_training[training_indices[i]],
                    polarity_matrix=self.internal_data_loader.polarity_matrix_training[training_indices[i]]
                )
                majority_count_validation, with_backup_count_training, remaining_validation_indices = \
                    self.predict_sentiment_of_sentences(
                        sentence_polarities=self.internal_data_loader.word_polarities_training[test_indices[i]],
                        sentence_negation = self.internal_data_loader.negation_in_training[test_indices[i]],
                        polarity_matrix=self.internal_data_loader.polarity_matrix_training[test_indices[i]]
                )

                accuracy_majority = majority_count_validation / results['size_of_validation_sets']
                accuracy_with_back = majority_count_validation / results['size_of_validation_sets']
                results['cross_val_accuracy_majority'].append(accuracy_majority)
                results['cross_val_accuracy_with_backup'].append(accuracy_with_back)

                remaining_indices.append(remaining_validation_indices)

            results['cross_val_mean_accuracy_majority'] = np.mean(results['cross_val_accuracy_majority'])
            results['cross_val_stdeviation_majority'] = np.std(results['cross_val_accuracy_majority'])

            results['cross_val_mean_accuracy_with_backup'] = np.mean(results['cross_val_accuracy_with_backup'])
            results['cross_val_stdeviation_with_backup'] = np.std(results['cross_val_accuracy_with_backup'])

            remaining_sentences['cross_val_indices'] = remaining_indices

        majority_count_training, with_backup_count_training, _ = self.predict_sentiment_of_sentences(
                    sentence_polarities=self.internal_data_loader.word_polarities_training,
                    sentence_negation=self.internal_data_loader.negation_in_training,
                    polarity_matrix=self.internal_data_loader.polarity_matrix_training
                )

        accuracy_majority_training = majority_count_training / results['size_of_training_set']
        accuracy_with_back_training = with_backup_count_training / results['size_of_training_set']
        results['in_sample_accuracy_majority'] = accuracy_majority_training
        results['in_sample_accuracy_with_backup'] = accuracy_with_back_training

        majority_count_test, with_backup_count_test, remaining_test_indices = self.predict_sentiment_of_sentences(
            sentence_polarities=self.internal_data_loader.word_polarities_test,
            sentence_negation=self.internal_data_loader.negation_in_test,
            polarity_matrix=self.internal_data_loader.polarity_matrix_test
        )

        accuracy_majority_test = majority_count_test / results['size_of_test_set']
        accuracy_with_back_test = with_backup_count_test / results['size_of_test_set']
        results['out_of_sample_accuracy_majority'] = accuracy_majority_test
        results['out_of_sample_accuracy_with_backup'] = accuracy_with_back_test

        remaining_sentences['test_set_indices'] = remaining_test_indices

        with open(self.config.file_of_results, 'w') as outfile:
            json.dump(results, outfile, ensure_ascii=False)

        with open(self.config.remaining_data, 'w') as outfile:
            json.dump(remaining_sentences, outfile, ensure_ascii=False)