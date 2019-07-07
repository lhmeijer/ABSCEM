import numpy as np
import json


class OntologyReasoner:

    def __init__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    @staticmethod
    def predict_sentiment_of_sentences(sentence_polarities, lemmas, sentence_negations, polarity_matrix):

        number_of_sentences = sentence_polarities.shape[0]

        polarity_count = np.array([0, 0, 0])
        majority_count = np.array([0, 0, 0])
        with_backup_count = np.array([0, 0, 0])
        able_to_predict_count = np.array([0, 0, 0])
        concept_from_ontology = np.array([0, 0, 0, 0])
        indices_able_to_predict = []
        indices_not_able_to_predict = []

        for index_sentence in range(number_of_sentences):

            index_true_polarity = polarity_matrix[index_sentence].tolist().index(1)
            polarity_count[index_true_polarity] += 1

            number_of_words_in_sentence = len(sentence_polarities[index_sentence])
            positive_polarities = []
            negative_polarities = []

            negations = ['no', 'not', 'never', 'n\'t']

            for index_word in range(number_of_words_in_sentence):
                negation = False
                for negation_indices in sentence_negations[index_sentence]:

                    index = negation_indices[1]

                    if lemmas[index_sentence][negation_indices[0]] in negations:
                        index = negation_indices[0]
                    elif lemmas[index_sentence][negation_indices[1]] in negations:
                        index = negation_indices[1]

                    if index_word >= index - 2 and index_word <= index + 2:
                        negation = True

                if sentence_polarities[index_sentence][index_word] == [1, 0, 0]:

                    if negation:
                        negative_polarities.append(True)
                    else:
                        positive_polarities.append(True)

                elif sentence_polarities[index_sentence][index_word] == [0, 1, 0]:

                    if negation:
                        positive_polarities.append(True)
                    else:
                        negative_polarities.append(True)

            polarity_argmax = np.argmax(polarity_matrix[index_sentence])
            if True in positive_polarities and True not in negative_polarities:

                concept_from_ontology[0] += 1
                if polarity_matrix[index_sentence][1] == 1:
                    indices_not_able_to_predict.append(index_sentence)
                else:
                    indices_able_to_predict.append(index_sentence)
                    able_to_predict_count[polarity_argmax] += 1

                if polarity_matrix[index_sentence][0] == 1:
                    majority_count[0] += 1
                    with_backup_count[0] += 1
            elif True not in positive_polarities and True in negative_polarities:

                concept_from_ontology[1] += 1
                if polarity_matrix[index_sentence][1] == 1:
                    indices_not_able_to_predict.append(index_sentence)
                else:
                    indices_able_to_predict.append(index_sentence)
                    able_to_predict_count[polarity_argmax] += 1

                if polarity_matrix[index_sentence][2] == 1:
                    majority_count[2] += 1
                    with_backup_count[2] += 1

            elif True in positive_polarities and True in negative_polarities:

                concept_from_ontology[2] += 1
                indices_not_able_to_predict.append(index_sentence)
                if polarity_matrix[index_sentence][0] == 1:
                    majority_count[0] += 1

            elif True not in positive_polarities and True not in negative_polarities:

                concept_from_ontology[3] += 1
                indices_not_able_to_predict.append(index_sentence)
                if polarity_matrix[index_sentence][0] == 1:
                    majority_count[0] += 1

        return majority_count, with_backup_count, polarity_count, indices_not_able_to_predict, able_to_predict_count, \
               concept_from_ontology, indices_able_to_predict

    def run(self):

        x_train = np.array(self.internal_data_loader.word_polarities_training)
        train_negations = np.array(self.internal_data_loader.negation_in_training)
        train_lemmas = np.array(self.internal_data_loader.lemmatized_training)
        y_train = np.array(self.internal_data_loader.polarity_matrix_training)

        x_test = np.array(self.internal_data_loader.word_polarities_test)
        test_negations = np.array(self.internal_data_loader.negation_in_test)
        test_lemmas = np.array(self.internal_data_loader.lemmatized_test)
        y_test = np.array(self.internal_data_loader.polarity_matrix_test)

        if self.config.cross_validation:

            training_indices, validation_indices = self.internal_data_loader.get_indices_cross_validation()

            results = {}
            val_remaining_indices = []
            tr_remaining_indices = []

            val_able_to_pred_indices = []
            tr_able_to_pred_indices = []

            train_single_acc_majority = list(range(self.config.cross_validation_rounds))
            train_single_acc_with_backup = list(range(self.config.cross_validation_rounds))
            validation_single_acc_majority = list(range(self.config.cross_validation_rounds))
            validation_single_acc_with_backup = list(range(self.config.cross_validation_rounds))

            for i in range(self.config.cross_validation_rounds):

                x_train_cross = x_train[training_indices[i]]
                y_train_cross = y_train[training_indices[i]]
                train_lemmas_cross = train_lemmas[training_indices[i]]
                train_negations_cross = train_negations[training_indices[i]]

                x_validation_cross = x_train[validation_indices[i]]
                y_validation_cross = y_train[validation_indices[i]]
                validation_lemmas_cross = train_lemmas[validation_indices[i]]
                validation_negations_cross = train_negations[validation_indices[i]]

                majority_count_training, with_backup_count_training, count_training, wrong_tr_indices, \
                tr_pol_able_to_pred, tr_able_to_pred, correct_tr_indices = self.predict_sentiment_of_sentences(
                        sentence_polarities=x_train_cross,
                        lemmas=train_lemmas_cross,
                        sentence_negations=train_negations_cross,
                        polarity_matrix=y_train_cross
                    )

                accuracy_majority_training = majority_count_training / count_training

                majority_count_validation, with_backup_count_validation, count_validation, wrong_val_indices,  \
                val_pol_able_to_pred, val_able_to_pred, correct_val_indices = self.predict_sentiment_of_sentences(
                    sentence_polarities=x_validation_cross,
                    lemmas=validation_lemmas_cross,
                    sentence_negations=validation_negations_cross,
                    polarity_matrix=y_validation_cross
                )

                val_remaining_indices.append(wrong_val_indices)
                tr_remaining_indices.append(wrong_tr_indices)
                val_able_to_pred_indices.append(correct_val_indices)
                tr_able_to_pred_indices.append(correct_tr_indices)

                accuracy_majority_validation = majority_count_validation / count_validation

                train_single_acc_majority[i] = np.sum(majority_count_training) / np.sum(count_training)
                train_single_acc_with_backup[i] = np.sum(with_backup_count_training) / np.sum(tr_pol_able_to_pred)
                validation_single_acc_majority[i] = np.sum(majority_count_validation) / np.sum(count_validation)
                validation_single_acc_with_backup[i] = np.sum(with_backup_count_validation) / np.sum(val_pol_able_to_pred)

                result = {
                    'classification model': self.config.name_of_model,
                    'n_in_training_sample': count_training.tolist(),
                    'n_in_validation_sample': count_validation.tolist(),
                    'train_single_acc_majority': train_single_acc_majority[i],
                    'train_single_acc_with_backup': train_single_acc_with_backup[i],
                    'test_single_acc_majority': validation_single_acc_majority[i],
                    'test_single_acc_with_backup': validation_single_acc_with_backup[i],
                    'train_acc_majority': accuracy_majority_training.tolist(),
                    'validation_acc_majority': accuracy_majority_validation.tolist(),
                    'train_pred_y_majority': majority_count_training.tolist(),
                    'train_pred_y_with_backup': with_backup_count_training.tolist(),
                    'train_polarities_able_to_predict': tr_pol_able_to_pred.tolist(),
                    'train_able_to_predict': tr_able_to_pred.tolist(),
                    'validation_pred_y_majority': majority_count_validation.tolist(),
                    'validation_pred_y_with_backup': with_backup_count_validation.tolist(),
                    'validation_polarities_able_to_predict': val_pol_able_to_pred.tolist(),
                    'validation_able_to_predict': val_able_to_pred.tolist()
                }

                results['cross_validation_round_' + str(i)] = result

            remaining_sentences = {
                'tr_able_to_pred': tr_able_to_pred_indices,
                'te_able_to_pred': val_able_to_pred_indices,
                'tr_not_able_to_pred': tr_remaining_indices,
                'te_not_able_to_pred': val_remaining_indices

            }

            with open(self.config.remaining_data_cross_val, 'w') as outfile:
                json.dump(remaining_sentences, outfile, ensure_ascii=False)

            results['mean_accuracy_train_single_acc_majority'] = np.mean(train_single_acc_majority)
            results['standard_deviation_train_single_acc_majority'] = np.std(train_single_acc_majority)
            results['mean_accuracy_train_single_acc_with_backup'] = np.mean(train_single_acc_with_backup)
            results['standard_deviation_train_single_acc_with_backup'] = np.std(train_single_acc_with_backup)
            results['mean_accuracy_validation_single_acc_majority'] = np.mean(validation_single_acc_majority)
            results['standard_deviation_validation_single_acc_majority'] = np.std(validation_single_acc_majority)
            results['mean_accuracy_validation_single_acc_with_backup'] = np.mean(validation_single_acc_with_backup)
            results['standard_deviation_validation_single_acc_with_backup'] = np.std(validation_single_acc_with_backup)

            with open(self.config.file_of_cross_val_results, 'w') as outfile:
                json.dump(results, outfile, ensure_ascii=False, indent=0)

        else:

            majority_count_training, with_backup_count_training, count_training, tr_not_able_to_pred_indices, \
            tr_pol_able_to_pred, tr_able_to_pred, tr_able_to_pred_indices = self.predict_sentiment_of_sentences(
                sentence_polarities=x_train,
                lemmas=train_lemmas,
                sentence_negations=train_negations,
                polarity_matrix=y_train
            )

            accuracy_majority_training = majority_count_training / count_training

            majority_count_test, with_backup_count_test, count_test, te_not_able_to_pred_indices, te_pol_able_to_pred, \
            te_able_to_pred, te_able_to_pred_indices = self.predict_sentiment_of_sentences(
                sentence_polarities=x_test,
                lemmas=test_lemmas,
                sentence_negations=test_negations,
                polarity_matrix=y_test
            )

            remaining_sentences = {
                'tr_able_to_pred': tr_able_to_pred_indices,
                'te_able_to_pred': te_able_to_pred_indices,
                'tr_not_able_to_pred': tr_not_able_to_pred_indices,
                'te_not_able_to_pred': te_not_able_to_pred_indices
            }

            with open(self.config.remaining_data, 'w') as outfile:
                json.dump(remaining_sentences, outfile, ensure_ascii=False)

            accuracy_majority_test = majority_count_test / count_test

            results = {
                'classification model': self.config.name_of_model,
                'n_in_training_sample': count_training.tolist(),
                'n_in_test_sample': count_test.tolist(),
                'train_single_acc_majority': np.sum(majority_count_training) / np.sum(count_training),
                'train_single_acc_with_backup': np.sum(with_backup_count_training) / np.sum(count_training),
                'test_single_acc_majority': np.sum(majority_count_test) / np.sum(count_test),
                'test_single_acc_with_backup': np.sum(with_backup_count_test) / np.sum(count_test),
                'train_acc_majority': accuracy_majority_training.tolist(),
                'test_acc_majority': accuracy_majority_test.tolist(),
                'train_pred_y_majority': majority_count_training.tolist(),
                'train_pred_y_with_backup': with_backup_count_training.tolist(),
                'train_polarities_able_to_predict': tr_pol_able_to_pred.tolist(),
                'train_able_to_predict': tr_able_to_pred.tolist(),
                'test_pred_y_majority': majority_count_test.tolist(),
                'test_pred_y_with_backup': with_backup_count_test.tolist(),
                'test_polarities_able_to_predict': te_pol_able_to_pred.tolist(),
                'test_able_to_predict': te_able_to_pred.tolist()
            }

            with open(self.config.file_of_results, 'w') as outfile:
                json.dump(results, outfile, ensure_ascii=False, indent=0)
