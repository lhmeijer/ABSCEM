import numpy as np
from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierPolarityConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierMentionConfig


class SentenceExplaining:

    def __init__(self, neural_language_model, diagnostic_classifiers, local_interpretable_model):
        self.neural_language_model = neural_language_model
        self.diagnostic_classifiers = diagnostic_classifiers
        self.local_interpretable_model = local_interpretable_model

    def explain_sentence(self, sentence_id):

        sentences_id = self.neural_language_model.internal_data_loader.sentence_id_in_training

        sentence_index = None
        try:
            sentence_index = sentences_id.index(sentence_id)
        except ValueError:
            print(sentence_id + "is not in sentences_id")

        x_training = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        y_training = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)
        lemmatized_sentence = self.neural_language_model.internal_data_loader.lemmatized_training

        x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
            self.neural_language_model.config.split_embeddings(np.array([x_training[sentence_index]]),
                                                               np.array([train_aspects[sentence_index]]),
                                                               self.neural_language_model.config.max_sentence_length,
                                                               self.neural_language_model.config.max_target_length)

        pred, layer_information = self.neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                                     x_left_sen_len, x_tar_len, x_right_sen_len)

        sentence_explanation = {
            'neural_language_model': self.neural_language_model.config.name_of_model
        }

        relevance_lr, relevance_pd, subsets_relevance_lr, subsets_relevance_pd = \
            self.local_interpretable_model.single_run(x=np.array([x_training[sentence_index]]),
                                                      aspects_polarity=y_training[sentence_index],
                                                      y_pred=pred,
                                                      aspects_indices=np.array([train_aspects[sentence_index]]),
                                                      lemmatized_sentence=lemmatized_sentence[sentence_index])

        weighted_hidden_state = {}

        n_left_words = x_left_sen_len[0]

        for j in range(n_left_words):

            dict_of_word = {
                'relevance_linear_regression': relevance_lr[j],
                'relevance_pred_difference': relevance_pd[j]
            }

            left_word_embedding = x_left_part[0][j].tolist()
            left_hidden_state = layer_information['left_hidden_state'][0][j].tolist()

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                for i in range(self.neural_language_model.config.n_iterations_hop):
                    dict_of_word['attention_score' + str(i)] = \
                        layer_information['left_attention_score' + str(i)][0][j].tolist()
                    weighted_hidden_state['weighted_left_hidden_state' + str(i)] = \
                        layer_information['weighted_left_hidden_state' + str(i)][0][j].tolist()
            else:
                dict_of_word['attention_score'] = layer_information['left_attention_score'][0][j].tolist()
                weighted_hidden_state['weighted_left_hidden_state'] = \
                    layer_information['weighted_left_hidden_state'][0][j].tolist()

            # Diagnostic Classifier for Part of Speech Tagging
            file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_embeddings')
            dict_of_word['embedding_pred_pos_tags'] = \
                DiagnosticClassifierPOSConfig.classifier.predict(left_word_embedding, file)
            file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_states')
            dict_of_word['hidden_states_pred_pos_tags'] = \
                DiagnosticClassifierPOSConfig.classifier.predict(left_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted' + str(i))
                    dict_of_word['weighted_states_pred_pos_tags' + str(i)] = \
                        DiagnosticClassifierPOSConfig.classifier.predict(
                            weighted_hidden_state['weighted_left_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted')
                dict_of_word['weighted_states_pred_pos_tags'] = \
                    DiagnosticClassifierPOSConfig.classifier.predict(
                        weighted_hidden_state['weighted_left_hidden_state'], file)

            # Diagnostic Classifier for Polarity towards the aspect
            file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_embeddings')
            dict_of_word['embedding_pred_polarities'] = \
                DiagnosticClassifierPolarityConfig.classifier.predict(left_word_embedding, file)
            file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_states')
            dict_of_word['hidden_states_pred_polarities'] = \
                DiagnosticClassifierPolarityConfig.classifier.predict(left_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted' + str(i))
                    dict_of_word['weighted_states_pred_polarities' + str(i)] = \
                        DiagnosticClassifierPolarityConfig.classifier.predict(
                            weighted_hidden_state['weighted_left_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted')
                dict_of_word['weighted_states_pred_polarities'] = \
                    DiagnosticClassifierPolarityConfig.classifier.predict(
                        weighted_hidden_state['weighted_left_hidden_state'], file)

            # Diagnostic Classifier for relation towards the aspect
            file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_embeddings')
            dict_of_word['embedding_pred_relations'] = \
                DiagnosticClassifierRelationConfig.classifier.predict(left_word_embedding, file)
            file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_states')
            dict_of_word['hidden_states_pred_relations'] = \
                DiagnosticClassifierRelationConfig.classifier.predict(left_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted' + str(i))
                    dict_of_word['weighted_states_pred_relations' + str(i)] = \
                        DiagnosticClassifierRelationConfig.classifier.predict(
                            weighted_hidden_state['weighted_left_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted')
                dict_of_word['weighted_states_pred_relations'] = \
                    DiagnosticClassifierRelationConfig.classifier.predict(
                        weighted_hidden_state['weighted_left_hidden_state'], file)

            # Diagnostic Classifier for word mentions
            file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_embeddings')
            dict_of_word['embedding_pred_mentions'] = \
                DiagnosticClassifierMentionConfig.classifier.predict(left_word_embedding, file)
            file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'left_states')
            dict_of_word['hidden_states_pred_mentions'] = \
                DiagnosticClassifierMentionConfig.classifier.predict(left_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted' + str(i))
                    dict_of_word['weighted_states_pred_mentions' + str(i)] = \
                        DiagnosticClassifierMentionConfig.classifier.predict(
                            weighted_hidden_state['weighted_left_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_weighted')
                dict_of_word['weighted_states_pred_mentions'] = \
                    DiagnosticClassifierMentionConfig.classifier.predict(
                        weighted_hidden_state['weighted_left_hidden_state'], file)

            sentence_explanation[lemmatized_sentence[sentence_index][j]] = dict_of_word

        n_right_words = x_right_sen_len[0]
        end_index = train_aspects[sentence_index][-1]

        for j in range(n_right_words):

            dict_of_word = {
                'relevance_linear_regression': relevance_lr[end_index + 1 + j],
                'relevance_pred_difference': relevance_pd[end_index + 1 + j]
            }

            right_word_embedding = x_right_part[0][j].tolist()
            right_hidden_state = layer_information['right_hidden_state'][0][j].tolist()

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                for i in range(self.neural_language_model.config.n_iterations_hop):
                    dict_of_word['attention_score' + str(i)] = \
                        layer_information['right_attention_score' + str(i)][0][j].tolist()
                    weighted_hidden_state['weighted_right_hidden_state' + str(i)] = \
                        layer_information['weighted_right_hidden_state' + str(i)][0][j].tolist()
            else:
                dict_of_word['attention_score'] = layer_information['right_attention_score'][0][j].tolist()
                weighted_hidden_state['weighted_right_hidden_state'] = \
                    layer_information['weighted_right_hidden_state'][0][j].tolist()

            # Diagnostic Classifier for Part of Speech Tagging
            file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_embeddings')
            dict_of_word['embedding_pred_pos_tags'] = \
                DiagnosticClassifierPOSConfig.classifier.predict(right_word_embedding, file)
            file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_states')
            dict_of_word['hidden_states_pred_pos_tags'] = \
                DiagnosticClassifierPOSConfig.classifier.predict(right_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted' + str(i))
                    dict_of_word['weighted_states_pred_pos_tags' + str(i)] = \
                        DiagnosticClassifierPOSConfig.classifier.predict(
                            weighted_hidden_state['weighted_right_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted')
                dict_of_word['weighted_states_pred_pos_tags'] = \
                    DiagnosticClassifierPOSConfig.classifier.predict(
                        weighted_hidden_state['weighted_right_hidden_state'], file)

            # Diagnostic Classifier for Polarity towards the aspect
            file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_embeddings')
            dict_of_word['embedding_pred_polarities'] = \
                DiagnosticClassifierPolarityConfig.classifier.predict(right_word_embedding, file)
            file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_states')
            dict_of_word['hidden_states_pred_polarities'] = \
                DiagnosticClassifierPolarityConfig.classifier.predict(right_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted' + str(i))
                    dict_of_word['weighted_states_pred_polarities' + str(i)] = \
                        DiagnosticClassifierPolarityConfig.classifier.predict(
                            weighted_hidden_state['weighted_right_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierPolarityConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted')
                dict_of_word['weighted_states_pred_polarities'] = \
                    DiagnosticClassifierPolarityConfig.classifier.predict(
                        weighted_hidden_state['weighted_right_hidden_state'], file)

            # Diagnostic Classifier for relation towards the aspect
            file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_embeddings')
            dict_of_word['embedding_pred_relations'] = \
                DiagnosticClassifierRelationConfig.classifier.predict(right_word_embedding, file)
            file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_states')
            dict_of_word['hidden_states_pred_relations'] = \
                DiagnosticClassifierRelationConfig.classifier.predict(right_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted' + str(i))
                    dict_of_word['weighted_states_pred_relations' + str(i)] = \
                        DiagnosticClassifierRelationConfig.classifier.predict(
                            weighted_hidden_state['weighted_right_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted')
                dict_of_word['weighted_states_pred_relations'] = \
                    DiagnosticClassifierRelationConfig.classifier.predict(
                        weighted_hidden_state['weighted_right_hidden_state'], file)

            # Diagnostic Classifier for word mentions
            file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_embeddings')
            dict_of_word['embedding_pred_mentions'] = \
                DiagnosticClassifierMentionConfig.classifier.predict(right_word_embedding, file)
            file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                self.neural_language_model.config.name_of_model, 'right_states')
            dict_of_word['hidden_states_pred_mentions'] = \
                DiagnosticClassifierMentionConfig.classifier.predict(right_hidden_state, file)

            if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                for i in range(self.neural_language_model.config.n_iterations_hop):
                    file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted' + str(i))
                    dict_of_word['weighted_states_pred_mentions' + str(i)] = \
                        DiagnosticClassifierMentionConfig.classifier.predict(
                            weighted_hidden_state['weighted_right_hidden_state' + str(i)], file)

            else:
                file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_weighted')
                dict_of_word['weighted_states_pred_mentions'] = \
                    DiagnosticClassifierMentionConfig.classifier.predict(
                        weighted_hidden_state['weighted_right_hidden_state'], file)

            sentence_explanation[lemmatized_sentence[sentence_index][end_index + 1 + j]] = dict_of_word

        counter = 0
        for attribute in subsets_relevance_lr:
            counter += 1
            for word in attribute['word_attribute']:
                dict_of_word = sentence_explanation[word]
                dict_of_word['subset_linear_reg' + str(counter)] = attribute

        counter = 0
        for attribute in subsets_relevance_pd:
            counter += 1
            for word in attribute['word_attribute']:
                dict_of_word = sentence_explanation[word]
                dict_of_word['subset_pred_dif' + str(counter)] = attribute

        print("sentence_explanation ", sentence_explanation)
