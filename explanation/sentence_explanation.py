import numpy as np
import json
import os
from config import DiagnosticClassifierPOSConfig, DiagnosticClassifierAspectSentimentConfig, \
    DiagnosticClassifierRelationConfig, DiagnosticClassifierWordSentimentConfig, DiagnosticClassifierMentionConfig


class SentenceExplaining:

    def __init__(self, neural_language_model, diagnostic_classifiers, local_interpretable_model):
        self.neural_language_model = neural_language_model
        self.diagnostic_classifiers = diagnostic_classifiers
        self.local_interpretable_model = local_interpretable_model

    def explain_sentence(self, sentence_id):

        sentences_id = np.array(self.neural_language_model.internal_data_loader.sentence_id_in_training)
        indices = np.arange(sentences_id.shape[0])

        sentence_indices = indices[sentences_id == sentence_id]

        x_training = np.array(self.neural_language_model.internal_data_loader.word_embeddings_training_all)
        train_aspects = np.array(self.neural_language_model.internal_data_loader.aspect_indices_training)
        y_training = np.array(self.neural_language_model.internal_data_loader.polarity_matrix_training)
        lemmatized_sentence = self.neural_language_model.internal_data_loader.lemmatized_training
        original_sentence = self.neural_language_model.internal_data_loader.original_sentence_training

        sentence_explanations = []
        print("sentence_indices ", sentence_indices)

        for sentence_index in sentence_indices.tolist():

            x_left_part, x_target_part, x_right_part, x_left_sen_len, x_tar_len, x_right_sen_len = \
                self.neural_language_model.internal_data_loader.split_embeddings(
                    np.array([x_training[sentence_index]]), np.array([train_aspects[sentence_index]]),
                    self.neural_language_model.config.max_sentence_length,
                    self.neural_language_model.config.max_target_length)

            pred, layer_information = self.neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                                         x_left_sen_len, x_tar_len, x_right_sen_len)

            sentence_explanation = {
                'neural_language_model': self.neural_language_model.config.name_of_model,
                'lemmatized_sentence': lemmatized_sentence[sentence_index],
                'original_sentence': original_sentence[sentence_index],
                'prediction': pred[0].tolist(),
                'true_y': y_training[sentence_index].tolist(),
                'aspects': train_aspects[sentence_index],
                'sentence_id': sentence_id,
                'sentence_index': sentence_index
            }

            relevance_lr, intercept_lr, subsets_relevance_lr, intercept_slr, subsets_relevance_pd = \
                self.local_interpretable_model.single_run(x=np.array(x_training[sentence_index]),
                                                          y_pred=pred[0],
                                                          aspects_indices=np.array(train_aspects[sentence_index]),
                                                          lemmatized_sentence=lemmatized_sentence[sentence_index])

            weighted_hidden_state = {}
            print("relevance_lr ", relevance_lr)
            print("subsets_relevance_lr ", subsets_relevance_lr)
            print("subsets_relevance_pd ", subsets_relevance_pd)

            n_left_words = x_left_sen_len[0]

            for j in range(n_left_words):

                pos = relevance_lr[j][0] - intercept_lr[0]
                neu = relevance_lr[j][1] - intercept_lr[1]
                neg = relevance_lr[j][2] - intercept_lr[2]

                dict_of_word = {
                    'relevance_linear_regression': [pos, neu, neg]
                }

                left_word_embedding = x_left_part[0][j]
                left_hidden_state = layer_information['left_hidden_state'][0][0][j]

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        dict_of_word['attention_score_' + str(i)] = \
                            layer_information['left_attention_score_' + str(i)][0][0][0][j].tolist()
                        weighted_hidden_state['weighted_left_hidden_state_' + str(i)] = \
                            layer_information['weighted_left_hidden_state_' + str(i)][0][0][j].tolist()
                else:
                    dict_of_word['attention_score'] = layer_information['left_attention_score'][0][0][0][j].tolist()
                    weighted_hidden_state['weighted_left_hidden_state'] = \
                        layer_information['weighted_left_hidden_state'][0][0][j].tolist()

                # Diagnostic Classifier for Part of Speech Tagging
                file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_pos_tags'] = \
                        DiagnosticClassifierPOSConfig.classifier_embeddings.predict(np.array([left_word_embedding]),
                                                                                    file, '_le')[0][0].tolist()

                file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_pos_tags'] = \
                        DiagnosticClassifierPOSConfig.classifier_states.predict(np.array([left_hidden_state]), file,
                                                                                '_ls')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_pos_tags_' + str(i)] = \
                                DiagnosticClassifierPOSConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_left_hidden_state_' + str(i)]]), file,
                                    '_lw' + str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_pos_tags'] = \
                            DiagnosticClassifierPOSConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_left_hidden_state']]), file, '_lw')[0][0].tolist()

                # Diagnostic Classifier for sentiment towards the aspect
                file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_aspect_sentiments'] = \
                        DiagnosticClassifierAspectSentimentConfig.classifier_embeddings.predict(
                            np.array([left_word_embedding]), file, '_le')[0][0].tolist()

                file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_aspect_sentiments'] = \
                        DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
                            np.array([left_hidden_state]), file,'_ls')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_aspect_sentiments_' + str(i)] = \
                                DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
                                 np.array([weighted_hidden_state['weighted_left_hidden_state_' + str(i)]]), file,
                                 '_lw' + str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_aspect_sentiments'] = \
                            DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_left_hidden_state']]), file, '_lw')[0][0].tolist()

                # Diagnostic Classifier for relation towards the aspect
                file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_relations'] = \
                        DiagnosticClassifierRelationConfig.classifier_embeddings.predict(
                            np.array([left_word_embedding]), file,'_le')[0][0].tolist()

                file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_relations'] = \
                        DiagnosticClassifierRelationConfig.classifier_states.predict(np.array([left_hidden_state]),
                                                                                     file, '_ls')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):

                        file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_relations_' + str(i)] = \
                                DiagnosticClassifierRelationConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_left_hidden_state_' + str(i)]]),
                                    file, '_lw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_relations'] = \
                            DiagnosticClassifierRelationConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_left_hidden_state']]), file, '_lw')[0][0].tolist()

                # Diagnostic Classifier for word sentiments
                file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_word_sentiments'] = \
                        DiagnosticClassifierWordSentimentConfig.classifier_embeddings.predict(
                            np.array([left_word_embedding]), file, '_le')[0][0].tolist()

                file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_word_sentiments'] = \
                        DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
                            np.array([left_hidden_state]), file, '_ls')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):

                        file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_word_sentiments_' + str(i)] = \
                                DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_left_hidden_state_' + str(i)]]), file,
                                    '_lw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_word_sentiments'] = \
                            DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_left_hidden_state']]), file, '_lw')[0][0].tolist()

                # Diagnostic Classifier for word mentions
                file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_mentions'] = \
                        DiagnosticClassifierMentionConfig.classifier_embeddings.predict(
                            np.array([left_word_embedding]), file, '_le')[0][0].tolist()

                file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'left_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_mentions'] = \
                        DiagnosticClassifierMentionConfig.classifier_states.predict(np.array([left_hidden_state]), file,
                                                                                    '_ls')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_mentions_' + str(i)] = \
                                DiagnosticClassifierMentionConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_left_hidden_state_' + str(i)]]), file,
                                    '_lw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'left_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_mentions'] = \
                            DiagnosticClassifierMentionConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_left_hidden_state']]), file, '_lw')[0][0].tolist()

                sentence_explanation[lemmatized_sentence[sentence_index][j]] = dict_of_word

            n_right_words = x_right_sen_len[0]
            begin_index = train_aspects[sentence_index][0]
            end_index = train_aspects[sentence_index][-1]

            for j in range(n_right_words):

                pos = relevance_lr[begin_index + j][0] - intercept_lr[0]
                neu = relevance_lr[begin_index + j][1] - intercept_lr[1]
                neg = relevance_lr[begin_index + j][2] - intercept_lr[2]

                dict_of_word = {
                    'relevance_linear_regression': [pos, neu, neg]
                }

                right_word_embedding = x_right_part[0][j]
                right_hidden_state = layer_information['right_hidden_state'][0][0][j]

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":

                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        dict_of_word['attention_score_' + str(i)] = \
                            layer_information['right_attention_score_' + str(i)][0][0][0][j].tolist()
                        weighted_hidden_state['weighted_right_hidden_state_' + str(i)] = \
                            layer_information['weighted_right_hidden_state_' + str(i)][0][0][j]
                else:
                    dict_of_word['attention_score'] = layer_information['right_attention_score'][0][0][0][j].tolist()
                    weighted_hidden_state['weighted_right_hidden_state'] = \
                        layer_information['weighted_right_hidden_state'][0][0][j]

                # Diagnostic Classifier for Part of Speech Tagging
                file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_pos_tags'] = \
                        DiagnosticClassifierPOSConfig.classifier_embeddings.predict(np.array([right_word_embedding]), file, '_re')[0][0].tolist()

                file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_pos_tags'] = \
                        DiagnosticClassifierPOSConfig.classifier_states.predict(np.array([right_hidden_state]), file, '_rs')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_pos_tags_' + str(i)] = \
                                DiagnosticClassifierPOSConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_right_hidden_state_' + str(i)]]), file, '_rw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_pos_tags'] = \
                            DiagnosticClassifierPOSConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_right_hidden_state']]), file, '_rw')[0][0].tolist()

                # Diagnostic Classifier for sentiment towards the aspect
                file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_aspect_sentiments'] = \
                        DiagnosticClassifierAspectSentimentConfig.classifier_embeddings.predict(np.array([right_word_embedding]),
                                                                                                file, '_re')[0][0].tolist()
                file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_aspect_sentiments'] = \
                        DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(np.array([right_hidden_state]),
                                                                                            file, '_rs')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_aspect_sentiments_' + str(i)] = \
                                DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_right_hidden_state_' + str(i)]]), file, '_rw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_aspect_sentiments'] = \
                            DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_right_hidden_state']]), file, '_rw')[0][0].tolist()

                # Diagnostic Classifier for relation towards the aspect
                file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_relations'] = \
                        DiagnosticClassifierRelationConfig.classifier_embeddings.predict(np.array([right_word_embedding]),
                                                                                         file, '_re')[0][0].tolist()
                file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_relations'] = \
                        DiagnosticClassifierRelationConfig.classifier_states.predict(np.array([right_hidden_state]),
                                                                                     file, '_rs')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_relations_' + str(i)] = \
                                DiagnosticClassifierRelationConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_right_hidden_state_' + str(i)]]), file, '_rw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_relations'] = \
                            DiagnosticClassifierRelationConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_right_hidden_state']]), file, '_rw')[0][0].tolist()

                # Diagnostic Classifier for ontology mentions
                file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_mentions'] = \
                        DiagnosticClassifierMentionConfig.classifier_embeddings.predict(np.array([right_word_embedding]),
                                                                                        file, '_re')[0][0].tolist()
                file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_mentions'] = \
                        DiagnosticClassifierMentionConfig.classifier_states.predict(np.array([right_hidden_state]), file, '_rs')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_mentions_' + str(i)] = \
                                DiagnosticClassifierMentionConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_right_hidden_state_' + str(i)]]), file, '_rw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_mentions'] = \
                            DiagnosticClassifierMentionConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_right_hidden_state']]), file, '_rw')[0][0].tolist()

                # Diagnostic Classifier for word sentiments
                file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_embeddings_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['embedding_pred_word_sentiments'] = \
                        DiagnosticClassifierWordSentimentConfig.classifier_embeddings.predict(np.array([right_word_embedding]),
                                                                                              file, '_re')[0][0].tolist()
                file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                    self.neural_language_model.config.name_of_model, 'right_states_')
                if os.path.isfile(file + ".index"):
                    dict_of_word['hidden_states_pred_word_sentiments'] = \
                        DiagnosticClassifierWordSentimentConfig.classifier_states.predict(np.array([right_hidden_state]),
                                                                                          file, '_rs')[0][0].tolist()

                if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
                    for i in range(self.neural_language_model.config.n_iterations_hop):
                        file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                            self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
                        if os.path.isfile(file + ".index"):
                            dict_of_word['weighted_states_pred_word_sentiments_' + str(i)] = \
                                DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
                                    np.array([weighted_hidden_state['weighted_right_hidden_state_' + str(i)]]), file, '_rw'+str(i))[0][0].tolist()

                else:
                    file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
                        self.neural_language_model.config.name_of_model, 'right_weighted_')
                    if os.path.isfile(file + ".index"):
                        dict_of_word['weighted_states_pred_word_sentiments'] = \
                            DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
                                np.array([weighted_hidden_state['weighted_right_hidden_state']]), file, '_rw')[0][0].tolist()

                sentence_explanation[lemmatized_sentence[sentence_index][end_index + 1 + j]] = dict_of_word

            for counter in range(len(lemmatized_sentence[sentence_index]) - len(train_aspects[sentence_index])):
                attribute = subsets_relevance_lr[counter]
                for word_index in attribute['indices_attribute']:
                    word = lemmatized_sentence[sentence_index][word_index]
                    dict_of_word = sentence_explanation[word]
                    pos = attribute[0] - intercept_slr[0]
                    neu = attribute[1] - intercept_slr[1]
                    neg = attribute[2] - intercept_slr[2]
                    dict_of_word['subset_linear_reg'] = [pos, neu, neg]

            for counter in range(len(lemmatized_sentence[sentence_index]) - len(train_aspects[sentence_index])):
                attribute = subsets_relevance_pd[counter]
                for word_index in attribute['indices_attribute']:
                    word = lemmatized_sentence[sentence_index][word_index]
                    dict_of_word = sentence_explanation[word]
                    dict_of_word['subset_pred_dif'] = [attribute[0], attribute[1], attribute[2]]

            print("sentence_explanation ", sentence_explanation)
            sentence_explanations.append(sentence_explanation)

        file = self.neural_language_model.config.get_explanation_file(self.neural_language_model.config.name_of_model,
                                                                      sentence_id)
        with open(file, 'w') as outfile:
            json.dump(sentence_explanations, outfile, ensure_ascii=False)
