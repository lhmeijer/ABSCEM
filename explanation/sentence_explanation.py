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
            pred1, layer_information = self.neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                                         x_left_sen_len, x_tar_len, x_right_sen_len)
            pred2, layer_information = self.neural_language_model.predict(x_left_part, x_target_part, x_right_part,
                                                                         x_left_sen_len, x_tar_len, x_right_sen_len)

            print("pred ", pred)
            print("pred1 ", pred1)
            print("pred2 ", pred2)
        #     sentence_explanation = {
        #         'neural_language_model': self.neural_language_model.config.name_of_model,
        #         'lemmatized_sentence': lemmatized_sentence[sentence_index],
        #         'prediction': pred,
        #         'true_y': y_training[sentence_index],
        #         'aspects': train_aspects[sentence_index]
        #     }
        #
        #     relevance_lr, relevance_pd, subsets_relevance_lr, subsets_relevance_pd = \
        #         self.local_interpretable_model.single_run(x=np.array(x_training[sentence_index]),
        #                                                   aspects_polarity=y_training[sentence_index],
        #                                                   y_pred=pred,
        #                                                   aspects_indices=np.array(train_aspects[sentence_index]),
        #                                                   lemmatized_sentence=lemmatized_sentence[sentence_index])
        #
        #     weighted_hidden_state = {}
        #     print("relevance_lr ", relevance_lr)
        #     print("relevance_pd ", relevance_pd)
        #     print("subsets_relevance_lr ", subsets_relevance_lr)
        #     print("subsets_relevance_pd ", subsets_relevance_pd)
        #
        #     n_left_words = x_left_sen_len[0]
        #
        #     for j in range(n_left_words):
        #
        #         dict_of_word = {
        #             'relevance_linear_regression': relevance_lr[j],
        #             'relevance_pred_difference': relevance_pd[j]
        #         }
        #
        #         left_word_embedding = x_left_part[0][j]
        #         print("left_word_embedding.shape ", left_word_embedding.shape)
        #         left_hidden_state = layer_information['left_hidden_state'][0][j]
        #         print("left_hidden_state.shape ", left_hidden_state.shape)
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 dict_of_word['attention_score_' + str(i)] = \
        #                     layer_information['left_attention_score_' + str(i)][0][0][j].tolist()
        #                 weighted_hidden_state['weighted_left_hidden_state_' + str(i)] = \
        #                     layer_information['weighted_left_hidden_state_' + str(i)][0][j].tolist()
        #         else:
        #             dict_of_word['attention_score'] = layer_information['left_attention_score'][0][0][j].tolist()
        #             weighted_hidden_state['weighted_left_hidden_state'] = \
        #                 layer_information['weighted_left_hidden_state'][0][j].tolist()
        #
        #         # Diagnostic Classifier for Part of Speech Tagging
        #         file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_pos_tags'] = \
        #                 DiagnosticClassifierPOSConfig.classifier_embeddings.predict(left_word_embedding, file, '_le')
        #
        #         file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_pos_tags'] = \
        #                 DiagnosticClassifierPOSConfig.classifier_states.predict(left_hidden_state, file, '_ls')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_pos_tags_' + str(i)] = \
        #                         DiagnosticClassifierPOSConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_left_hidden_state_' + str(i)], file, '_lw' + str(i))
        #
        #         else:
        #             file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'left_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_pos_tags'] = \
        #                     DiagnosticClassifierPOSConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_left_hidden_state'], file, '_lw')
        #
        #         # Diagnostic Classifier for sentiment towards the aspect
        #         file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_aspect_sentiments'] = \
        #                 DiagnosticClassifierAspectSentimentConfig.classifier_embeddings.predict(left_word_embedding,
        #                                                                                         file, '_le')
        #         if os.path.isfile(file + ".index"):
        #             file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'left_states_')
        #             dict_of_word['hidden_states_pred_aspect_sentiments'] = \
        #                 DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(left_hidden_state, file,
        #                                                                                     '_ls')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_aspect_sentiments_' + str(i)] = \
        #                         DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
        #                          weighted_hidden_state['weighted_left_hidden_state_' + str(i)], file, '_lw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'left_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_aspect_sentiments'] = \
        #                     DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_left_hidden_state'], file, '_lw')
        #
        #         # Diagnostic Classifier for relation towards the aspect
        #         file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_relations'] = \
        #                 DiagnosticClassifierRelationConfig.classifier_embeddings.predict(left_word_embedding, file,
        #                                                                                  '_le')
        #         file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_relations'] = \
        #                 DiagnosticClassifierRelationConfig.classifier_states.predict(left_hidden_state, file, '_ls')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #
        #                 file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_relations_' + str(i)] = \
        #                         DiagnosticClassifierRelationConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_left_hidden_state_' + str(i)], file, '_lw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'left_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_relations'] = \
        #                     DiagnosticClassifierRelationConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_left_hidden_state'], file, '_lw')
        #
        #         # Diagnostic Classifier for word sentiments
        #         file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_word_sentiments'] = \
        #                 DiagnosticClassifierWordSentimentConfig.classifier_embeddings.predict(left_word_embedding, file,
        #                                                                                       '_le')
        #         file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_word_sentiments'] = \
        #                 DiagnosticClassifierWordSentimentConfig.classifier_states.predict(left_hidden_state, file,
        #                                                                                   '_ls')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #
        #                 file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_word_sentiments_' + str(i)] = \
        #                         DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_left_hidden_state_' + str(i)], file, '_lw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'left_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_word_sentiments'] = \
        #                     DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_left_hidden_state'], file, '_lw')
        #
        #         # Diagnostic Classifier for word mentions
        #         file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_mentions'] = \
        #                 DiagnosticClassifierMentionConfig.classifier_embeddings.predict(left_word_embedding, file,
        #                                                                                 '_le')
        #         file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'left_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_mentions'] = \
        #                 DiagnosticClassifierMentionConfig.classifier_states.predict(left_hidden_state, file, '_ls')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'left_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_mentions_' + str(i)] = \
        #                         DiagnosticClassifierMentionConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_left_hidden_state_' + str(i)], file, '_lw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'left_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_mentions'] = \
        #                     DiagnosticClassifierMentionConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_left_hidden_state'], file, '_lw')
        #
        #         sentence_explanation[lemmatized_sentence[sentence_index][j]] = dict_of_word
        #
        #     n_right_words = x_right_sen_len[0]
        #     end_index = train_aspects[sentence_index][-1]
        #
        #     for j in range(n_right_words):
        #
        #         dict_of_word = {
        #             'relevance_linear_regression': relevance_lr[end_index + 1 + j],
        #             'relevance_pred_difference': relevance_pd[end_index + j]
        #         }
        #
        #         right_word_embedding = np.array([x_right_part[0][j]])
        #         print("right_word_embedding.shape ", right_word_embedding.shape)
        #         right_hidden_state =  np.array([layer_information['right_hidden_state'][0][j]])
        #         print("right_hidden_state.shape ", right_hidden_state.shape)
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 dict_of_word['attention_score_' + str(i)] = \
        #                     layer_information['right_attention_score_' + str(i)][0][0][j].tolist()
        #                 weighted_hidden_state['weighted_right_hidden_state_' + str(i)] = \
        #                     np.array([layer_information['weighted_right_hidden_state_' + str(i)][0][j]])
        #         else:
        #             dict_of_word['attention_score'] = layer_information['right_attention_score'][0][0][j].tolist()
        #             weighted_hidden_state['weighted_right_hidden_state'] = \
        #                 np.array([layer_information['weighted_right_hidden_state'][0][j]])
        #
        #         # Diagnostic Classifier for Part of Speech Tagging
        #         file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_pos_tags'] = \
        #                 DiagnosticClassifierPOSConfig.classifier_embeddings.predict(right_word_embedding, file, '_re')
        #         file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_pos_tags'] = \
        #                 DiagnosticClassifierPOSConfig.classifier_states.predict(right_hidden_state, file, '_rs')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_pos_tags_' + str(i)] = \
        #                         DiagnosticClassifierPOSConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_right_hidden_state_' + str(i)], file, '_rw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierPOSConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'right_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_pos_tags'] = \
        #                     DiagnosticClassifierPOSConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_right_hidden_state'], file, '_rw')
        #
        #         # Diagnostic Classifier for sentiment towards the aspect
        #         file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_aspect_sentiments'] = \
        #                 DiagnosticClassifierAspectSentimentConfig.classifier_embeddings.predict(right_word_embedding,
        #                                                                                         file, '_re')
        #         file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_aspect_sentiments'] = \
        #                 DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(right_hidden_state,
        #                                                                                     file, '_rs')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_aspect_sentiments_' + str(i)] = \
        #                         DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_right_hidden_state_' + str(i)], file, '_rw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierAspectSentimentConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'right_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_aspect_sentiments'] = \
        #                     DiagnosticClassifierAspectSentimentConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_right_hidden_state'], file, '_rw')
        #
        #         # Diagnostic Classifier for relation towards the aspect
        #         file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_relations'] = \
        #                 DiagnosticClassifierRelationConfig.classifier_embeddings.predict(right_word_embedding,
        #                                                                                  file, '_re')
        #         file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_relations'] = \
        #                 DiagnosticClassifierRelationConfig.classifier_states.predict(right_hidden_state,
        #                                                                              file, '_rs')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_relations_' + str(i)] = \
        #                         DiagnosticClassifierRelationConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_right_hidden_state_' + str(i)], file, '_rw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierRelationConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'right_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_relations'] = \
        #                     DiagnosticClassifierRelationConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_right_hidden_state'], file, '_rw')
        #
        #         # Diagnostic Classifier for ontology mentions
        #         file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_mentions'] = \
        #                 DiagnosticClassifierMentionConfig.classifier_embeddings.predict(right_word_embedding,
        #                                                                                 file, '_re')
        #         file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_mentions'] = \
        #                 DiagnosticClassifierMentionConfig.classifier_states.predict(right_hidden_state, file, '_rs')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_mentions_' + str(i)] = \
        #                         DiagnosticClassifierMentionConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_right_hidden_state_' + str(i)], file, '_rw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierMentionConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'right_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_mentions'] = \
        #                     DiagnosticClassifierMentionConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_right_hidden_state'], file, '_rw')
        #
        #         # Diagnostic Classifier for word sentiments
        #         file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_embeddings_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['embedding_pred_word_sentiments'] = \
        #                 DiagnosticClassifierWordSentimentConfig.classifier_embeddings.predict(right_word_embedding,
        #                                                                                       file, '_re')
        #         file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #             self.neural_language_model.config.name_of_model, 'right_states_')
        #         if os.path.isfile(file + ".index"):
        #             dict_of_word['hidden_states_pred_word_sentiments'] = \
        #                 DiagnosticClassifierWordSentimentConfig.classifier_states.predict(right_hidden_state,
        #                                                                                   file, '_rs')
        #
        #         if self.neural_language_model.config.name_of_model == "LCR_Rot_hop_model":
        #             for i in range(self.neural_language_model.config.n_iterations_hop):
        #                 file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #                     self.neural_language_model.config.name_of_model, 'right_weighted_' + str(i) + "_")
        #                 if os.path.isfile(file + ".index"):
        #                     dict_of_word['weighted_states_pred_word_sentiments_' + str(i)] = \
        #                         DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
        #                             weighted_hidden_state['weighted_right_hidden_state_' + str(i)], file, '_rw'+str(i))
        #
        #         else:
        #             file = DiagnosticClassifierWordSentimentConfig.get_file_of_model_savings(
        #                 self.neural_language_model.config.name_of_model, 'right_weighted_')
        #             if os.path.isfile(file + ".index"):
        #                 dict_of_word['weighted_states_pred_word_sentiments'] = \
        #                     DiagnosticClassifierWordSentimentConfig.classifier_states.predict(
        #                         weighted_hidden_state['weighted_right_hidden_state'], file, '_rw')
        #
        #         sentence_explanation[lemmatized_sentence[sentence_index][end_index + 1 + j]] = dict_of_word
        #
        #     counter = 0
        #     for attribute in subsets_relevance_lr:
        #         counter += 1
        #         for word in attribute['word_attribute']:
        #             dict_of_word = sentence_explanation[word]
        #             dict_of_word['subset_linear_reg' + str(counter)] = attribute
        #
        #     counter = 0
        #     for attribute in subsets_relevance_pd:
        #         counter += 1
        #         for word in attribute['word_attribute']:
        #             dict_of_word = sentence_explanation[word]
        #             dict_of_word['subset_pred_dif' + str(counter)] = attribute
        #
        #     print("sentence_explanation ", sentence_explanation)
        #     sentence_explanations.append(sentence_explanation)
        #
        # file = self.neural_language_model.config.get_explanation_file(self.neural_language_model.config.name_of_model,
        #                                                               sentence_id)
        # with open(file, 'w') as outfile:
        #     json.dump(sentence_explanations, outfile, ensure_ascii=False)
