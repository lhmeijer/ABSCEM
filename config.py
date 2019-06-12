import tensorflow as tf
import numpy as np
from diagnostic_classifier.classifiers import SingleMLPClassifier
from local_interpretable_model.attribute_evaluator import LASSORegression, PredictionDifference
from local_interpretable_model.locality_algorithms import Perturbing
from local_interpretable_model.rule_based_classifier import RuleBasedClassifier


class Config:

    year = 2015
    embedding_dimension = 300
    hybrid_method = False
    cross_validation_rounds = 10
    cross_validation_percentage = 0.8
    cross_validation = False
    seed = 100

    external_train_data = "data/external_data/restaurant_train_" + str(year) + ".xml"
    external_test_data = "data/external_data/restaurant_test_" + str(year) + ".xml"

    internal_train_data = "data/internal_data/restaurant_train_" + str(year) + ".json"
    internal_test_data = "data/internal_data/restaurant_test_" + str(year) + ".json"

    remaining_data = "data/internal_data/remaining_indices_ontology_" + str(year) + ".json"
    remaining_data_cross_val = "data/internal_data/remaining_indices_ontology_cross_val_" + str(year) + ".json"

    cross_validation_indices_training = "data/internal_data/cross_val_" + str(cross_validation_rounds) + \
                                        "_training_indices_" + str(year) + ".json"
    cross_validation_indices_validation = "data/internal_data/cross_val_" + str(cross_validation_rounds) + \
                                          "_validation_indices_" + str(year) + ".json"

    glove_embeddings = "data/external_data/glove.42B." + str(embedding_dimension) + "d.txt"


class OntologyConfig(Config):

    name_of_model = "ontology_reasoner"
    cross_validation_rounds = 10
    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + ".json"


class SVMConfig(Config):

    name_of_model = "svm_model"
    cross_validation_rounds = 10
    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"


class NeuralLanguageModelConfig(Config):

    name_of_model = "neural_language_model"

    @staticmethod
    def loss_function(y, prob):
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = - tf.reduce_mean(y * tf.log(prob + 1e-40)) + tf.reduce_sum(regularization_loss)
        return loss

    @staticmethod
    def accuracy_function(y, prob):
        correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
        acc_num = tf.reduce_sum(tf.cast(correct_pred, dtype=tf.int32))
        acc_prob = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
        return acc_num, acc_prob


class CabascConfig(NeuralLanguageModelConfig):

    name_of_model = "CABASC_model"
    batch_size = 20
    number_hidden_units = 300
    l2_regularization = 0.00001
    number_of_iterations = 50
    max_sentence_length = 10
    max_target_length = 10
    number_of_classes = 3
    learning_rate = 0.0001
    momentum = 0.95
    keep_prob1 = 0.5
    keep_prob2 = 0.5
    random_base = 0.01

    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"
    file_to_save_model = "data/model_savings/CABASC_model_" + str(Config.year) + "_tf.model"

    @staticmethod
    def split_embeddings(word_embeddings, aspect_indices, max_sentence_length, max_target_length):
        number_of_sentences = np.shape(word_embeddings)

        left_part = np.zeros((number_of_sentences[0], max_sentence_length, 300))
        right_part = np.zeros((number_of_sentences[0], max_sentence_length, 300))

        words_in_left_context = np.zeros(number_of_sentences[0], dtype=int)
        words_in_right_context = np.zeros(number_of_sentences[0], dtype=int)

        for index in range(number_of_sentences[0]):

            begin_index_aspect = aspect_indices[index][0]
            end_index_aspect = aspect_indices[index][-1]

            np_word_embeddings = np.array(word_embeddings[index])
            max_embeddings = np_word_embeddings.shape[0]

            words_in_left_context[index] = end_index_aspect + 1
            words_in_right_context[index] = max_embeddings - begin_index_aspect

            left_part[index][:words_in_left_context[index]] = np_word_embeddings[0:end_index_aspect + 1]
            right_part[index][:words_in_right_context[index]] = np_word_embeddings[begin_index_aspect:]

        return left_part, None, right_part, words_in_left_context, None, words_in_right_context


class LCR_RotConfig(NeuralLanguageModelConfig):

    name_of_model = "LCR_Rot_model"
    batch_size = 20
    number_hidden_units = 300
    l2_regularization = 0.001
    number_of_iterations = 10
    max_sentence_length = 80
    max_target_length = 19
    number_of_classes = 3
    learning_rate = 0.07
    momentum = 0.9
    keep_prob1 = 0.5
    keep_prob2 = 0.5
    random_base = 0.01

    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"
    file_to_save_model = "data/model_savings/LCR_Rot_model_" + str(Config.year) + "/tf.model"

    tr_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/training_" + name_of_model + ".json"
    te_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/test_" + name_of_model + ".json"

    @staticmethod
    def split_embeddings(word_embeddings, aspect_indices, max_sentence_length, max_target_length):

        number_of_sentences = np.shape(word_embeddings)

        left_part = np.zeros((number_of_sentences[0], max_sentence_length, 300), dtype=float)
        right_part = np.zeros((number_of_sentences[0], max_sentence_length, 300), dtype=float)
        target_part = np.zeros((number_of_sentences[0], max_target_length, 300), dtype=float)

        words_in_left_context = np.zeros(number_of_sentences[0], dtype=int)
        words_in_target = np.zeros(number_of_sentences[0], dtype=int)
        words_in_right_context = np.zeros(number_of_sentences[0], dtype=int)

        for index in range(number_of_sentences[0]):

            begin_index_aspect = aspect_indices[index][0]
            end_index_aspect = aspect_indices[index][-1]

            np_word_embeddings = np.array(word_embeddings[index])
            max_embeddings = np_word_embeddings.shape[0]

            words_in_left_context[index] = begin_index_aspect
            words_in_target[index] = (end_index_aspect - begin_index_aspect) + 1
            words_in_right_context[index] = max_embeddings - (end_index_aspect + 1)

            left_part[index][:words_in_left_context[index]] = np_word_embeddings[0:begin_index_aspect]
            target_part[index][:words_in_target[index]] = np_word_embeddings[begin_index_aspect:end_index_aspect + 1]
            right_part[index][:words_in_right_context[index]] = np_word_embeddings[end_index_aspect + 1:]

        return left_part, target_part, right_part, words_in_left_context, words_in_target, words_in_right_context


class LCR_RotInverseConfig(LCR_RotConfig):

    name_of_model = "LCR_Rot_inverse_model"
    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"
    file_to_save_model = "data/model_savings/LCR_Rot_inverse_model_" + str(Config.year) + "_tf.model"


class LCR_RotHopConfig(LCR_RotConfig):

    name_of_model = "LCR_Rot_hop_model"
    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"
    file_to_save_model = "data/model_savings/LCR_Rot_hop_model_" + str(Config.year) + "_tf.model"

class DiagnosticClassifierPOSConfig(Config):

    neural_language_model = NeuralLanguageModelConfig
    name_of_model = "diagnostic classifier for ontology mentions"
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=300,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        model_name=name_of_model
    )

    file_of_results = "results/diagnostic_classifiers/" + str(Config.year) + "/" + \
                                             neural_language_model.name_of_model + \
                                             "_part_of_speech_tagging.json"

class DiagnosticClassifierPolarityConfig(Config):

    name_of_model = "diagnostic classifier for ontology mentions"
    neural_language_model = NeuralLanguageModelConfig
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=500,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        model_name=name_of_model
    )

    file_of_results = "results/diagnostic_classifiers/" + str(Config.year) + "/" + neural_language_model.name_of_model \
                      + "_polarity_towards_aspect.json"


class DiagnosticClassifierRelationConfig(Config):

    name_of_model = "diagnostic classifier for ontology mentions"
    neural_language_model = NeuralLanguageModelConfig
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=300,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        model_name=name_of_model
    )

    file_of_results = "results/diagnostic_classifiers/" + str(Config.year) + "/" + \
                                              neural_language_model.name_of_model + \
                                              "_relation_towards_aspect.json"


class DiagnosticClassifierMentionConfig(Config):

    name_of_model = "diagnostic classifier for ontology mentions"
    neural_language_model = NeuralLanguageModelConfig
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=300,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        model_name=name_of_model
    )

    file_of_results = "results/diagnostic_classifiers/" + str(Config.year) + "/" + \
                                       neural_language_model.name_of_model + \
                                       "_ontology_mention.json"


class LocalInterpretableConfig(Config):

    name_of_model = "Local Interpretable model"

    # algorithm to calculate the neighbours of a instance
    locality_model_name = "perturbing"
    locality_model = Perturbing(10)

    # rule based classifier
    rule_based_classifier_name = "C4.5"
    rule_based_classifier = RuleBasedClassifier(0.90, 5)

    # classifier to compute word relevance, lasso regression or prediction difference
    alpha = 0.05
    attribute_evaluator_name = "lasso_regression_with_alpha_" + str(alpha)
    attribute_evaluator = LASSORegression(alpha)

    # attribute_evaluator_name = "prediction difference"
    # attribute_evaluator = PredictionDifference()

    file_of_results = "results/local_interpretable_models/" + str(Config.year) + "/" + locality_model_name + "_" + \
                      rule_based_classifier_name + ".json"