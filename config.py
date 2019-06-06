import tensorflow as tf


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

    @staticmethod
    def loss_function(y, prob):
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = - tf.reduce_mean(y * tf.log(prob)) + sum(regularization_loss)
        return loss

    @staticmethod
    def accuracy_function(y, prob):
        correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
        acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
        acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
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


class LCR_RotConfig(NeuralLanguageModelConfig):

    name_of_model = "LCR_Rot_model"
    batch_size = 20
    number_hidden_units = 300
    l2_regularization = 0.0001
    number_of_iterations = 50
    max_sentence_length = 80
    max_target_length = 20
    number_of_classes = 3
    learning_rate = 0.0001
    momentum = 0.95
    keep_prob1 = 0.5
    keep_prob2 = 0.5
    random_base = 0.01

    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"
    file_to_save_model = "data/model_savings/LCR_Rot_model_" + str(Config.year) + "_tf.model"


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


class DiagnosticClassifierConfig(Config):

    # targets to train the diagnostic classifier
    part_of_speech_tagging = True
    negation_tagging = True
    polarity_towards_aspect = True
    relation_towards_aspect = True
    ontology_mention = True


class LocalInterpretableConfig(Config):

    # classifier to compute word relevance
    LASSO_regression = True
    prediction_difference = False