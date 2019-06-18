import tensorflow as tf
from diagnostic_classifier.classifiers import SingleMLPClassifier
from local_interpretable_model.locality_algorithms import Perturbing
from local_interpretable_model.rule_based_classifier import RuleBasedClassifier


class Config:

    year = 2016
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


class LCR_RotConfig(NeuralLanguageModelConfig):

    name_of_model = "LCR_Rot_model"
    batch_size = 20
    number_hidden_units = 300
    l2_regularization = 0.00001
    number_of_iterations = 50
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
    file_to_save_model = "abs_classifiers/model_savings/" + str(Config.year) + "/lcr_rot_model/tf.model"

    tr_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/training_" + name_of_model + ".json"
    te_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/test_" + name_of_model + ".json"


class LCR_RotInverseConfig(LCR_RotConfig):

    name_of_model = "LCR_Rot_inverse_model"

    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"
    file_to_save_model = "abs_classifiers/model_savings/" + str(Config.year) + "/lcr_rot_inverse_model/tf.model"
    tr_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/training_" + name_of_model + ".json"
    te_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/test_" + name_of_model + ".json"


class LCR_RotHopConfig(LCR_RotConfig):

    name_of_model = "LCR_Rot_hop_model"

    n_iterations_hop = 3

    file_of_results = "results/abs_classifiers/" + str(Config.year) + "/" + name_of_model + ".json"
    file_of_cross_val_results = "results/abs_classifiers/" + str(Config.year) + "/cross_val_" + name_of_model + "json"
    file_to_save_model = "abs_classifiers/model_savings/" + str(Config.year) + "/lcr_rot_hop_model/tf.model"
    tr_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/training_" + name_of_model + ".json"
    te_file_of_hidden_layers = "data/hidden_layers/" + str(Config.year) + "/test_" + name_of_model + ".json"


class DiagnosticClassifierPOSConfig(Config):

    name_of_model = "diagnostic classifier for part of speech tagging"
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=300,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        number_of_classes=5,
        dimension=300
    )

    @staticmethod
    def get_file_of_model_savings(name_of_nlm, name_of_hidden_state):
        return "diagnostic_classifier/model_savings/" + str(Config.year) + "/pos_tags/" + name_of_nlm + "_" + \
               name_of_hidden_state + "tf.model"

    @staticmethod
    def get_file_of_results(name_of_nlm):
        return "results/diagnostic_classifiers/" + str(Config.year) + "/" + name_of_nlm + \
               "_part_of_speech_tagging.json"


class DiagnosticClassifierPolarityConfig(Config):

    name_of_model = "diagnostic classifier for polarities towards the aspects"
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=500,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        number_of_classes=3,
        dimension=300
    )

    @staticmethod
    def get_file_of_model_savings(name_of_nlm, name_of_hidden_state):
        return "diagnostic_classifier/model_savings/" + str(Config.year) + "/polarities/" + name_of_nlm + "_" + \
               name_of_hidden_state + "tf.model"

    @staticmethod
    def get_file_of_results(name_of_nlm):
        return "results/diagnostic_classifiers/" + str(Config.year) + "/" + name_of_nlm + \
               "_polarity_towards_aspect.json"


class DiagnosticClassifierRelationConfig(Config):

    name_of_model = "diagnostic classifier for relations towards the aspects"
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=300,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        number_of_classes=2,
        dimension=300
    )

    @staticmethod
    def get_file_of_model_savings(name_of_nlm, name_of_hidden_state):
        return "diagnostic_classifier/model_savings/" + str(Config.year) + "/relations/" + name_of_nlm + "_" + \
               name_of_hidden_state + "tf.model"

    @staticmethod
    def get_file_of_results(name_of_nlm):
        return "results/diagnostic_classifiers/" + str(Config.year) + "/" + name_of_nlm + \
               "_relation_towards_aspect.json"


class DiagnosticClassifierMentionConfig(Config):

    name_of_model = "diagnostic classifier for ontology mentions"
    classifier = SingleMLPClassifier(
        learning_rate=0.001,
        number_hidden_units=300,
        number_of_epochs=100,
        keep_prob=0.8,
        batch_size=20,
        random_base=0.1,
        number_of_classes=14,
        dimension=300
    )

    @staticmethod
    def get_file_of_model_savings(name_of_nlm, name_of_hidden_state):
        return "diagnostic_classifier/model_savings/" + str(Config.year) + "/mentions/" + name_of_nlm + "_" + \
               name_of_hidden_state + "tf.model"

    @staticmethod
    def get_file_of_results(name_of_nlm):
        return "results/diagnostic_classifiers/" + str(Config.year) + "/" + name_of_nlm + \
                "_ontology_mention.json"


class LocalInterpretableConfig(Config):

    name_of_model = "Local Interpretable model"
    name_of_nlm = ""

    # algorithm to calculate the neighbours of an instance
    locality_model_name = "perturbing"
    locality_model = Perturbing(512)

    # rule based classifier, 3 is the number of subsets
    max_tree_depth = 4
    rule_based_classifier_name = "decision_tree"
    rule_based_classifier = RuleBasedClassifier(max_tree_depth)

    # classifier to compute word relevance, linear regression and prediction difference
    attribute_evaluator_name = "linear regression and prediction difference"
    n_of_subset = 3

    @staticmethod
    def get_file_of_results(name_of_nlm, locality_model_name=locality_model_name,
                            rule_based_classifier_name=rule_based_classifier_name):
        return "results/local_interpretable_models/" + str(Config.year) + "/" + name_of_nlm + "_" + \
                      locality_model_name + "_" + rule_based_classifier_name + ".json"
