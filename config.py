import tensorflow as tf


class Config:

    year = 2015
    embedding_dimension = 300
    hybrid = False
    cross_validation_rounds = 10
    cross_validation = False

    external_train_data = "data/external_data/restaurant_train_" + str(year) + ".xml"
    external_test_data = "data/external_data/restaurant_test_" + str(year) + ".xml"

    internal_train_data = "data/internal_data/restaurant_train_" + str(year) + ".json"
    internal_test_data = "data/internal_data/restaurant_test_" + str(year) + ".json"

    glove_embeddings = "data/external_data/glove.42B." + str(embedding_dimension) + "d.txt"



class OntologyConfig(Config):



class NeuralLanguageModelConfig(Config):

    @staticmethod
    def loss_function(y, prob):
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = - tf.reduce_mean(y * tf.log(prob)) + sum(regularization_loss)
        return loss

    @staticmethod
    def acc_func(y, prob):
        correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
        acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
        acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return acc_num, acc_prob


class CabascConfig(NeuralLanguageModelConfig):

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


class LCR_RotConfig(NeuralLanguageModelConfig):
    batch_size = 20
    number_hidden_units = 300
    l2_regularization = 0.00001
    number_of_iterations = 50

class LCR_RotInverseConfig(LCR_RotConfig):
    batch_size = 20
    number_hidden_units = 300
    l2_regularization = 0.00001
    number_of_iterations = 50

class LCR_RotHopConfig(LCR_RotConfig):
    batch_size = 20
    number_hidden_units = 300
    l2_regularization = 0.00001
    number_of_iterations = 50

class DiagnosticClassifierConfig(Config):

class LocalInterpretableConfig(Config):


    # tf.app.flags.DEFINE_integer("year", 2015, "year data set [2014]")
    # tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    # tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
    # tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
    # tf.app.flags.DEFINE_float('learning_rate', 0.07, 'learning rate')
    # tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    # tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
    # tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
    # tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
    # tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
    # tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
    # tf.app.flags.DEFINE_integer('n_iter', 50, 'number of train iter')
    # tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob')
    # tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob')
    # tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
    # tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
    # tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
    # tf.app.flags.DEFINE_string('is_r', '1', 'prob')
    # tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')