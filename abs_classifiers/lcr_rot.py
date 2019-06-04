from abs_classifiers.neural_language_model import NeuralLanguageModel
import tensorflow as tf
from neural_network_layers.attention_layers import attention_function
from neural_network_layers.nn_layers import bi_dynamic_rnn, softmax_layer

class LCRRot(NeuralLanguageModel):

    def __init__(self, config):
        self.config = config

    def model_itself(self, left_sentence_part, right_sentence_part, target_part):

        print('I am lcr rot.')
        _id = '_lcr_rot'

        cell = tf.keras.layers.LSTM

        # left hidden
        input_left = tf.nn.dropout(left_sentence_part, keep_prob=self.config.keep_prob1)
        left_hidden_state = bi_dynamic_rnn(cell, input_left, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')

        # right hidden
        input_right = tf.nn.dropout(right_sentence_part, keep_prob=self.config.keep_prob1)
        right_hidden_state = bi_dynamic_rnn(cell, input_right, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')


        # target hidden
        target = tf.nn.dropout(target_part, keep_prob=self.config.keep_prob1)
        target_hidden_state = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')

        # Pooling target hidden layer
        pool_t = reduce_mean_with_len(target_hidden_state, sen_len_tr)

        # attention left
        att_l = attention_function(left_hidden_state, pool_t, sen_len_fw, self.config.l2_regularization, FLAGS.random_base, 'att_l' + _id)

        left_context_representation = tf.squeeze(tf.matmul(att_l, left_hidden_state))

        # attention right
        att_r = attention_function(right_hidden_state, pool_t, sen_len_fw, self.config.l2_regularization, FLAGS.random_base, 'att_r' + _id)

        right_context_representation = tf.squeeze(tf.matmul(att_r, right_hidden_state))

        # attention target
        att_t_l = attention_function(target_hidden_state, left_context_representation, sen_len_tr, self.config.l2_regularization, FLAGS.random_base, 'att_t_l' + _id)

        target_left_context_representation = tf.squeeze(tf.matmul(att_t_l, target_hidden_state))

        att_t_r = attention_function(target_hidden_state, right_context_representation, sen_len_tr, self.config.l2_regularization, FLAGS.random_base, 'att_t_r' + _id)

        target_right_context_representation = tf.squeeze(tf.matmul(att_t_r, target_hidden_state))

        sentence_representation = tf.concat([left_context_representation, target_left_context_representation, target_right_context_representation, right_context_representation], 1)

        prob = softmax_layer(sentence_representation, 8 * FLAGS.n_hidden, FLAGS.random_base, self.config.keep_prob2, FLAGS.l2_reg, FLAGS.n_class)

        return prob, att_l, att_r, att_t_l, att_t_r