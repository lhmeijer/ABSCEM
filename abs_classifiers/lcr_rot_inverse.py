from abs_classifiers.neural_language_model import NeuralLanguageModel
import tensorflow as tf
from neural_network_layers.attention_layers import attention_function
from neural_network_layers.nn_layers import bi_dynamic_rnn, softmax_layer, reduce_mean_with_len


class LCRRotInverse(NeuralLanguageModel):

    def __init__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    def model_itself(self, left_sentence_parts, left_sentence_lengths, right_sentence_parts, right_sentence_lengths,
                     target_parts, target_lengths):

        print('I am lcr rot inverse.')

        _id = '_lcr_rot_inverse'

        cell = tf.keras.layers.LSTM

        rate = 1 - self.config.keep_prob1

        # left hidden states
        input_left = tf.nn.dropout(left_sentence_parts, rate=rate)
        left_hidden_state = bi_dynamic_rnn(cell, input_left, self.config.number_hidden_units, left_sentence_lengths)
        pool_l = reduce_mean_with_len(left_hidden_state, left_sentence_lengths)

        # right hidden states
        input_right = tf.nn.dropout(right_sentence_parts, rate=rate)
        right_hidden_state = bi_dynamic_rnn(cell, input_right, self.config.number_hidden_units, right_sentence_lengths)
        pool_r = reduce_mean_with_len(right_hidden_state, right_sentence_parts)

        # target hidden states
        target = tf.nn.dropout(target_parts, rate=rate)
        target_hidden_state = bi_dynamic_rnn(cell, target, self.config.number_hidden_units, target_lengths)

        # attention target left
        att_t_l = attention_function(target_hidden_state, pool_l, target_lengths, self.config.max_target_length,
                                     self.config.batch_size, 2 * self.config.number_hidden_units,
                                     self.config.l2_regularization, self.config.random_base, 'att_t_l' + _id)

        target_left_context_representation = tf.squeeze(tf.matmul(tf.transpose(att_t_l, perm=[0, 2, 1]),
                                                                  target_hidden_state))
        # attention target right
        att_t_r = attention_function(target_hidden_state, pool_r, target_lengths, self.config.max_target_length,
                                     self.config.batch_size, 2 * self.config.number_hidden_units,
                                     self.config.l2_regularization, self.config.random_base, 'att_t_r' + _id)

        target_right_context_representation = tf.squeeze(tf.matmul(tf.transpose(att_t_r, perm=[0, 2, 1]),
                                                                   target_hidden_state))

        # attention left
        att_l = attention_function(left_hidden_state, target_left_context_representation, left_sentence_lengths,
                                   self.config.max_sentence_length, self.config.batch_size,
                                   2 * self.config.number_hidden_units, self.config.l2_regularization,
                                   self.config.random_base, 'att_l' + _id)

        left_context_representation = tf.squeeze(tf.matmul(tf.transpose(att_l, perm=[0, 2, 1]), left_hidden_state))

        # attention right
        att_r = attention_function(right_hidden_state, target_right_context_representation, right_sentence_lengths,
                                   self.config.max_sentence_length, self.config.batch_size,
                                   2 * self.config.number_hidden_units, self.config.l2_regularization,
                                   self.config.random_base, 'att_r' + _id)

        right_context_representation = tf.squeeze(tf.matmul(tf.transpose(att_r, perm=[0, 2, 1]), right_hidden_state))

        sentence_representation = tf.concat([left_context_representation, target_left_context_representation,
                                             target_right_context_representation, right_context_representation], 1)

        prob = softmax_layer(sentence_representation, 8 * FLAGS.n_hidden, FLAGS.random_base, self.config.keep_prob2,
                             FLAGS.l2_reg, FLAGS.n_class)

        layer_information = {
            'left_hidden_state': left_hidden_state,
            'right_hidden_state': right_hidden_state,
            'target_hidden_state': target_hidden_state,
            'left_context_representation': left_context_representation,
            'right_context_representation': right_context_representation,
            'target_left_context_representation': target_left_context_representation,
            'target_right_context_representation': target_right_context_representation
        }

        return prob, layer_information, att_l, att_r, att_t_l, att_t_r