from abs_classifiers.neural_language_model import NeuralLanguageModel
import tensorflow as tf
from model_layers.attention_layers import attention_function
from model_layers.nn_layers import bi_dynamic_rnn, softmax_layer, reduce_mean_with_len


class LCRRotHopModel(NeuralLanguageModel):

    def __init__(self, config, internal_data_loader):
        super().__init__(config, internal_data_loader)

    def model_itself(self, left_sentence_parts, left_sentence_lengths, right_sentence_parts, right_sentence_lengths,
                     target_parts, target_lengths, keep_prob1, keep_prob2):

        print('I am lcr rot hop.')

        _id = '_lcr_rot_hop'

        cell = tf.contrib.rnn.LSTMCell

        rate = 1 - keep_prob1

        # left hidden states
        input_left = tf.nn.dropout(left_sentence_parts, rate=rate)
        left_hidden_state = bi_dynamic_rnn(cell, input_left, self.config.number_hidden_units, left_sentence_lengths,
                                           'l' + _id)
        pool_l = reduce_mean_with_len(left_hidden_state, left_sentence_lengths)

        # right hidden states
        input_right = tf.nn.dropout(right_sentence_parts, rate=rate)
        right_hidden_state = bi_dynamic_rnn(cell, input_right, self.config.number_hidden_units, right_sentence_lengths,
                                            'r' + _id)
        pool_r = reduce_mean_with_len(right_hidden_state, right_sentence_lengths)

        # target hidden states
        target = tf.nn.dropout(target_parts, rate=rate)
        target_hidden_state = bi_dynamic_rnn(cell, target, self.config.number_hidden_units, target_lengths, 't' + _id)

        # Pooling target hidden layer
        pool_t = reduce_mean_with_len(target_hidden_state, target_lengths)

        target_left_context_representation = pool_t
        target_right_context_representation = pool_t
        left_context_representation = pool_l
        right_context_representation = pool_r

        layer_information = {
            'left_hidden_state': left_hidden_state,
            'right_hidden_state': right_hidden_state,
            'target_hidden_state': target_hidden_state
        }

        for i in range(self.config.n_iterations_hop):

            # attention left
            att_l = attention_function(left_hidden_state, target_left_context_representation, left_sentence_lengths,
                                       2 * self.config.number_hidden_units, self.config.max_sentence_length,
                                       self.config.l2_regularization, self.config.random_base, 'att_l' + _id + str(i))
            layer_information['left_attention_score_' + str(i)] = att_l
            weighted_left_hidden_state = tf.math.multiply(tf.transpose(att_l, perm=[0, 2, 1]), left_hidden_state)
            layer_information['weighted_left_hidden_state_' + str(i)] = weighted_left_hidden_state
            left_context_representation = tf.squeeze(tf.matmul(att_l, left_hidden_state), [1])

            # attention right
            att_r = attention_function(right_hidden_state, target_right_context_representation, right_sentence_lengths,
                                       2 * self.config.number_hidden_units, self.config.max_sentence_length,
                                       self.config.l2_regularization, self.config.random_base, 'att_r' + _id + str(i))
            layer_information['right_attention_score_' + str(i)] = att_r
            weighted_right_hidden_state = tf.math.multiply(tf.transpose(att_r, perm=[0, 2, 1]), right_hidden_state)
            layer_information['weighted_right_hidden_state_'+str(i)] = weighted_right_hidden_state
            right_context_representation = tf.squeeze(tf.matmul(att_r, right_hidden_state), [1])

            # attention target left
            att_t_l = attention_function(target_hidden_state, left_context_representation, target_lengths,
                                         2 * self.config.number_hidden_units, self.config.max_target_length,
                                         self.config.l2_regularization, self.config.random_base,
                                         'att_t_l' + _id + str(i))
            layer_information['target_left_attention_score_' + str(i)] = att_t_l
            weighted_target_left_hidden_state = tf.math.multiply(tf.transpose(att_t_l, perm=[0, 2, 1]),
                                                                 target_hidden_state)
            layer_information['weighted_target_left_hidden_state_' + str(i)] = weighted_target_left_hidden_state
            target_left_context_representation = tf.squeeze(tf.matmul(att_t_l, target_hidden_state), [1])

            # attention target right
            att_t_r = attention_function(target_hidden_state, right_context_representation, target_lengths,
                                         2 * self.config.number_hidden_units, self.config.max_target_length,
                                         self.config.l2_regularization, self.config.random_base,
                                         'att_t_r' + _id + str(i))
            layer_information['target_right_attention_score_' + str(i)] = att_t_r
            weighted_target_right_hidden_state = tf.math.multiply(tf.transpose(att_t_r, perm=[0, 2, 1]),
                                                                  target_hidden_state)
            layer_information['weighted_target_right_hidden_state_' + str(i)] = weighted_target_right_hidden_state
            target_right_context_representation = tf.squeeze(tf.matmul(att_t_r, target_hidden_state), [1])

        sentence_representation = tf.concat([left_context_representation, target_left_context_representation,
                                             target_right_context_representation, right_context_representation], 1)

        prob = softmax_layer(sentence_representation, 8 * self.config.number_hidden_units, self.config.random_base,
                             self.config.keep_prob2, self.config.l2_regularization, self.config.number_of_classes)

        return prob, layer_information
