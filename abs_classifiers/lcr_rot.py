from abs_classifiers.neural_language_model import NeuralLanguageModel
import tensorflow as tf
from neural_network_layers.attention_layers import attention_function
from neural_network_layers.nn_layers import bi_dynamic_rnn, softmax_layer, reduce_mean_with_len


class LCRRot(NeuralLanguageModel):

    def __init__(self, config, internal_data_loader):
        super().__init__(config, internal_data_loader)

    def model_itself(self, left_sentence_parts, left_sentence_lengths, right_sentence_parts, right_sentence_lengths,
                     target_parts, target_lengths, keep_prob1, keep_prob2):

        print('I am lcr rot.')

        _id = '_lcr_rot'

        cell = tf.contrib.rnn.LSTMCell

        rate = 1 - keep_prob1

        # left hidden states
        input_left = tf.nn.dropout(left_sentence_parts, rate=rate)
        left_hidden_state = bi_dynamic_rnn(cell, input_left, self.config.number_hidden_units, left_sentence_lengths, 'l' + _id)

        # right hidden states
        input_right = tf.nn.dropout(right_sentence_parts, rate=rate)
        right_hidden_state = bi_dynamic_rnn(cell, input_right, self.config.number_hidden_units, right_sentence_lengths, 'r' + _id)

        # target hidden states
        target = tf.nn.dropout(target_parts, rate=rate)
        target_hidden_state = bi_dynamic_rnn(cell, target, self.config.number_hidden_units, target_lengths, 't' + _id)

        # pooling target hidden layer
        pool_t = reduce_mean_with_len(target_hidden_state, target_lengths)

        # attention left
        att_l = attention_function(left_hidden_state, pool_t, left_sentence_lengths, 2 * self.config.number_hidden_units,
                                   self.config.l2_regularization, self.config.random_base, 'att_l' + _id)
        print("att_l ", att_l)
        print("right_hidden_state ", right_hidden_state)
        print("tf.matmul(att_l, left_hidden_state) ", tf.matmul(att_l, left_hidden_state))
        print("element-wise ", tf.math.multiply(att_l, left_hidden_state))
        weighted_left_hidden_state = tf.math.multiply(tf.transpose(att_l, perm=[0, 2, 1]), left_hidden_state)
        print("weighted_left_hidden_state ", weighted_left_hidden_state)

        left_context_representation = tf.squeeze(tf.matmul(att_l, left_hidden_state), [1])

        # attention right
        att_r = attention_function(right_hidden_state, pool_t, right_sentence_lengths, 2 * self.config.number_hidden_units,
                                   self.config.l2_regularization, self.config.random_base, 'att_r' + _id)

        print("tf.matmul(att_r, right_hidden_state) ", tf.matmul(att_r, right_hidden_state))
        weighted_right_hidden_state = tf.math.multiply(tf.transpose(att_r, perm=[0, 2, 1]), right_hidden_state)
        right_context_representation = tf.squeeze(tf.matmul(att_r, right_hidden_state), [1])

        # attention target
        att_t_l = attention_function(target_hidden_state, left_context_representation, target_lengths,
                                     2 * self.config.number_hidden_units, self.config.l2_regularization,
                                     self.config.random_base, 'att_t_l' + _id)
        print("att_t_l ", att_t_l)
        print("target_hidden_state ", target_hidden_state)
        weighted_target_left_hidden_state = tf.math.multiply(tf.transpose(att_t_l, perm=[0, 2, 1]), target_hidden_state)

        target_left_context_representation = tf.squeeze(tf.matmul(att_t_l, target_hidden_state), [1])

        att_t_r = attention_function(target_hidden_state, right_context_representation, target_lengths,
                                     2 * self.config.number_hidden_units, self.config.l2_regularization,
                                     self.config.random_base, 'att_t_r' + _id)

        weighted_target_right_hidden_state = tf.math.multiply(tf.transpose(att_t_r, perm=[0, 2, 1]),
                                                              target_hidden_state)

        target_right_context_representation = tf.squeeze(tf.matmul(att_t_r, target_hidden_state), [1])
        print('target_right_context_representation ', target_right_context_representation)

        sentence_representation = tf.concat([left_context_representation, target_left_context_representation,
                                             target_right_context_representation, right_context_representation], 1)

        prob = softmax_layer(sentence_representation, 8 * self.config.number_hidden_units, self.config.random_base,
                             keep_prob2, self.config.l2_regularization, self.config.number_of_classes)

        layer_information = {
            'left_hidden_state': left_hidden_state,
            'right_hidden_state': right_hidden_state,
            'target_hidden_state': target_hidden_state,
            'weighted_left_hidden_state': weighted_left_hidden_state,
            'weighted_right_hidden_state': weighted_right_hidden_state,
            'weighted_target_left_hidden_state': weighted_target_left_hidden_state,
            'weighted_target_right_hidden_state': weighted_target_right_hidden_state
        }

        return prob, layer_information
