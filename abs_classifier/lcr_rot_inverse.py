from abs_classifier.neural_language_model import NeuralLanguageModel
import tensorflow as tf


class LCRRotInverse(NeuralLanguageModel):

    def __init__(self, config):
        self.config = config

    def model_itself(self, left_sentence_part, right_sentence_part, target_part):
        print('I am lcr rot inverse.')

        cell = tf.contrib.rnn.LSTMCell

        # left hidden
        input_left = tf.nn.dropout(left_sentence_part, keep_prob=self.config.keep_prob1)
        left_hidden_state = bi_dynamic_rnn(cell, input_left, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
        pool_l = reduce_mean_with_len(left_hidden_state, sen_len_fw)

        # right hidden
        input_right = tf.nn.dropout(right_sentence_part, keep_prob=self.config.keep_prob1)
        right_hidden_state = bi_dynamic_rnn(cell, input_right, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
        pool_r = reduce_mean_with_len(right_hidden_state, sen_len_bw)

        # target hidden
        target = tf.nn.dropout(target_part, keep_prob=self.config.keep_prob1)
        target_hidden_state = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')


        # attention target
        att_t_l = bilinear_attention_layer(target_hidden_state, pool_l, sen_len_tr,
                                           2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tl')

        target_left_context_representation = tf.squeeze(tf.matmul(att_t_l, target_hidden_state))

        att_t_r = bilinear_attention_layer(target_hidden_state, pool_r, sen_len_tr,
                                           2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tr')

        target_right_context_representation = tf.squeeze(tf.matmul(att_t_r, target_hidden_state))

        # attention left
        att_l = bilinear_attention_layer(left_hidden_state, target_left_context_representation, sen_len_fw, 2 * FLAGS.n_hidden, FLAGS.l2_reg,
                                         FLAGS.random_base, 'l')

        left_context_representation = tf.squeeze(tf.matmul(att_l, left_hidden_state))

        # attention right
        att_r = bilinear_attention_layer(right_hidden_state, target_right_context_representation, sen_len_fw, 2 * FLAGS.n_hidden, FLAGS.l2_reg,
                                         FLAGS.random_base, 'l')

        right_context_representation = tf.squeeze(tf.matmul(att_r, right_hidden_state))


        sentence_representation = tf.concat([left_context_representation, target_left_context_representation, target_right_context_representation, right_context_representation], 1)

        prob = softmax_layer(sentence_representation, 8 * FLAGS.n_hidden, FLAGS.random_base, self.config.keep_prob2,
                             FLAGS.l2_reg, FLAGS.n_class)

        return prob, att_l, att_r, att_t_l, att_t_r