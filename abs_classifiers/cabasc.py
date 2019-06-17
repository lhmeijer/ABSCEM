from abs_classifiers.neural_language_model import NeuralLanguageModel
import tensorflow as tf
from model_layers.nn_layers import dynamic_rnn, softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from model_layers.attention_layers import Mlp_attention_layer, cam_mlp_attention_layer, mlp_layer, triple_attention_layer



class CABASCModel(NeuralLanguageModel):

    def __init__(self, config, internal_data_loader):
        self.config = config
        self.internal_data_loader = internal_data_loader

    def model_itself(self, left_sentence_parts, left_sentence_lengths, right_sentence_parts, right_sentence_lengths,
                     target_parts, target_lengths):

        print('I am cabasc.')

        cell = tf.keras.layers.GRUCell

        rate = 1 - self.config.keep_prob1

        # left GRU
        input_left = tf.nn.dropout(left_sentence_parts, rate=rate)
        left_hidden_state = dynamic_rnn(cell, input_left, self.config.number_hidden_units, left_sentence_lengths)  # batch_size x N x d

        # right GRU
        input_right = tf.nn.dropout(right_sentence_parts, rate=rate)
        right_hidden_state = dynamic_rnn(cell, input_right,self.config.number_hidden_units, right_sentence_lengths)  # batch_size x N x d

        # MLP layer for attention weight
        # left MLP
        beta_left = cam_mlp_attention_layer(left_hidden_state, left_sentence_lengths, self.config.number_hidden_units,
                                            self.config.l2_regularization, self.config.random_base, 'bl')  # batch_size x 1 x N
        beta_left = tf.squeeze(tf.nn.dropout(beta_left, rate=rate))  # batch_size x N
        beta_left = tf.reverse(beta_left, [1])

        # right MLP
        beta_right = cam_mlp_attention_layer(right_hidden_state, right_sentence_lengths,
                                             self.config.number_hidden_units, self.config.l2_regularization,
                                             self.config.random_base, 'br')  # batch_size x 1 x N
        beta_right = tf.squeeze(tf.nn.dropout(beta_right, rate=rate))  # batch_size x N

        beta_add = tf.add(beta_left, beta_right)  # batch_size x N
        beta = tf.multiply(beta_add, mult_mask)  # batch_size x N
        beta = tf.expand_dims(beta, 2)  # batch_size x N x 1
        beta_tiled = tf.tile(beta, [1, 1, self.config.embedding_dimension])
        beta_3d = tf.reshape(beta_tiled, [-1, self.config.max_sentence_length, self.config.embedding_dimension])  # batch_size x N x d

        # Memory Embedding
        M = tf.multiply(sent_full, beta_3d)  # batch_size x N x d
        # M = sent_full # Model B

        # Target Embedding
        target_tiled = tf.tile(target, [1, self.config.max_sentence_length])
        target_3d = tf.reshape(target_tiled, [-1, self.config.max_sentence_length, FLAGS.embedding_dim])  # batch_size x N x d

        # Average Target
        ave_sent_tiled = tf.tile(tf.expand_dims(ave_sent, 1), [1, self.config.max_sentence_length, 1])
        ave_sent_3d = tf.reshape(ave_sent_tiled,
                                 [-1, self.config.max_sentence_length, FLAGS.embedding_dim])  # batch_size x N x d

        att_weights = triple_attention_layer(M, target_3d, ave_sent_3d, sen_len_fw + sen_len_bw - sen_len_tr,
                                             FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base,
                                             'eq7')  # batch_size x 1 x N

        v_ts = tf.matmul(att_weights, M)  # batch_size x 1 x d
        v_ns = tf.add(v_ts, tf.expand_dims(ave_sent, 1))  # batch_size x 1 x d
        v_ns = tf.nn.dropout(v_ns, keep_prob1)

        v_ms = mlp_layer(v_ns, FLAGS.embedding_dim, FLAGS.l2_reg, FLAGS.random_base, 'eq5')  # batch_size x 1 x d

        prob = softmax_layer(tf.squeeze(v_ms), FLAGS.embedding_dim, FLAGS.random_base, keep_prob1, FLAGS.l2_reg,
                             FLAGS.n_class)

        return prob, att_weights, beta_left, beta_right, v_ms