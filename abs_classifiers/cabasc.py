from abs_classifiers.neural_language_model import NeuralLanguageModel
import tensorflow as tf
from neural_network_layers.nn_layers import dynamic_rnn, softmax_layer
from neural_network_layers.attention_layers import cam_mlp_attention_layer, triple_attention_layer, mlp_layer


class CABASCModel(NeuralLanguageModel):

    def __init__(self, config):
        self.config = config

    def model_itself(self, left_sentence_part, right_sentence_part, target_part):

        print('I am cabasc.')

        # CAM Module
        cell = tf.keras.layers.GRUCell
        # left GRU
        input_fw = tf.nn.dropout(left_sentence_part, self.config.keep_prob1)
        hiddens_l = dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'leftGRU',
                                'all')  # batch_size x N x d

        # right GRU
        input_bw = tf.nn.dropout(input_bw, keep_prob1)
        hiddens_r = dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'rightGRU',
                                'all')  # batch_size x N x d

        # MLP layer for attention weight
        # left MLP
        beta_left = cam_mlp_attention_layer(hiddens_l, sen_len_fw, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base,
                                            'bl')  # batch_size x 1 x N
        beta_left = tf.squeeze(tf.nn.dropout(beta_left, keep_prob1))  # batch_size x N
        beta_left = tf.reverse(beta_left, [1])

        # right MLP
        beta_right = cam_mlp_attention_layer(hiddens_r, sen_len_bw, FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base,
                                             'br')  # batch_size x 1 x N
        beta_right = tf.squeeze(tf.nn.dropout(beta_right, keep_prob1))  # batch_size x N

        beta_add = tf.add(beta_left, beta_right)  # batch_size x N
        beta = tf.multiply(beta_add, mult_mask)  # batch_size x N
        beta = tf.expand_dims(beta, 2)  # batch_size x N x 1
        beta_tiled = tf.tile(beta, [1, 1, FLAGS.embedding_dim])
        beta_3d = tf.reshape(beta_tiled, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])  # batch_size x N x d

        # Memory Embedding
        M = tf.multiply(sent_full, beta_3d)  # batch_size x N x d
        # M = sent_full # Model B

        # Target Embedding
        target_tiled = tf.tile(target, [1, FLAGS.max_sentence_len])
        target_3d = tf.reshape(target_tiled, [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])  # batch_size x N x d

        # Average Target
        ave_sent_tiled = tf.tile(tf.expand_dims(ave_sent, 1), [1, FLAGS.max_sentence_len, 1])
        ave_sent_3d = tf.reshape(ave_sent_tiled,
                                 [-1, FLAGS.max_sentence_len, FLAGS.embedding_dim])  # batch_size x N x d

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