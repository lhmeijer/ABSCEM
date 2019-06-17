import tensorflow as tf


def dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name):

    outputs, state = tf.nn.dynamic_rnn(
        cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )  # outputs -> batch_size * max_len * n_hidden
    mask = tf.reverse(tf.cast(tf.sequence_mask(length, max_len), tf.float32), [1])
    mask_tiled = tf.tile(mask, [1, n_hidden])
    mask_3d = tf.reshape(mask_tiled, tf.shape(outputs))
    return tf.multiply(outputs, mask_3d)


def bi_dynamic_rnn(cell, inputs, n_hidden, length, scope_name):

    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    return outputs


def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keepdims=False) / length
    return inputs


def softmax_layer(inputs, n_hidden, random_base, keep_prob, l2_reg, n_class, scope_name='1'):
    w = tf.get_variable(
        name='softmax_w' + scope_name,
        shape=[n_hidden, n_class],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b' + scope_name,
        shape=[n_class],
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    with tf.name_scope('softmax'):
        rate = 1 - keep_prob
        outputs = tf.nn.dropout(inputs, rate=rate)
        predict = tf.matmul(outputs, w) + b
        predict = tf.nn.softmax(predict)
    return predict
