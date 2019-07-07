import tensorflow as tf


def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keepdims=True) + 1e-9
    return inputs / _sum


def attention_function(inputs, attend, length, n_hidden, max_len, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param max_len:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """

    batch_size = tf.shape(inputs)[0]
    max_length = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[max_len],
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_length, n_hidden])
    attend = tf.expand_dims(attend, 2)
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_length])
    tmp = tf.tanh(tmp + b)
    return softmax_with_len(tmp, length, max_length)
