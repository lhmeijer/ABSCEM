import tensorflow as tf


class NeuralLanguageModel:

    def __int__(self, config):
        self.config = config

    def model_itself(self, left_sentence_part, right_sentence_part, target_part):
        pass

    def run(self):

        left_part = tf.placeholder(tf.int32, [None, self.config.max_sentence_length])
        sen_len = tf.placeholder(tf.int32, None)

        right_part = tf.placeholder(tf.int32, [None, self.config.max_sentence_length])
        sen_len_bw = tf.placeholder(tf.int32, [None])

        target_part = tf.placeholder(tf.int32, [None, self.config.max_target_length])
        tar_len = tf.placeholder(tf.int32, [None])

        y = tf.placeholder(tf.float32, [None, self.config.number_of_classes])

        prob = self.model_itself(left_sentence_part=left_part, right_sentence_part=right_part, target_part=target_part)
        loss = self.config.loss_function(y, prob)
        acc = self.config.accuracy_function(y, prob)

        optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=self.config.momentum)
        trainer = optimizer.minimize(loss)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
            #     for index in batch_index(len(yi), batch_size, 1, is_shuffle):
            #         feed_dict = {
            #             x: x_f[index],
            #             x_bw: x_b[index],
            #             y: yi[index],
            #             sen_len: sen_len_f[index],
            #             sen_len_bw: sen_len_b[index],
            #             target_words: target[index],
            #             tar_len: tl[index],
            #             keep_prob1: kp1,
            #             keep_prob2: kp2,
            #         }
            #         yield feed_dict, len(index)

            for i in range(self.config.number_of_iterations):
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word,
                                               tr_tar_len,
                                               FLAGS.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                    _ = sess.run([trainer], feed_dict=train)

            acc, cost, cnt = 0., 0., 0
            fw, bw, tl, tr, ty, py = [], [], [], [], [], []
            p = []
            for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                            te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                    _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                        [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob], feed_dict=test)
                    fw += list(_fw)
                    bw += list(_bw)
                    tl += list(_tl)
                    tr += list(_tr)
                else:
                    _loss, _acc, _ty, _py, _p = sess.run([loss, acc_num, true_y, pred_y, prob], feed_dict=test)
                ty += list(_ty)
                py += list(_py)
                p += list(_p)
                acc += _acc
                cost += _loss * num
                cnt += num
            print
            'all samples={}, correct prediction={}'.format(cnt, acc)
            acc = acc / cnt
            cost = cost / cnt




