from text_cnn import TextCNN
import tensorflow as tf
import os
import time
import datetime
import numpy as np


class TrainCNN(object):

    def __init__(self, whether_word2vec, whether_tf, vocab_size, embedding_dim, sequence_length, vocab_processor, num_classes, tf_dict_size,
                 tf_emb_size=10, filter_sizes=[1, 2, 3], num_filters=50):
        """

        :param whether_word2vec:
        :param whether_tf:
        :param vocab_size: 词典的长度
        :param embedding_dim:
        :param sequence_length: samples的最大长度
        :param vocab_processor: 词典,需要保存到本地.在use model的时候需要这个词典
        :param num_classes:
        :param tf_dict_size:
        :param tf_emb_size:
        :param filter_sizes:
        :param num_filters:
        :return:
        """
        g = tf.Graph()
        with g.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.cnn = TextCNN(
                    whether_word2vec=whether_word2vec,
                    whether_tf=whether_tf,
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    sequence_length=sequence_length,
                    num_classes=num_classes,
                    tf_dict_size=tf_dict_size,
                    tf_emb_size=tf_emb_size,
                    filter_sizes=filter_sizes,
                    num_filters=num_filters,
                    l2_reg_lambda=0.0
                )

                self.optimizer = tf.train.AdamOptimizer(1e-3)
                self.grads_and_vars = self.optimizer.compute_gradients(self.cnn.loss)
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

                # Keep track of gradient values and sparsity (optional)
                self.grad_summaries = []
                for g, v in self.grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        self.grad_summaries.append(grad_hist_summary)
                        self.grad_summaries.append(sparsity_summary)
                self.grad_summaries_merged = tf.merge_summary(self.grad_summaries)

                # Output directory for models and summaries
                self.timestamp = str(int(time.time()))
                self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", self.timestamp))
                # print("Writing to {}\n".format(self.out_dir))

                # Summaries for loss and accuracy
                self.loss_summary = tf.scalar_summary("loss", self.cnn.loss)
                self.acc_summary = tf.scalar_summary("accuracy", self.cnn.accuracy)

                # Train Summaries
                self.train_summary_op = tf.merge_summary([self.loss_summary, self.acc_summary])
                self.train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
                self.train_summary_writer = tf.train.SummaryWriter(self.train_summary_dir, self.sess.graph)

                # Dev summaries
                self.dev_summary_op = tf.merge_summary([self.loss_summary, self.acc_summary])
                self.dev_summary_dir = os.path.join(self.out_dir, "summaries", "dev")
                self.dev_summary_writer = tf.train.SummaryWriter(self.dev_summary_dir, self.sess.graph)

                # checkpoint directory. Tensorflow assumes this directory already exits so we need to create it.
                self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
                self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                # 创建Saver来管理模型中的所有变量, add ops to save and restore all the variables.
                self.saver = tf.train.Saver(tf.all_variables())

                # Write vocabulary, save the vocabulary to disk.
                vocab_processor.save(os.path.join(self.out_dir, "vocab"))

                self.sess.run(tf.initialize_all_variables())
                time_str = datetime.datetime.now().isoformat()
                self.fp = open("result_"+str(time_str), "w")

    def train_step(self, whether_word2vec, w_batch, y_batch, embedding):
        print('train whether_word2vec:', whether_word2vec)
        if whether_word2vec:
            feed_dict = {
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
                self.cnn.word_placeholder: embedding
            }
            _, step, summaries, loss, accuracy, predictions, embedding_init = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions, self.cnn.embedding_init], feed_dict=feed_dict)
        else:
            print("train not word2vec!", whether_word2vec)
            feed_dict = {
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
            }
            _, step, summaries, loss, accuracy, predictions, = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions], feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("train# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def train_step_tf(self, whether_word2vec, w_batch, t_tf_batch, a_tf_batch, j_tf_batch, y_batch, embedding):
        print('train whether_word2vec:', whether_word2vec)
        if whether_word2vec:
            feed_dict = {
                self.cnn.t_tf: t_tf_batch,
                self.cnn.a_tf: a_tf_batch,
                self.cnn.j_tf: j_tf_batch,
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
                self.cnn.word_placeholder: embedding
            }
            _, step, summaries, loss, accuracy, predictions, embedding_init = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions, self.cnn.embedding_init], feed_dict=feed_dict)
        else:
            feed_dict = {
                self.cnn.t_tf: t_tf_batch,
                self.cnn.a_tf: a_tf_batch,
                self.cnn.j_tf: j_tf_batch,
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
            }
            _, step, summaries, loss, accuracy, predictions, = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions], feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("train# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def test_step(self, whether_word2vec, w_batch, y_batch, embedding, writer=None):
        print("test step:")
        if whether_word2vec:
            feed_dict = {
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
                self.cnn.word_placeholder: embedding
            }
            _, step, summaries, loss, accuracy, predictions, embedding_init = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions, self.cnn.embedding_init], feed_dict=feed_dict)
        else:
            print('test not word2vec!', whether_word2vec)
            feed_dict = {
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
            }
            _, step, summaries, loss, accuracy, predictions, = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions], feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("test# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.fp.write("test# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.fp.write('\n')
        if writer:
            writer.add_summary(summaries, step)

    def test_step_tf(self, whether_word2vec, w_batch, t_tf_batch, a_tf_batch, j_tf_batch, y_batch, embedding, writer=None):
        if whether_word2vec:
            feed_dict = {
                self.cnn.t_tf: t_tf_batch,
                self.cnn.a_tf: a_tf_batch,
                self.cnn.j_tf: j_tf_batch,
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
                self.cnn.word_placeholder: embedding
            }
            _, step, summaries, loss, accuracy, predictions, embedding_init = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions, self.cnn.embedding_init], feed_dict=feed_dict)
        else:
            feed_dict = {
                self.cnn.t_tf: t_tf_batch,
                self.cnn.a_tf: a_tf_batch,
                self.cnn.j_tf: j_tf_batch,
                self.cnn.input_x: w_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
            }
            _, step, summaries, loss, accuracy, predictions, = \
                self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                               self.cnn.accuracy, self.cnn.predictions], feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("test# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.fp.write("test# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.fp.write('\n')
        if writer:
            writer.add_summary(summaries, step)

    def cnn_train(self, whether_word2vec, whether_tf, embedding, w_tr, w_te, t_tf_tr, t_tf_te, a_tf_tr, a_tf_te, j_tf_tr, j_tf_te, y_tr, y_te,
                  test_every=200, checkpoint_every=200, batch_size=64, num_epoch=10, shuffle=True):
        # Generate batches per_epoch and execute train step
        data_size = len(w_tr)
        num_batch_per_epoch = int(len(w_tr)/batch_size) + 1
        print("train...")
        print('whether_word2vec:', whether_word2vec)
        for epoch in range(num_epoch):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                s_w_tr = w_tr[shuffle_indices]
                s_y_tr = y_tr[shuffle_indices]
                s_t_tf_tr = t_tf_tr[shuffle_indices]
                s_a_tf_tr = a_tf_tr[shuffle_indices]
                s_j_tf_tr = j_tf_tr[shuffle_indices]
            else:
                s_w_tr = w_tr
                s_y_tr = y_tr
                s_t_tf_tr = t_tf_tr
                s_a_tf_tr = a_tf_tr
                s_j_tf_tr = j_tf_tr
            for batch_num in range(num_batch_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)
                if whether_tf:
                    self.train_step_tf(whether_word2vec, s_w_tr[start:end], s_t_tf_tr[start:end], s_a_tf_tr[start:end], s_j_tf_tr[start:end], s_y_tr[start:end], embedding)
                else:
                    self.train_step(whether_word2vec, s_w_tr[start:end], s_y_tr[start:end], embedding)
                current_step = tf.train.global_step(self.sess, self.global_step)
                if current_step % test_every == 0:
                    print("\nEvaluation:")
                    if whether_tf:
                        self.test_step_tf(whether_word2vec, w_te, t_tf_te, a_tf_te, j_tf_te, y_te, embedding, writer=self.dev_summary_writer)
                    else:
                        self.test_step(whether_word2vec, w_te, y_te, embedding, writer=self.dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
