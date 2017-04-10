import tensorflow as tf
import os
import time
import datetime
import numpy as np
from text_cnn_pos import *


class TrainCNN_POS(object):
    def __init__(self, vocab_size, embedding_dim, pos_vocab_size, pos_emb_dim, sequence_length, num_classes, filter_sizes=[1, 2, 3],
                 num_filters=50):
        g = tf.Graph()
        with g.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.cnn = TextCNN_POS(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    pos_vocab_size=pos_vocab_size,
                    pos_emb_dim=pos_emb_dim,
                    sequence_length=sequence_length,
                    num_classes=num_classes,
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
                # vocab_processor.save(os.path.join(self.out_dir, "vocab"))

                self.sess.run(tf.initialize_all_variables())
                time_str = datetime.datetime.now().isoformat()
                self.fp = open("result_"+str(time_str), "w")

    def train_step(self, w_batch, p_batch, y_batch):

        feed_dict = {
            self.cnn.input_x: w_batch,
            self.cnn.input_pos: p_batch,
            self.cnn.input_y: y_batch,
            self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
        }
        _, step, summaries, loss, accuracy, predictions = \
            self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                           self.cnn.accuracy, self.cnn.predictions], feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("train# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def test_step(self, w_batch, p_batch, y_batch, writer=None):
        feed_dict = {
            self.cnn.input_x: w_batch,
            self.cnn.input_pos: p_batch,
            self.cnn.input_y: y_batch,
            self.cnn.dropout_keep_prob: 0.5,     # set 0.5 at train step
        }
        _, step, summaries, loss, accuracy, predictions = \
            self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss,
                           self.cnn.accuracy, self.cnn.predictions], feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("test# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.fp.write("test# {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.fp.write('\n')
        if writer:
            writer.add_summary(summaries, step)

    def cnn_train_pos(self, w_tr, w_te, p_tr, p_te, y_tr, y_te, test_every=200, checkpoint_every=200, batch_size=64, num_epoch=10,
                         shuffle=True):
        # Generate batches per_epoch and execute train step
        data_size = len(w_tr)
        num_batch_per_epoch = int(len(w_tr)/batch_size) + 1
        print("train...")
        for epoch in range(num_epoch):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                s_w_tr = w_tr[shuffle_indices]
                p_w_tr = p_tr[shuffle_indices]
                s_y_tr = y_tr[shuffle_indices]
            else:
                s_w_tr = w_tr
                p_w_tr = p_tr
                s_y_tr = y_tr
            for batch_num in range(num_batch_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)

                self.train_step(s_w_tr[start:end], p_w_tr[start:end], s_y_tr[start:end])
                current_step = tf.train.global_step(self.sess, self.global_step)
                if current_step % test_every == 0:
                    print("\nEvaluation:")
                    # test_length = 2000
                    # if int(len(w_te)/10) < 2000:
                    test_length = int(len(w_te)/5)
                    self.test_step(w_te[0:test_length], p_te[0:test_length], y_te[0:test_length], writer=self.dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
