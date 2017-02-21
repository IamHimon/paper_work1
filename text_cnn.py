import tensorflow as tf


class TextCNN(object):
    def __init__(self, whether_word2vec, whether_tf, vocab_size, embedding_dim, sequence_length, tf_dict_size, tf_emb_size, num_classes, filter_sizes,
                 num_filters, l2_reg_lambda=0.0):

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # sum_dim = embedding_dim + tf_emb_size * 3
        sum_dim = 0

        # placeholder for input,output and dropout, wait for feed_dict
        self.t_tf = tf.placeholder(tf.int32, [None, sequence_length], name='t_tf')
        self.a_tf = tf.placeholder(tf.int32, [None, sequence_length], name='a_tf')
        self.j_tf = tf.placeholder(tf.int32, [None, sequence_length], name='j_tf')
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 根据标志whether_word2vec来选择两种方式
            if whether_word2vec:
                # 直接用word2vec训练好的词向量,令trainable=False,
                word2vec = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="word2vec")
                self.word_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim], name="word_placeholder")
                self.embedding_init = word2vec.assign(self.word_placeholder)
                sum_dim += embedding_dim

            else:
                print('whether_word2vec')
                # 如果不用事先训练好的word2vec词向量,就需要随机初始化来构造一个词向量.
                # initialize these using a random uniform distribution
                word2vec = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1, +1), name="word2vec")
            # embedding_lookup() creates the actual embedding operation
            # aim shape: [None, seq_len, embedding_size, 1]
            w_emb = tf.nn.embedding_lookup(word2vec, self.input_x)
            self.X = w_emb

            if whether_tf:
                ini_tf_emb = tf.Variable(tf.random_uniform([tf_dict_size, tf_emb_size], -1.0, +1.0), 'tf_vec')
                t_tf_emb = tf.nn.embedding_lookup(ini_tf_emb, self.t_tf)
                a_tf_emb = tf.nn.embedding_lookup(ini_tf_emb, self.a_tf)
                j_tf_emb = tf.nn.embedding_lookup(ini_tf_emb, self.j_tf)
                self.X = tf.concat(2, [self.X, t_tf_emb, a_tf_emb, j_tf_emb])   # 更新X
                sum_dim += tf_emb_size * 3  # 更新sum dimension

            # 直接用word2vec训练好的词向量,令trainable=False,
            # word2vec = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="word2vec")
            # self.word_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim], name="word_placeholder")
            # self.embedding_init = word2vec.assign(self.word_placeholder)

            # initialize tf_dic using a random uniform distribution
            # ini_tf_emb = tf.Variable(tf.random_uniform([tf_dict_size, tf_emb_size], -1.0, +1.0))

            # embedding_lookup() creates the actual embedding operation,三个特征值用同一个词典
            # aim shape: [None, seq_len, embedding_size, 1]
            # t_tf_emb = tf.nn.embedding_lookup(ini_tf_emb, self.t_tf)
            # a_tf_emb = tf.nn.embedding_lookup(ini_tf_emb, self.a_tf)
            # j_tf_emb = tf.nn.embedding_lookup(ini_tf_emb, self.j_tf)

            # w_emb = tf.nn.embedding_lookup(word2vec, self.input_x)

            # self.X = tf.concat(2, [w_emb, t_tf_emb, a_tf_emb, j_tf_emb])
            self.X_expanded = tf.expand_dims(self.X, -1)
        # convolution and max-polling layers
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, sum_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="c_W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="c_b")
                conv = tf.nn.conv2d(
                    self.X_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Fully connected payer ,scores and predections
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="f_W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="f_b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # self.softmax_score = tf.nn.softmax(self.scores, name="softmax_scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
