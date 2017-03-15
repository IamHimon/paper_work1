import tensorflow as tf


class TextCNN_ONEHOT(object):
    def __init__(self, vocab_size, embedding_dim, sequence_length, num_classes, filter_sizes,
                 num_filters, l2_reg_lambda=0.0):
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        sum_dim = embedding_dim

        # placeholder for input,output and dropout, wait for feed_dict
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # initialize these using a random uniform distribution
            word2vec = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1, +1), name="word2vec")
            # embedding_lookup() creates the actual embedding operation
            # aim shape: [None, seq_len, embedding_size, 1]
            self.w_emb = tf.nn.embedding_lookup(word2vec, self.input_x)

            self.X_expanded = tf.expand_dims(self.w_emb, -1)
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
