import nltk
# import tensorflow as tf
l2 = 'Wei-Hsi Hung,Kuanchin Chen and Chieh-Pin Lin,Does the proactive personality mitigate the adverse effect of technostress on productivity in the mobile environment,Telematics and Informatics,2015,32(6),143-157'

words = nltk.word_tokenize('1 4 3 - 1 5 7')
print(words)
word_tag = nltk.pos_tag(words)
print(word_tag)


# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))