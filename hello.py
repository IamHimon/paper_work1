import tensorflow as tf
from utils import *
from load_word2vec import *
import random
import re


def test_tf(embedding, input_x, sequence_length):
    input_x = tf.placeholder(tf.string, [None, sequence_length], name="input_x")
    #
    # word2vec = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="word2vec")
    # word_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim], name="word_placeholder")
    # word2vec.assign(word_placeholder)

    emb = tf.nn.embedding_lookup(embedding, input_x)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        # sess.run(init_op, feed_dict={input_x: input_x, word_placeholder: embedding})

        print(sess.run(emb))


def index_vocab(word, vocab):
    index = 0
    for w in vocab:
        index += 1
        if w == word:
            return index


def sample2index(samples, vocab):
    inner_index_list = []
    outer_index_list = []

    for sample in samples:
        print(sample)
        for word in sample:
            index = np.where(vocab == word)
            if len(index[0]) != 0:
                i = index[0][0]
            else:
                i = 0
            inner_index_list.append(i)
        outer_index_list.append(inner_index_list)
        inner_index_list = []
    return outer_index_list


def test1():
    # x_raw = ["Optimizing Event Pattern Matching Using Business Process Models",
    #      "Binbin Gu",
    #      "Zhixu Li",
    #      "Frequent Closed Sequence Mining without Candidate Maintenance",
    #      "Chen Yang",
    #      "Yelena Yesha"]
    # y_test = [0, 1, 1, 0, 1, 1]
    #
    # input_list = [x.split() for x in x_raw]
    # for i in input_list:
    #     print(i)
    # input_pad = makePaddedList(36, input_list)
    # for i in input_pad:
    #     print(i)
    #
    # # # load word2vec array
    # print("loading word2vec:")
    # path = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
    # vocab, embedding = load_from_binary(path)
    # vocab_size, embedding_dim = embedding.shape

    # print(index_vocab('Sequence', vocab))
    # print(np.where(vocab == 'Sequence')[0][0])
    # index = np.where(vocab == '<pad>')
    # if len(index[0]) == 0:
    #     index = 0
    # print(index)
    #
    # index_list = sample2index_matrix(input_pad, vocab, 36)
    # print(index_list)

    # test_tf(embedding, index_list, 36)

    # sample = ['Agents-based', 'design', 'for', 'fault', 'management', 'systems', 'in', 'industrial', 'processes', '<p>']
    # l = sample2index(sample, vocab)
    # print(l)
    loss = [[-7.81613398, 12.59275246, -9.82943439],
             [-8.21374607, 11.90228271, -7.99127913],
             [-8.09581947, 11.60901165, -7.79938269],
             [-6.60628796, 13.57494068, -11.94314194],
             [-8.11062527, 12.61278534,  -9.55984306],
             [-11.80429077, 18.12654686, -14.37622833],
             [-12.34054852, 17.99167442, -13.79800224],
             [-8.32250309, 12.67250538, -9.48696995],
             [21.00398254, -13.31361675, -11.81652069],
             [20.87360764, -14.28156853, -9.39016056]]
    index = [i for i in range(len(loss))]
    print(index)
    loss_dict = dict(zip(index, loss))
    print(loss_dict)
    predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    # predictions = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    t_index = np.where(predictions == 0)
    print(t_index)
    t_num = len(t_index[0])
    title_index = 0
    max_temp = -1000
    max_index = 0
    for i in t_index[0]:
        if max_temp < loss[i][0]:
            max_temp = loss[i][0]
            max_index = i
        print(i)
        print(loss[i])
    print(max_index)


def build_date_data(date_size):
    dates = []
    date = []
    for s in range(date_size):
        p = random.sample([i for i in range(500)], 2)
        date.append(str(p[0]))
        date.append('_')
        date.append(str(p[1]))
        dates.append(date)
        date = []
    return dates


if __name__ == '__main__':
    print('main')
    # dates = build_date_data(20)
    # print(dates)
    # n = np.zeros([2, 2])
    # print(n)
    s1 = "Spatio-temporal Event Modeling and Ranking, Xuefei Li, Hongyun Cai, Zi Huang, Yang Yang and Xiaofang Zhou, " \
         "14th International Conference on Web Information System Engineering,2013"
    s2 = "[0,1,1,1,1,1,2,3]"
    matchObj = re.match(r'\[', s2)
    print(matchObj)
    l = [x for x in eval(s2)]
    print(l)







