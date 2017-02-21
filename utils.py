import numpy as np
from load_word2vec import *
import time,datetime
import re

# 单独加载journal,因为训练集中journal的样例太少了
def load_all_journals():
    fp = open('v3/all_journal_1614_.txt', 'r')
    lines = fp.readlines()
    journals_l = []
    for line in lines:
        journals_l.append(line.strip('.'))
    journals = [journal.split() for journal in journals_l]
    return journals


def load_all_v3titles():
    fp = open('v3/titles4v3.txt', 'r')
    lines = fp.readlines()
    titles_l = []
    for line in lines:
        titles_l.append(line.strip('.'))
    titles = [title.split() for title in titles_l]
    return titles


def load_all_v3authors():
    fp = open('v3/authors4v3.txt', 'r')
    lines = fp.readlines()
    authors_l = []
    for line in lines:
        authors_l.append(line.strip('.'))
    authors = [author.split() for author in authors_l]
    return authors


# read samples from training dataset,return list of all titles, authors and journals without duplicates sample.
def readData2(ftrain):
    titles_set = set()
    authors_set = set()
    # journals_set = set()

    title_length = 0
    journal_length = 0
    fp = open(ftrain, 'r')
    samples = fp.readlines()
    for sample in samples:
        # print(sample)
        temp = sample.strip().split('#$')
        if len(temp) == 3:
            title = temp[0]
            authors = temp[1]
            journal = temp[2]
            # title, authors, journal = sample.strip().split('#$')
            if len(title.split(' ')) > title_length:
                title_length = len(title.split(' '))
            if len(journal.split(' ')) > journal_length:
                journal_length = len(journal.split(' '))
            # build titles set
            titles_set.add(title.strip('.'))
            # build author set
            author = authors.split(',')
            for a in author:
                authors_set.add(a)
            # build journal set
            # journals_set.add(journal.strip('.'))

    titles_list = [t.split() for t in titles_set]
    authors_list = [s.split() for s in authors_set if s != '']
    # journals_list = [j.split() for j in journals_set]

    return max(title_length, journal_length), titles_list, authors_list     #, journals_list


def makePaddedList(maxl, sent_contents, pad_symbol='<p>'):
    # maxl = max([len(sent) for sent in sent_contents])
    # print("padding maxl:", maxl)
    T = []
    for sent in sent_contents:
        t = []
        lenth = len(sent)
        for i in range(lenth):
            t.append(sent[i])
        for i in range(lenth, maxl):
            t.append(pad_symbol)
        T.append(t)
    return T


# longer-trim, shorter-padding
def makePaddedList2(maxl, sent_contents, pad_symbol='<p>'):
    # maxl = max([len(sent) for sent in sent_contents])
    # print("padding maxl:", maxl)
    T = []
    for sent in sent_contents:
        t = []
        lenth = len(sent)
        if lenth < maxl:
            for i in range(lenth):
                t.append(sent[i])
            for i in range(lenth, maxl):
                t.append(pad_symbol)
        else:
            for i in range(maxl):
                t.append(sent[i])
        T.append(t)
    return T


def makeWordList(sent_list):
    """

    :param sent_list:
    :return:返回一个字典,{'word1':index1,'word2':index2,....},index从1开始
    """
    wf = {}
    for sent in sent_list:      # 构造字典wf,键是单个word,值是出现的次数
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0

    wl = {}
    i = 0
    wl['unkown'] = 0
    for w, f in wf.items():     # 构造字典wl,键是单个word, 值是下标,从1开始
        # print(w, ' ', f)
        i += 1
        wl[w] = i
    return wl


def mapWordToId(sent_contents, word_dict):
    """

    :param sent_contents:
    :param word_dict:
    :return:把所有的sentence变成矩阵(None, embedding_dimensionality),横坐标是每个word在word_dict中对应的index,纵坐标是每个sentence
    """
    T = []
    for sent in sent_contents:
        t = []
        for w in sent:
            t.append(word_dict[w])
        T.append(t)
    return T


def mapLabelToId(sent_labels, label_dict):
    return [label_dict[label] for label in sent_labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有yield说明不是一个普通函数，是一个Generator.
    函数效果：对data，一共分成num_epochs个阶段（epoch），在每个epoch内，如果shuffle=True，就将data重新洗牌，
    批量生成(yield)一批一批的重洗过的data，每批大小是batch_size，一共生成int(len(data)/batch_size)+1批。
    Generate a  batch iterator for a dataset.
    :param data:
    :param batch_size:每批data的size
    :param num_epochs:阶段数目
    :param shuffle:洗牌
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int(len(data)/batch_size) + 1  # 每段的batch数目
    for epoch in range(num_epochs):
        if shuffle:
            # np.random.permutation(),得到一个重新排列的序列(Array)
            # np.arrange(),得到一个均匀间隔的array.
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]    # 重新洗牌的data
        else:
            shuffle_data = data
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]   # all elements index between start_index and end_inde


# convert samples data [[w1,w2,,,],[],[],[]]to matrix [[i1,i2],[],[],[]]according the vocabulary,
# matrix's indexes are the location of the word in the vocabulary.
def sample2index_matrix(samples, vocab, max_length):
    outer_index_list = []

    for sample in samples:
        outer_index_list.append(index_sample(sample, vocab, max_length))
    return outer_index_list


def index_sample(sample, vocab, max_length):
    current_index = 0
    inner_index_list = []
    for word in sample:
        index = np.where(vocab == word)
        if len(index[0]) != 0:
            i = index[0][0]
        else:
            i = 1   # vocab 中没有的就设定为<p>
        inner_index_list.append(i)
        if word == '<p>':
            for c in range(current_index, max_length-1):
                inner_index_list.append(1)
            break
        current_index += 1

    return inner_index_list


# 对预测结果的loss,计算出各类得分的最大值
def get_max_score(loss):
    m1 = 0
    m2 = 0
    m3 = 0
    arr = np.array(loss)
    arr.max()


def load_journal4test(journal_path):
    o = open(journal_path, 'r')
    lines = o.readlines()
    journals4test = []
    j_labels = []
    for journal in lines:
        # print(journal)
        journals4test.append(journal.strip())
        j_labels.append(2)
    return journals4test, j_labels


def load_author4test(author_path):
    o = open(author_path, 'r')
    lines = o.readlines()
    authors4test = []
    a_labels = []
    for author in lines:
        # print(author)
        authors4test.append(author.strip())
        a_labels.append(1)
    return authors4test, a_labels


def load_title4test(title_path):
    o = open(title_path, 'r')
    lines = o.readlines()
    titles4test = []
    t_labels = []
    for title in lines:
        # print(title)
        titles4test.append(title.strip())
        t_labels.append(0)
    return titles4test, t_labels


# save experiment result,only save samples which was categorized wrongly.
def save_experiment_result(result_path, x_raw, y_test, predictions, Accuracy):
    write = open(result_path, 'w+')
    size = len(x_raw)
    l = ''
    p = ''
    write.write('Classification Accuracy: ' + str(Accuracy)+'\n')
    for i in range(size):
        label = str(y_test[i])
        prediction = str(predictions[i])
        if label != prediction:
            if label == '0':
                l = 'T'
            elif label == '1':
                l = 'A'
            elif label == '2':
                l = 'J'

            if prediction == '0':
                p = 'T'
            elif prediction == '1':
                p = 'A'
            elif prediction == '2':
                p = 'J'
            print(x_raw[i]+' '+l+' '+p)
            write.write(x_raw[i]+'\t'+l+'\t'+p+'\n')
    write.close()


def load_data_not_word2vec():
    titles = []
    authors = []
    journal = []
    j_fp = open('v3/all_journal_1614_.txt', 'r')
    j_lines = j_fp.readlines()
    for line in j_lines:
        journal.append(line.strip())
    j_fp.close()

    t_fp = open('v3/titles4v3.txt', 'r')
    t_lines = t_fp.readlines()
    for line in t_lines:
        titles.append(line.strip())
    t_fp.close()

    a_fp = open('v3/authors4v3.txt', 'r')
    a_lines = a_fp.readlines()
    for line in a_lines:
        authors.append(line.strip())
    a_fp.close()
    return titles, authors, journal


# 调整预测结果,因为每条记录都只有一个title和一个journal,所以在softmax的概率中,当模型的预测相同时,
# 概率相对更大的被调整成相应label.
# revise the predictions according the strategy.
def revise_predictions(predictions, loss):
    t_index = np.where(predictions == 0)
    # print(t_index)
    max_temp = -10000
    max_index = 0
    if len(t_index[0]) > 1:
        print('more title,error!')
        for i in t_index[0]:
            if max_temp < loss[i][0]:
                max_temp = loss[i][0]
                max_index = i
        for i in t_index[0]:
            if i != max_index:
                predictions[i] = 2
        # print('title max index:', max_index)
        # print(predictions)
    j_index = np.where(predictions == 2)
    # print(j_index)
    if len(j_index[0]) > 1:
        print('more journal,error!')
        for i in j_index[0]:
            print(i)
            if max_temp < loss[i][2]:
                max_temp = loss[i][2]
                max_index = i
        for i in j_index[0]:
            if i != max_index:
                predictions[i] = 1
        # print('title max index:', max_index)
    print("after revise:", predictions)
    return predictions


def read_test_data(file_path):
    predictions = []
    read = open(file_path, 'r')
    lines = read.readlines()
    for line in lines:
        if re.match(r'\[', line):
            predictions = [x for x in eval(line)]
        else:
            print(line)

if __name__ == '__main__':
    print('main')
    read_test_data('data/temp_ada')

'''
    print('main')
    # load word2vec array
    print("loading word2vec:")
    path = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
    vocab, embedding = load_from_binary(path)
    vocab_size, embedding_dim = embedding.shape

    sample = ['On', 'a', 'Gauntlet', 'Thrown', 'by', 'David', 'Gries', '<p>', '<p>', '<p>']

    print(len(sample))
    l = index_sample(sample, vocab, 10)
    print(l)
    print(len(l))

    sample2 = [['On', 'a', 'Gauntlet', 'Thrown', 'by', 'David', 'Gries'],
               ['Foundations', 'and', 'Trends', 'in', 'Programming', 'Languages'],
               ['124', '-', '125', 'ksfdoikjlk']]
    l2 = sample2index_matrix(sample2, vocab, 10)
    print(l2)
    print(len(l2))
    # title4test, t_labels = load_journal4test("v3/all_journal_1614_.txt")
    title4test, t_labels = load_author4test("v3/authors4test.txt")
    title4test, t_labels = load_title4test("v3/titles4test.txt")
    print(title4test)
    print(t_labels)
    print(len(title4test))
    print(len(t_labels))
    print(title4test[0])
    print(title4test[1])
    titles, authors, journals = load_data_not_word2vec()
    x = titles + authors + journals
    print(titles[0])
    print(len(titles))
    print(authors[0])
    print(authors)
    print(len(authors))
    print(journals[0])
    print(len(journals))

    print(len(x))
    pad1 = makePaddedList(10, sample2)
    pad11 = makePaddedList(5, sample2)
    pad2 = makePaddedList2(5, sample2)
    print(pad1)
    print(pad11)
    print(pad2)
'''



