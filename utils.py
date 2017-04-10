import numpy as np
from load_word2vec import *
import time,datetime
import re
import nltk


# 单独加载journal,因为训练集中journal的样例太少了
def load_all_journals(fp):
    # fp = open('v3/all_journal_1614_.txt', 'r')
    # lines = fp.readlines()
    journals_l = []
    for line in open(fp, 'r'):
        journals_l.append(line.lower())
    journals = [journal.split() for journal in journals_l]
    return journals


def load_all_v3titles(fp):
    # fp = open('v3/titles4v3.txt', 'r')
    # lines = fp.readlines()
    titles_l = []
    for line in open(fp, 'r'):
        titles_l.append(line.lower())
    titles = [title.split() for title in titles_l]
    return titles


def load_all_v3authors(fp):
    # fp = open('v3/authors4v3.txt', 'r')
    # fp = open('dataset_workshop/linked_authors_no_punctuation.txt', 'r') # linked author train CNN
    # lines = fp.readlines()
    authors_l = []
    for line in open(fp, 'r'):
        authors_l.append(line.lower())
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


def makePaddedList_index(maxl, sent_contents, pad_symbol):
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
def makePaddedList2(maxl, sent_contents, pad_symbol):
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


def makePosFeatures(sent_contents):
    """
    :param sent_contents:
    :return:sent_contents中每个sentence都构建一个list,存放sentence中每个word的标注信息.
    """
    pos_tag_list = []
    for sent in sent_contents:
        # print(sent)

        pos_tag = nltk.pos_tag(sent)
        # print(pos_tag)
        pos_tag = list(zip(*pos_tag))[1]    # 拆开pos_tag
        # print(pos_tag)
        pos_tag_list.append(pos_tag)
    return pos_tag_list


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


def sample2index_matrix2(samples, vocab):
    outer_index_list = []

    for sample in samples:
        outer_index_list.append(index_sample2(sample, vocab))
    return outer_index_list


def index_sample2(sample, vocab):
    inner_index_list = []
    for word in sample:
        index = np.where(vocab == word)
        if len(index[0]) != 0:
            i = index[0][0]
        else:
            i = 1   # vocab 中没有的就设定为<p>
        inner_index_list.append(i)
        if word == '<p>':
            break
    return inner_index_list


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
        # print('more title,error!')
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
        # print('more journal,error!')
        for i in j_index[0]:
            # print(i)
            if max_temp < loss[i][2]:
                max_temp = loss[i][2]
                max_index = i
        for i in j_index[0]:
            if i != max_index:
                predictions[i] = 1
        # print('title max index:', max_index)
    # print("after revise:", predictions)
    return predictions


# block = np.arrange(len(blocks))
def all_revise_predictions(predictions, loss, block):
    copy_predictions = predictions.copy()
    # print(copy_predictions)

    for p in predictions:
        max_temp = -10000
        max_index = 0
        # print(predictions)
        # print('p:', str(p))
        index = np.where(copy_predictions == p)
        # print(index[0])
        if len(index[0]) > 1:
            for i in index[0]:
                if max_temp < loss[i][p]:
                    max_temp = loss[i][p]
                    max_index = i
            # print('max_index:', str(max_index))
            # print('rest label:')
            # print(make_rest_label(copy_predictions, block))
            for i in index[0]:
                if i != max_index:
                    copy_predictions[i] = make_rest_label(copy_predictions, block)[0]
        # else:
        #     continue
        # print(copy_predictions)
        if (np.sort(copy_predictions) == block).all():
            result = [str(p) for p in copy_predictions]
            return result


def make_rest_label(predictions, block):
    rest_label = []
    for b in block:
        if b not in predictions:
            rest_label.append(b)
    return rest_label


# judge if string is more numeric or text
def n_or_t(block):
    reobj = re.compile('\d')
    results = reobj.findall(block)
    num_ratio = len(results)/len(block)
    # "page 1-9",这应该是number占比最小的情况了
    if num_ratio >= 0.25:
        token = 'n'
    else:
        token = 't'
    return token


def read_test_data(file_path):
    samples = []
    labels = []
    read = open(file_path, 'r')
    lines = read.readlines()
    # print(len(lines))
    for line in lines:
        if re.match(r'\[', line):
            label = [x for x in eval(line)]
            labels.append(label)
        else:
            samples.append(line)
    return samples, labels


# match numeric block and return the corresponding label:year-3,page-5,volume-4,
# if match failed return 6
def match_regex(block):
    label = 6
    year_regex1 = r'.*([1-2][0-9]{3})'  # '2014
    year_regex2 = r'.*(\([1-2][0-9]{3}\))'  # '(2014)'
    year_regex = "|".join([year_regex2, year_regex1])
    year_match_result = re.match(year_regex, block)

    page_regex1 = r'.*([0-9]+\-[0-9]+$)'
    page_regex2 = r'.*([0-9]+.\-.[0-9]+$)'
    page_regex3 = r'.*(pages.[0-9]+.\-.[0-9]+$)'
    page_regex4 = r'.*(pp.[0-9]+.\-.[0-9]+$)'
    page_regex = "|".join([page_regex1, page_regex2, page_regex3, page_regex4])
    page_match_result = re.match(page_regex, block)

    volume_regex = r'.[0-9]+\([0-9]+\)'
    volume_match_result = re.match(volume_regex, block)

    if year_match_result:
        label = 3
    elif page_match_result:
        label = 5
    elif volume_match_result:
        label = 4
    return label


def label_numeric(x_numeric):
    numeric_predictions = []
    for block in x_numeric:
        numeric_predictions.append(match_regex(block))
    return numeric_predictions


def fixed_length_list(num):
    lis = []
    for i in range(num):
        lis.append([])
    return lis


def same_elem_count(l1, l2):
    return len([i for i in l1 if i in l2])


# merge tow predictions
def merge_predictions(t_predictions, t_index, n_predictions, n_index):
    max_length = len(t_predictions) + len(n_predictions)
    predictions = fixed_length_list(max_length)
    c = 0
    for t_i in t_index:
        predictions[t_i] = t_predictions[c]
        c += 1
    d = 0
    for n_i in n_index:
        predictions[n_i] = n_predictions[d]
        d += 1
    return predictions


def build_y_train_publication(titles_contents, authors_contents, journals_contents):
    print("Building label dict:")
    titles_length = len(titles_contents)
    authors_length = len(authors_contents)
    journals_length = len(journals_contents)
    t_list = ['T' for i in range(titles_length)]
    a_list = ['A' for a in range(authors_length)]
    j_list = ['J' for j in range(journals_length)]
    label_list = t_list + a_list + j_list
    label_dict = {'T': 0, 'A': 1, 'J': 2}
    label_dict_size = len(label_dict)

    print("Preparing y_train:")
    y_t = mapLabelToId(label_list, label_dict)
    y_train = np.zeros((len(y_t), label_dict_size))
    for i in range(len(y_t)):
        y_train[i][y_t[i]] = 1
    print("Preparing y_train over!")
    return y_train, label_dict_size


if __name__ == '__main__':
    print('he.he.'.strip('.'))
    # j = load_all_journals()
    # print(j)
    # print('main')
    # samples, labels = read_test_data('data/temp_ada.txt')
    # for i in range(len(samples)):
    #     print('x:', samples[i])
    #     print('y:', labels[i])

    # print(len(samples))
    # print(len(predictions))
    # print(samples[10])
    # print(labels[10])
    # for i in labels[10]:
    #     print(i)
    # x = [0, 0, 1, 1, 1, 2, 2, 5, 3]
    # y = [0, 1, 1, 1, 1, 1, 2, 5, 3]
    #
    # a = float(same_elem_count(x, y))
    # print(a)

    # try:
    #     x_raw = samples[10].strip().split(',')
    #     print(x_raw)
    #     x_numeric = []
    #     x_text = []
    #     numeric_index = []
    #     text_index = []
    #     for x in x_raw:
    #         token = n_or_t(x)
    #         if token == 't':
    #             x_text.append(x)
    #             text_index.append(x_raw.index(x))
    #         if token == 'n':
    #             x_numeric.append(x)
    #             numeric_index.append(x_raw.index(x))
    #
    #     # x_text send into CNN model , got loss and predictions
    #     text_predictions = [0, 1, 1, 1, 1, 1, 2]
    #     # text_predictions = revise_predictions(predictions, loss)
    #
    #     num_predictions = label_numeric(x_numeric)
    #     print(num_predictions)
    #
    #
    #
    #     # input_x = [x.split() for x in x_raw]
    #     # print(input_x)
    # except Exception as e:
    #     print("Exception:%s" % e)


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



