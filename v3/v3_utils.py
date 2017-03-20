from decimal import Decimal as D
from decimal import getcontext
from decimal import setcontext
import re
getcontext().prec = 5


#  calculate TF(term frequency)
def cal_tf(word, word_count_dict, sum_word_count):
    t_tf = 0.0
    if word in word_count_dict.keys():
        t_tf = word_count_dict[word] / sum_word_count
        # t_tf = '%.5f' % (word_count_dict[word] / sum_word_count)
    return t_tf


# calculate one sample's tf value in dataset
# return: [w1_tf,w2_tf,,,,,,wn_t]
def make_tf_feature(sent_contents, word_count_dict, sum_word_coun):
    # t_path = '/home/himon/PycharmProjects/paper_work1/v3/titles4v3.txt'
    # t_word_count, t_sum_word_count = count_word(t_path)
    tf = []
    temp = []
    for sent in sent_contents:
        # print(sent)
        for title in sent:
            # print(cal_title_tf(title.lower(), t_word_count, t_sum_word_count))
            temp.append(cal_tf(title.lower(), word_count_dict, sum_word_coun))
        tf.append(temp)
        temp = []
    return tf


# 归一化,对于sample中的同一个word,他的三个tf值相加等于1
# 输入不需要单独输入各个tf list,而作为整个list输入
# 而且输出不是字符串格式
#返回也是返回一整个list
def normalize_tf_v2(all_tf_list):
    sum = 0.0
    temp = []
    result = []
    for j in range(len(all_tf_list[0])):
        for i in range(len(all_tf_list)):
            sum += all_tf_list[i][j]
        # print('sum=', sum)
        for i in range(len(all_tf_list)):
            temp.append(float('%.5f' % (all_tf_list[i][j] / sum)))
        result.append(temp)
        temp = []
        # print('====================')
    return result

# 从文件中读取内容，统计词频
def count_word(path):
    result = {}
    with open(path) as file_obj:
        all_the_text = file_obj.read()
        #大写转小写
        all_the_text = all_the_text.lower()
        #正则表达式替换特殊字符
        all_the_text = re.sub("\"|,|\.", "", all_the_text)

        word_list = all_the_text.split()
        sum_word_count = len(word_list)
        # print(sum_word_count)
        for word in word_list:
            if word not in result:
                result[word] = 0
            result[word] += 1

        return result, sum_word_count


def get_normalize_tf_result():
    print('tf')


# 构造词频的字典,区间是[0,1],间隔0.0001,一共1001个.
def build_tf_dic():
    a = D(0.0)
    tf_list = ['0.00000']
    for i in range(100000):
        a += D(0.00001)
        tf_list.append(str('%.5f' % a))
        # print('%.5f' % a)
    tf_dic = {}
    index = 0
    for j in tf_list:
        index += 1
        tf_dic[j] = index
    return tf_dic


def read_title(file_path):
    output_path = 'titles4v3.txt'
    write = open(output_path, 'w+')
    o = open(file_path, 'r')
    titles = o.readlines()
    count = 0
    for title in titles:
        print(title)
        write.write(title)
        count += 1
        if count > 5000:
            break


def build_test_title(file_path):
    output_path = 'titles4test.txt'
    write = open(output_path, 'w+')
    o = open(file_path, 'r')
    titles = o.readlines()
    count = 0
    for title in titles:
        count += 1
        if count < 5000:
            continue
        print(title)
        write.write(title)
        if count > 10000:
            break


def read_author(file_path):
    output_path = 'authors4v3.txt'
    write = open(output_path, 'w+')
    o = open(file_path, 'r')
    titles = o.readlines()
    count = 0
    for title in titles:
        print(title)
        write.write(title)
        count += 1
        if count > 5000:
            break


def build_test_author(file_path):
    output_path = 'authors4test.txt'
    write = open(output_path, 'w+')
    o = open(file_path, 'r')
    titles = o.readlines()
    count = 0
    for title in titles:
        count += 1
        if count < 5000:
            continue
        print(title)
        write.write(title)
        if count > 7000:
            break


def save_dict(dic, sum_count, write):
    write.write(str(sum_count)+'\n')
    for key, value in dic.items():
        write.write(str(key)+'\t'+str(value)+'\n')
    write.close()


def save_tf_dic():
    # title
    title_path = 'titles4v3.txt'
    title_dic_output = 'titles_dic.txt'
    t_write = open(title_dic_output, 'w+')
    title_count, sum_title_count = count_word(title_path)
    save_dict(title_count, sum_title_count, t_write)

    # author
    author_path = 'authors4v3.txt'
    author_dic_output = 'authors_dict.txt'
    a_write = open(author_dic_output, 'w+')
    author_count, sum_author_count = count_word(author_path)
    save_dict(author_count, sum_author_count, a_write)

    # journal
    j_path = 'all_journal_1614_.txt'
    j_dic_output = 'journals_dict.txt'
    j_write = open(j_dic_output, 'w+')
    j_count, sum_j_count = count_word(j_path)
    save_dict(j_count, sum_j_count, j_write)


# calculate title tf
def cal_title_tf(word, t_word_count, t_sum_word_count):
    t_tf = 0.0
    if word in t_word_count.keys():
        t_tf = t_word_count[word] / t_sum_word_count
    return t_tf


# calculate author tf
def cal_author_tf(word, a_word_count, a_sum_word_count):
    a_tf = 0.0
    if word in a_word_count.keys():
        a_tf = a_word_count[word] / a_sum_word_count
    return a_tf


# calculate journal tf
def cal_journal_tf(word, j_word_count, j_sum_word_cout):
    j_tf = 0.0
    if word in j_word_count.keys():
        j_tf = j_word_count[word] / j_sum_word_cout
    return j_tf


# calculate year tf
def cal_year_tf(word, y_word_count, y_sum_word_cout):
    j_tf = 0.0
    if word in y_word_count.keys():
        j_tf = y_word_count[word] / y_sum_word_cout
    return j_tf


# calculate journal tf
def cal_volume_tf(word, v_word_count, v_sum_word_cout):
    j_tf = 0.0
    if word in v_word_count.keys():
        j_tf = v_word_count[word] / v_sum_word_cout
    return j_tf


# calculate journal tf
def cal_pages_tf(word, p_word_count, p_sum_word_cout):
    j_tf = 0.0
    if word in p_word_count.keys():
        j_tf = p_word_count[word] / p_sum_word_cout
    return j_tf


def cal_word_tf(word):
    # calculate title tf
    t_path = 'titles4v3.txt'
    t_word_count, t_sum_word_count = count_word(t_path)
    t_tf = 0.0
    if word in t_word_count.keys():
        t_tf = t_word_count[word] / t_sum_word_count
    print(t_tf)

    # calculate author tf
    a_path = 'authors4v3.txt'
    a_word_count, a_sum_word_count = count_word(a_path)
    a_tf = 0.0
    if word in a_word_count.keys():
        a_tf = a_word_count[word] / a_sum_word_count
    print(a_tf)

    # calculate journal tf
    j_path = 'all_journal_1614_.txt'
    j_word_count, j_sum_word_cout = count_word(j_path)
    j_tf = 0.0
    if word in j_word_count.keys():
        j_tf = j_word_count[word] / j_sum_word_cout
    print(j_tf)

    # 归一化
    normalized_t_tf = t_tf / (t_tf + a_tf + j_tf)
    normalized_a_tf = a_tf / (t_tf + a_tf + j_tf)
    normalized_j_tf = j_tf / (t_tf + a_tf + j_tf)

    print(normalized_a_tf)
    print(normalized_j_tf)
    print(normalized_t_tf)
    return normalized_t_tf, normalized_a_tf, normalized_j_tf




# calculate one sample's tf value in title dataset
# return: [w1_tf,w2_tf,,,,,,wn_t]
def make_title_tf_feature(sent_contents):
    t_path = '/home/himon/PycharmProjects/paper_work1/v3/titles4v3.txt'
    t_word_count, t_sum_word_count = count_word(t_path)
    title_tf = []
    temp = []
    for sent in sent_contents:
        # print(sent)
        for title in sent:
            # print(cal_title_tf(title.lower(), t_word_count, t_sum_word_count))
            temp.append(cal_title_tf(title.lower(), t_word_count, t_sum_word_count))
        title_tf.append(temp)
        temp = []
    return title_tf


# calculate one sample's tf value in author dataset
# return: [w1_tf,w2_tf,,,,,,wn_t]
def make_author_tf_feature(sent_contents):
    a_path = '/home/himon/PycharmProjects/paper_work1/v3/authors4v3.txt'
    a_word_count, a_sum_word_count = count_word(a_path)
    author_tf = []
    temp = []
    for sent in sent_contents:
        for author in sent:
            temp.append(cal_author_tf(author.lower(), a_word_count, a_sum_word_count))
        author_tf.append(temp)
        temp = []
    return author_tf


# calculate one sample's tf value in journal dataset
# return: [w1_tf,w2_tf,,,,,,wn_t]
def make_journal_tf_feature(sent_contents):
    j_path = '/home/himon/PycharmProjects/paper_work1/v3/all_journal_1614_.txt'
    j_word_count, j_sum_word_cout = count_word(j_path)
    journal_tf = []
    temp = []
    for sent in sent_contents:
        for journal in sent:
            temp.append(cal_journal_tf(journal.lower(), j_word_count, j_sum_word_cout))
        journal_tf.append(temp)
        temp = []
    return journal_tf


# calculate one sample's tf value in journal dataset
# return: [w1_tf,w2_tf,,,,,,wn_t]
def make_year_tf_feature(sent_contents):
    j_path = '/home/himon/PycharmProjects/paper_work1/v3/all_journal_1614_.txt'
    j_word_count, j_sum_word_cout = count_word(j_path)
    journal_tf = []
    temp = []
    for sent in sent_contents:
        for journal in sent:
            temp.append(cal_journal_tf(journal.lower(), j_word_count, j_sum_word_cout))
        journal_tf.append(temp)
        temp = []
    return journal_tf


# calculate one sample's tf value in journal dataset
# return: [w1_tf,w2_tf,,,,,,wn_t]
def make_volume_tf_feature(sent_contents):
    j_path = '/home/himon/PycharmProjects/paper_work1/v3/all_journal_1614_.txt'
    j_word_count, j_sum_word_cout = count_word(j_path)
    journal_tf = []
    temp = []
    for sent in sent_contents:
        for journal in sent:
            temp.append(cal_journal_tf(journal.lower(), j_word_count, j_sum_word_cout))
        journal_tf.append(temp)
        temp = []
    return journal_tf


# calculate one sample's tf value in journal dataset
# return: [w1_tf,w2_tf,,,,,,wn_t]
def make_pages_tf_feature(sent_contents):
    j_path = '/home/himon/PycharmProjects/paper_work1/v3/all_journal_1614_.txt'
    j_word_count, j_sum_word_cout = count_word(j_path)
    journal_tf = []
    temp = []
    for sent in sent_contents:
        for journal in sent:
            temp.append(cal_journal_tf(journal.lower(), j_word_count, j_sum_word_cout))
        journal_tf.append(temp)
        temp = []
    return journal_tf


# 归一化,对于sample中的同一个word,他的三个tf值相加等于1
def normalize_tf(title_tf, author_tf, journal_tf):
    # print('normalize')
    nor_t = []
    nor_a = []
    nor_j = []
    t_temp = []
    a_temp = []
    j_temp = []
    sent_num = len(title_tf)
    for i in range(sent_num):
        t_sent = title_tf[i]
        a_sent = author_tf[i]
        j_sent = journal_tf[i]
        sent_length = len(t_sent)
        for j in range(sent_length):
            if (t_sent[j]+a_sent[j]+a_sent[j]) != 0.0:
                t_temp.append(str('%.5f' % (t_sent[j] / (t_sent[j] + a_sent[j] + j_sent[j]))))
                a_temp.append(str('%.5f' % (a_sent[j] / (t_sent[j] + a_sent[j] + j_sent[j]))))
                j_temp.append(str('%.5f' % (j_sent[j] / (t_sent[j] + a_sent[j] + j_sent[j]))))
                # t_sent[j] = str('%.6f' % (t_sent[j] / (t_sent[j] + a_sent[j] + j_sent[j])))
                # a_sent[j] = str('%.6f' % (a_sent[j] / (t_sent[j] + a_sent[j] + j_sent[j])))
                # j_sent[j] = str('%.6f' % (j_sent[j] / (t_sent[j] + a_sent[j] + j_sent[j])))
            else:
                t_temp.append('0.00000')
                a_temp.append('0.00000')
                j_temp.append('0.00000')
        nor_a.append(a_temp)
        nor_j.append(j_temp)
        nor_t.append(t_temp)
        t_temp = []
        a_temp = []
        j_temp = []
    # print(nor_t)
    # print(nor_a)
    # print(nor_j)
    return nor_t, nor_a, nor_j


def lower_journals():
    fo = open('all_journal_1614_.txt', 'r')
    fw = open('../dataset_workshop/lower_all_journal.txt', 'w+')
    journals = fo.readlines()
    for j in journals:
        fw.write(j.lower())
    fo.close()
    fw.close()

if __name__ == '__main__':
    # file_name = '../dataset_workshop/temp_titles_kb.txt'
    # title_word_count_dict, title_sum_word_count = count_word(file_name)
    # l2 = [['Mathematics', 'and', 'Computers', 'in', 'Simulation']]
    # tf = make_tf_feature(l2, title_word_count_dict, title_sum_word_count)
    # print(tf)
    # print(sum_word_count)
    # print(tf)
    #
    # for key, value in tf.items():
    #     print(key + ":%d" % value)
    # file_path = "all_title_1517347_.txt"
    # read_title(file_path)

    # file_path2 = "all_author_1137677_.txt"
    # read_author(file_path2)
    # save_tf_dic()
    # cal_word_tf('<p>')
    #
    # l = [['mathematics', 'and', 'Computers', 'in', 'Simulation', 'Jayaram', 'Bhasker', 'Shi-Xia', 'Liu', 'meng', 'hu', 'The', 'Usability', 'Engineering', 'Life', 'Cycle', '2014']]
    # l2 = [['Mathematics', 'and', 'Computers', 'in', 'Simulation']]
    # t_tf = make_title_tf_feature(l2)
    # a_tf = make_author_tf_feature(l2)
    # j_tf = make_journal_tf_feature(l2)
    # all_tf_list = t_tf + a_tf + j_tf
    # print(all_tf_list)
    # print(t_tf)
    # print(a_tf)
    # print(j_tf)
    # print('====')
    # result = normalize_tf_v2(all_tf_list)
    # print(result)
    # print(nor_t)
    # print(nor_a)
    # print(nor_j)

    # tf_dic = build_tf_dic()
    # print(tf_dic)
    # print(len(tf_dic))
    # # print(tf_dic)
    # content = ['The', 'MOSIX', 'multicomputer', 'operating', 'system', 'for', 'high', 'performance', 'cluster', 'computing', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>', '<p>']
    # build_test_title(file_path)
    # build_test_author(file_path2)
    lower_journals()

