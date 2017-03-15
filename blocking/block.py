import re
import random
import math

def load_all_journals(journal_fp):
    # fp = open('../v3/all_journal_1614_.txt', 'r')
    fp = open(journal_fp, 'r')
    lines = fp.readlines()
    journals_l = []
    for line in lines:
        journals_l.append(line.strip())
    return journals_l


def load_all_v3titles(title_fp):
    # fp = open('../v3/titles4v3.txt', 'r')
    fp = open(title_fp, 'r')
    lines = fp.readlines()
    titles_l = []
    for line in lines:
        titles_l.append(line.strip())
    return titles_l


def load_all_v3authors(author_fp):
    # fp = open('../v3/authors4v3.txt', 'r')
    fp = open(author_fp, 'r')
    lines = fp.readlines()
    authors_l = []
    for line in lines:
        authors_l.append(line.strip())
    return authors_l


def load_year4KB(year_fp):
    # fo = open('../dataset_workshop/year_kb.txt', 'r')
    fo = open(year_fp, 'r')
    lines = fo.readlines()
    years = []
    for line in lines:
        years.append(line.strip())
    return years


def load_volume4KB(volume_fp):
    # fo = open('../dataset_workshop/volume_kb.txt', 'r')
    fo = open(volume_fp, 'r')
    lines = fo.readlines()
    volumes = []
    for line in lines:
        volumes.append(line.strip())
    return volumes


def load_pages4KB(pages_fp):
    # fo = open('../dataset_workshop/temp_page.txt', 'r')
    fo = open(pages_fp, 'r')
    lines = fo.readlines()
    pages = []
    for line in lines:
        pages.append(line.strip())
    return pages


def loadKB2(title_fp, author_fp, journal_fp, year_fp, volume_fp, pages_fp):
    journals = load_all_journals(journal_fp)
    titles = load_all_v3titles(title_fp)
    authors = load_all_v3authors(author_fp)
    years = load_year4KB(year_fp)
    volumes = load_volume4KB(volume_fp)
    pages = load_pages4KB(pages_fp)
    KB = {'Journal': journals, 'Title': titles, 'Author': authors, 'Year': years, 'Volume': volumes, 'Pages': pages}
    return KB


# 判断year,volume,pages跟判断title,author,journal不一样,前三个要一样,后三个包含就可以，
# 这个的策略是只要先找到就break程序，然后label就记为找到它所在ＫＢ的类别．
def isContain2(block, KB):
    label = None
    Find = False
    # print('Y')
    for kb in KB['Year']:
        if block == kb:
            label = 'Year'
            Find = True
            return label
    # print('V')
    for kb in KB['Volume']:
        if block == kb:
            label = 'Volume'
            Find = True
            return label
    # print('P')
    for kb in KB['Pages']:
        if block == kb:
            label = 'Pages'
            Find = True
            return label
    # print('A')
    for kb in KB['Author']:
        if block in kb:
            label = 'Author'
            Find = True
            return label
    # print('T')
    for kb in KB['Title']:
        if block in kb:
            label = 'Title'
            Find = True
            return label
    # print('J')
    for kb in KB['Journal']:
        if block in kb:
            label = 'Journal'
            Find = True
            return label
    return label


#记录block出现在所有类中label，以及它在各类中的次数
def isContain3(block, KB):
    labels = {}
    for kb in KB['Year']:
        if block == kb:
            label = 'Year'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Volume']:
        if block == kb:
            label = 'Volume'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Pages']:
        if block == kb:
            label = 'Pages'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Author']:
        if block in kb:
            label = 'Author'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Title']:
        if block in kb:
            label = 'Title'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    for kb in KB['Journal']:
        if block in kb:
            label = 'Journal'
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
    return labels


# 确定一个block的label,输入是一个block中每个word在KB中确定的label,如:block:'Wei-Hsi Hung', label_list:['Author', 'Author']
# 策略:label_list的最后一个label值记为这个block的label
def confirm_label(label_list):
    label = ''
    if label_list:
        for l in label_list:
            if l:
                label = l
    else:
        label = 'Unknow'
    return label


'''
1.最后一项的key
2.如果最后一项有多个key,value最大的key为label
3.year,volume单独的策略,只要在year_kb,volume_kb中出现就确定为year,volume
'''
def confirm_label2(label_list):
    author_confidence = 2
    title_confidence = 2
    journal_confidence = 2
    page_confidence = 2
    year_confidence = 2
    page_confidence = 2
    label = ''
    if label_list:
        for l in label_list:
            if l:
                label = l
                print(l)
                confidences = calculate_conficence(l)
                print(confidences)
        # print(label)
    else:
        label = 'Unknown'
    return label


def calculate_conficence(label_list):
    confidences = {}
    for label_dict in label_list:
        print(label_dict)
        for key in label_dict.keys():
            if key+'_confidence' in confidences:
                confidences[key+'_confidence'] += label_dict[key]
            else:
                confidences[key+'_confidence'] = label_dict[key]
    return confidences


#属于每个key(label)的block放入到一个list中
def add2dict(my_dict, key, value):
    value = ' '.join(value)
    if key in my_dict.keys():
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
    return my_dict


# 以block为key,label为值
def add2dict2(my_dict, key, value):
    print('2')


def print_my_dict(d):
    for key in d.keys():
        for value in d[key]:
            print(key+': '+value)
            # print(key + ':'+' '.join(value))


def dict2list(d):
    label_list = []
    block_list = []
    for key in d.keys():
        for value in d[key]:
            # print(key+': '+value)
            label_list.append(key)
            block_list.append(value)
    return label_list, block_list


#
def doBlock2(sample, KB):
    r1 = '\,+'
    # r = '|'.join([r1, r2])
    sample = re.sub(r1, ' ', sample).split()   # convert string to word list
    # print(sample)
    label = ''
    labels = []
    all_blocks = {}
    i = 0
    while i < len(sample):
        block = sample[i:i + 1]
        indice = 0
        for j in range(i+1, len(sample) + 1):
            temp = sample[i:j]
            # print('temp:', temp)

            label =isContain2(' '.join(temp), KB)
            if label is not None:
                labels.append(label)
            # print('label:', label)
            if label and j < len(sample):
                block = temp
            else:
                indice = j - 1
                break
            if j == len(sample):
                print(temp)
                indice = j
                block = temp
                break

        print("labels:", labels)
        # print('label:', confirm_label(labels))
        # print('block:', block)
        all_blocks =add2dict(all_blocks, confirm_label(labels), block)
        if i == indice:
            i += 1
        else:
            i = indice
        labels = []

    return all_blocks


def doBlock3(sample, KB):
    r1 = '\,+'
    # r = '|'.join([r1, r2])
    sample = re.sub(r1, ' ', sample).split()   # convert string to word list
    # print(sample)
    label = ''
    labels = []
    all_blocks = {}
    result_blocks = []
    result_labels = []
    i = 0
    while i < len(sample):
        block = sample[i:i + 1]
        indice = 0
        for j in range(i+1, len(sample) + 1):
            temp = sample[i:j]
            # print('temp:', temp)

            label =isContain3(' '.join(temp), KB)   # return dict
            if label is not None:
                labels.append(label)
            # print('label:', label)
            if label and j < len(sample):
                block = temp
            else:
                indice = j - 1
                break
            if j == len(sample):
                print(temp)
                indice = j
                block = temp
                break

        print("labels:", labels)
        print('label:', confirm_label(labels))
        print('block:', block)
        result_blocks.append(' '.join(block))
        result_labels.append(confirm_label(labels))

        if i == indice:
            i += 1
        else:
            i = indice
        labels = []

    return result_blocks, result_labels


def selectMinKey(a, i, length):
    k = i
    for j in range(i+1, length):
        if a[k] > a[j]:
            k = j
    return k


def selectSort(record, label, block):
    length = len(block)
    indexes = []
    for i in range(len(block)):
        # print(block[i])
        if block[i] not in record:
            return
        index = record.index(block[i])
        indexes.append(index)

    for i in range(length):
        key = selectMinKey(indexes, i, length)
        if key != 1:
            tmp = indexes[i]
            indexes[i] = indexes[key]
            indexes[key] = tmp

            tmp = block[i]
            block[i] = block[key]
            block[key] = tmp

            tmp = label[i]
            label[i] = label[key]
            label[key] = tmp

if __name__ == '__main__':
    author_fp = '../dataset_workshop/temp_authors_kb.txt'
    title_fp = '../dataset_workshop/temp_titles_kb.txt'
    journal_fp = '../v3/all_journal_1614_.txt'
    year_fp = '../dataset_workshop/year_kb.txt'
    volume_fp = '../dataset_workshop/volume_kb.txt'
    pages_fp = '../dataset_workshop/temp_page_kb.txt'
    p1 = 'Wei-Hsi Hung,Kuanchin Chen,Chieh-Pin Lin,Does the proactive personality mitigate the adverse effect of technostress on productivity in the mobile environment?,Telematics and Informatics,2015,32,143-157'
    p2 = 'Mathematics and Computers in Simulation,Jayaram Bhasker,Shi-Xia Liu,meng hu,The Usability Engineering Life Cycle, 2014'
    p3 = 'Nuno Sepúlveda,Carlos Daniel Paulino,Carlos Penha Gonçalves,Bayesian analysis of allelic penetrance models for complex binary traits.,Computational Statistics & Data Analysis,2009,53,1271-1283'
    KB = loadKB2(author_fp=author_fp, title_fp=title_fp,journal_fp=journal_fp,year_fp=year_fp,volume_fp=volume_fp,pages_fp=pages_fp)
    # print(isContain2('Models', KB))
    # print(isContain3('Models', KB))
    blocks, labels = doBlock3(p3, KB)
    # all_blocks= doBlock2(p1, KB)
    # print(all_blocks)
    # print_my_dict(all_blocks)
    # labels, blocks = dict2list(all_blocks)
    print(labels)
    print(blocks)
    # selectSort(p1, label=labels, block=blocks)
    # print(labels)
    # print(blocks)

    # fo = open('../dataset_workshop/temp_dataset2.txt', 'r')
    # lines = fo.readlines()
    # random.shuffle(lines)
    # for line in lines:
    #     print(line.strip())
    #     all_blocks = doBlock2(line, KB)
    #     labels, blocks = dict2list(all_blocks)
    #     # print(labels)
    #     # print(blocks)
    #     selectSort(line, label=labels, block=blocks)
    #     print(labels)
    #     print(blocks)
    #     print('============================')
    label_list = [{'Title': 32, 'Journal': 29}, {'Journal': 1, 'Title': 1}, {'Journal': 1}, {'Journal': 1}, {'Journal': 1}]
    # confirm_label2(label_list)
    print(calculate_conficence(label_list))


