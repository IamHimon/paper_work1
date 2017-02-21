# resize samples to the same length sentence


def find_max_length():
    fp = open('corpus4word2vec.txt', 'r')
    lines = fp.readlines()
    max_length = 0
    for line in lines:
        length = len(line.split())
        max_leng = max(length, max_length)
    print(max_length)  # 111
    return max_length


def resize_all_samples(max_length):
    fp = open('set_all_title_author_journal.txt', 'r')
    output = open('resized_corpus4word2vec.txt', 'w+')
    lines = fp.readlines()
    for line in lines:
        resize_line = line.strip() + make_pad_string(max_length - len(line.split()))
        # print(resize_line)
        output.write(resize_line+'\n')
    fp.close()
    output.close()


def readData(f_path):
    titles_set = set()
    authors_set = set()
    journals_set = set()
    fp = open(f_path, 'r')
    samples = fp.readlines()
    for sample in samples:
        temp = sample.strip().split('#$')
        if len(temp) == 3:
            title = temp[0]
            authors = temp[1]
            journal = temp[2]
            # title, authors, journal = sample.strip().split('#$')
            # build titles set
            titles_set.add(title.strip('.'))
            # build author set
            author = authors.split(',')
            for a in author:
                authors_set.add(a)
            # build journal set
            journals_set.add(journal.strip('.'))
    return titles_set, authors_set, journals_set


def build_set_taj():
    f_path = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/all_title_author_journal.txt'
    f_path2 = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/temp_title_author_journal.txt'
    t_set, a_set, j_set = readData(f_path)
    t_length = len(t_set)
    a_length = len(a_set)
    j_length = len(j_set)
    t_output = open('all_title_'+str(t_length)+'_.txt', 'w+')
    a_output = open('all_author_'+str(a_length)+'_.txt', 'w+')
    j_output = open('all_journal_'+str(j_length)+'_.txt', 'w+')
    print("Build set end, start write titles:")
    for t in t_set:
        t_output.write(t+'\n')
    t_output.close()

    print("Write titles end,start write authors:")
    for a in a_set:
        a_output.write(a+'\n')
    a_output.close()

    print("Write authors end, start write journals:")
    for j in j_set:
        j_output.write(j+'\n')
    j_output.close()
    print("Write journals end!")


# ['a','b','c'] ==> ['a','b','c',,....'<p>','<p>','<p>','<p>','<p>'],其中填充后的list长度是pad_length.
def make_pad_string(pad_length):
    str = ''
    for i in range(pad_length):
        str += ' <p>'
    return str

if __name__ == '__main__':
    # line = "meng hu"
    # resize_line = line + make_pad_string(100 - len(line.split()))
    # print(resize_line)
    # title = 'Decomposition of Graphs and Monotone Formula Size of Homogeneous Functions.'
    # print(title.strip('.'))
    # resize_all_samples(111)
    build_set_taj()



