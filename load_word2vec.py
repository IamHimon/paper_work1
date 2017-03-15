import numpy as np
from numpy import linalg as LA
import time

def unitvec(vec):
    return (1.0 / LA.norm(vec, ord=2)) * vec


def load_from_binary(fname, vocabUnicodeSize=78, desired_vocab=None, encoding="utf-8"):
    """
    fname : path to file
    vocabUnicodeSize: the maximum string length (78, by default)
    desired_vocab: if set, this will ignore any word and vector that
                   doesn't fall inside desired_vocab.
    Returns

    """
    if fname.endswith('.bin'):
        with open(fname, 'rb') as fin:
            header = fin.readline()
            vocab_size, vector_size = list(map(int, header.split()))

            vocab = np.empty(vocab_size, dtype='<U%s' % vocabUnicodeSize)
            vectors = np.empty((vocab_size, vector_size), dtype=np.float)
            binary_len = np.dtype(np.float32).itemsize * vector_size
            for i in range(vocab_size):
                # read word
                word = b''
                while True:
                    ch = fin.read(1)
                    if ch == b' ':
                        break
                    word += ch
                include = desired_vocab is None or word in desired_vocab
                if include:
                    vocab[i] = word.decode(encoding)

                # read vector
                vector = np.fromstring(fin.read(binary_len), dtype=np.float32)
                if include:
                    vectors[i] = unitvec(vector)
                fin.read(1)  # newline

            if desired_vocab is not None:
                vectors = vectors[vocab != '', :]
                vocab = vocab[vocab != '']
    else:
        print("please feed a .bin word2vec model!")

    return vocab, vectors


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


# 错误的优化方法,比如[1, 5, 3915]
def word2id_vocab(sent_contents, vocab):
    T = []
    for sent in sent_contents:
        t = []
        i = 0
        for w in sent:
            index = map_vocab(w, vocab)
            t.append(index)
            if index == 1:
                break
            i += 1
        for j in range(i, len(sent)-1):
            t.append(1)
        T.append(t)
    return T


def word2id_vocab2(sent_contents, vocab):
    T = []
    for sent in sent_contents:
        t = []
        i = 0
        for w in sent:
            index = map_vocab(w, vocab)
            t.append(index)
        T.append(t)
    return T


def map_vocab(word, vocab):
    index = np.where(vocab == word)
    if index[0]:
        return index[0][0]
    else:
        return 1

if __name__ == '__main__':
    print("main")
    path = '/home/himon/PycharmProjects/paper_work1/word2vec/dblp.model.syn0.npy'
    path2 = "/home/himon/Jobs/nlps/word2vec/vectors.bin"
    path3 = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
    a, b = load_from_binary(path3)
    # print(a.shape)
    # print(b.shape)
    # c, d = b.shape
    # print(c)
    # print(d)
    # print(a.take(1))
    # index = np.where(a == 'relation')
    # print(index)
    # if index[0]:
    #     print(index[0][0])
    # else:
    #     print(1)
    #
    # print(map_vocab('relation', vocab=a))
    start = time.clock()
    sample2 = [['On', 'a', 'Gauntlet', 'Thrown', 'by', 'David', 'Gries', '<p>', '<p>', '<p>', '<p>'],
       ['Foundations', 'and', 'Trends', 'in', 'Programming', 'Languages', '<p>', '<p>', '<p>', '<p>'],
       ['124', '-', '125', 'ksfdoikjlk', '<p>', '<p>', '<p>', '<p>', '<p>'],
       ['124', '-', '125', 'ksfdoikjlk', '<p>', '<p>', '<p>', '<p>', '<p>'],
       ['124', '-', '125', 'ksfdoikjlk', '<p>', '<p>', '<p>', '<p>', '<p>'],
       ['124', '-', '125', 'ksfdoikjlk', '<p>', '<p>', '<p>', '<p>', '<p>'],
       ['124', '-', '125', 'ksfdoikjlk', '<p>', '<p>', '<p>', '<p>', '<p>'],
               ['Briefings', 'in', 'Bioinformatics']]
    x = word2id_vocab(sample2, vocab=a)
    print(x)
    end = time.clock()
    print('time consuming:', end - start)

    x2 = word2id_vocab2(sample2, vocab=a)
    print(x2)
    en2 = time.clock()
    print("time consuming:", en2 - end)





