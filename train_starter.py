from utils import *
from train_cnn import *
from sklearn.cross_validation import KFold
from load_word2vec import *
from v3.v3_utils import *
from tensorflow.contrib import learn

whether_word2vec = True
whether_tf = False

# load word2vec array
print("Loading word2vec:")
path = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
vocab, embedding = load_from_binary(path)
vocab_size, embedding_dim = embedding.shape
print("embedding_dim:", embedding_dim)
print("shape:", embedding.shape)
print("Loading succeed!")

print("Reading sample data:")
journals = load_all_journals()
titles = load_all_v3titles()
authors = load_all_v3authors()
x_text = titles + authors + journals
max_sample_length = max([len(x) for x in x_text])
print("max_document_length:", max_sample_length)

# 这里设定为max_sample_length,因为下面函数,if samples are longer,they will be trimmed, if shorter-padding
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sample_length)


if whether_word2vec:
    print('Transforming samples to matrix, preparing data for train:')
    w_train_raw = sample2index_matrix2(x_text, vocab)
    w_train = np.array(makePaddedList_index(max_sample_length, w_train_raw, 1))   # should be np.array() but list
    # w_train = np.array(sample2index_matrix(taj_contents, vocab, max_sample_length))   # should be np.array() but list
    print('w_train shape:', w_train.shape)
    print(w_train[0])
    print(w_train[1])
else:
    print("Building w_train according word2vec vocabulary:")
    # load data
    # f_titles, f_authors, f_journals = load_data_not_word2vec()
    # x_text = f_titles + f_authors + f_journals

    w_train_raw = word2id_vocab2(x_text, vocab)
    w_train = np.array(makePaddedList_index(max_sample_length, w_train_raw, 1))
    # 因为这里直接用原始的sample来构建矩阵,所以还需要padding,"<p> ==> 1"

    print(w_train.shape)
    print(w_train[0])
    print(w_train[1])

    # w_train = np.array(list(vocab_processor.fit_transform(x_text)))
    # print("w_train shape:", w_train.shape)
    # vocab_size = len(vocab_processor.vocabulary_)
    # print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    # embedding_dim = 100
    # embedding = np.full([vocab_size, embedding_dim], 1.0, dtype=float)


if whether_tf:
    print("Making padding:")
    titles_contents = makePaddedList(max_sample_length, titles)
    authors_contents = makePaddedList(max_sample_length, authors)
    journals_contents = makePaddedList(max_sample_length, journals)
    taj_contents = titles_contents+authors_contents+journals_contents
    samples_sum_size = len(taj_contents)

    # 构建词频特征
    print("build term frequency dictionary:")
    tf_dic = build_tf_dic()
    # print(tf_dic)
    tf_dic_size = len(tf_dic) + 1
    # 求各部分特征值
    print("titles' TF:")
    titles_t_tf = make_title_tf_feature(titles_contents)
    titles_a_tf = make_author_tf_feature(titles_contents)
    titles_j_tf = make_journal_tf_feature(titles_contents)
    nor_title_t_tf, nor_title_a_tf, nor_title_j_tf = normalize_tf(titles_t_tf, titles_a_tf, titles_j_tf)
    print("authors' TF:")
    authors_t_tf = make_title_tf_feature(authors_contents)
    authors_a_tf = make_author_tf_feature(authors_contents)
    authors_j_tf = make_journal_tf_feature(authors_contents)
    nor_author_t_tf, nor_author_a_tf, nor_author_j_tf = normalize_tf(authors_t_tf, authors_a_tf, authors_j_tf)
    print("journals' TF:")
    journals_t_tf = make_title_tf_feature(journals_contents)
    journals_a_tf = make_author_tf_feature(journals_contents)
    journals_j_tf = make_author_tf_feature(journals_contents)
    nor_journals_t_tf, nor_journals_a_tf, nor_journals_j_tf = normalize_tf(journals_t_tf, journals_a_tf, journals_j_tf)
    all_nor_t_tf = nor_title_t_tf + nor_author_t_tf + nor_journals_t_tf
    all_nor_a_tf = nor_title_a_tf + nor_author_a_tf + nor_journals_a_tf
    all_nor_j_tf = nor_title_j_tf + nor_author_j_tf + nor_journals_a_tf
    t_tf_train = np.array(mapWordToId(all_nor_t_tf, tf_dic))
    a_tf_train = np.array(mapWordToId(all_nor_a_tf, tf_dic))
    j_tf_train = np.array(mapWordToId(all_nor_j_tf, tf_dic))
    print('title shape:', t_tf_train.shape)
    print('author shape:', a_tf_train.shape)
    print('journal shape:', j_tf_train.shape)
    print("Building TF feature over!")
else:
    # if not use TF feature,initialize all value to zero.
    print('NOT HAVE TF!')
    print("Making padding:")
    titles_contents = makePaddedList(max_sample_length, titles)
    authors_contents = makePaddedList(max_sample_length, authors)
    journals_contents = makePaddedList(max_sample_length, journals)
    taj_contents = titles_contents+authors_contents+journals_contents
    samples_sum_size = len(taj_contents)

    tf_dic_size = 0
    t_tf_train = np.full([samples_sum_size, max_sample_length], 1, dtype=int)
    a_tf_train = np.full([samples_sum_size, max_sample_length], 1, dtype=int)
    j_tf_train = np.full([samples_sum_size, max_sample_length], 1, dtype=int)
    print('title shape:', t_tf_train.shape)
    print('author shape:', a_tf_train.shape)
    print('journal shape:', j_tf_train.shape)
    print("Building TF feature over!")


y_train, label_dict_size = build_y_train_publication(titles_contents, authors_contents, journals_contents)

# shuffle here firstly!
data_size = len(w_train)
shuffle_indices = np.random.permutation(np.arange(data_size))
s_w_train = w_train[shuffle_indices]
s_y_train = y_train[shuffle_indices]
s_t_tf_train = t_tf_train[shuffle_indices]
s_a_tf_train = a_tf_train[shuffle_indices]
s_j_tf_train = j_tf_train[shuffle_indices]
print(s_w_train.shape)
print(s_y_train.shape)
print(s_t_tf_train.shape)
print(s_a_tf_train.shape)
print(s_j_tf_train.shape)

# print("max sample length:", max_sample_length)
# print("vocab_size:", vocab_size)


# ===================================
print("Start to train:")
print("Initial TrainCNN: ")
train = TrainCNN(whether_word2vec=whether_word2vec,
                 whether_tf=whether_tf,
                 vocab_size=vocab_size,
                 embedding_dim=embedding_dim,   # 词向量维度,或者embedding的维度
                 sequence_length=max_sample_length,     # padding之后的句子长度
                 vocab_processor=vocab_processor,
                 num_classes=label_dict_size,
                 tf_dict_size=tf_dic_size
                 )
# Split train/test set, use 10_fold cross_validation
print("k_fold train:")
k_fold = KFold(len(s_w_train), n_folds=5)
for train_indices, test_indices in k_fold:
    w_tr, w_te = s_w_train[train_indices], s_w_train[test_indices]
    t_tf_tr, t_tf_te = s_t_tf_train[train_indices], s_t_tf_train[test_indices]
    a_tf_tr, a_tf_te = s_a_tf_train[train_indices], s_a_tf_train[test_indices]
    j_tf_tr, j_tf_te = s_j_tf_train[train_indices], s_j_tf_train[test_indices]
    y_tr, y_te = s_y_train[train_indices], s_y_train[test_indices]

    train.cnn_train(whether_word2vec, whether_tf, embedding, w_tr, w_te, t_tf_tr, t_tf_te, a_tf_tr, a_tf_te, j_tf_tr, j_tf_te, y_tr, y_te)

