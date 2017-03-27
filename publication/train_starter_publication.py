from tensorflow.contrib import learn
from publication.tools import *
from train_cnn_onehot import *
from sklearn.cross_validation import KFold

whether_word2vec = False
whether_tf = False

# # load word2vec array
# print("Loading word2vec:")
# path = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
# vocab, embedding = load_from_binary(path)
# vocab_size, embedding_dim = embedding.shape
# print("embedding_dim:", embedding_dim)
# print("shape:", embedding.shape)
# print("Loading succeed!")

print("Reading sample data:")
journals = load_all_journals('../v3/all_journal_1614_.txt')
titles = load_all_v3titles('../v3/titles4v3.txt')
authors = load_all_v3authors('../dataset_workshop/linked_authors_no_punctuation.txt')
years = load_years('../dataset_workshop/temp_year.txt')
volumes = load_volume('../dataset_workshop/temp_artificial_volumes.txt')
pages = load_pages('../dataset_workshop/temp_page.txt_kb')


x_text = titles + authors + journals + years + volumes + pages
max_sample_length = max([len(x) for x in x_text])
print("max_document_length:", max_sample_length)


# print("build vocab:")
print('reload vocab:')
vocab = load_dict('publication_complete_dict.pickle')
vocab_size = len(vocab)

print("Preparing w_train:")
w_train_raw = map_word2index(x_text, vocab)
w_train = np.array(makePaddedList2(max_sample_length, w_train_raw, 0))
# print(w_train)
print("w_train shape:", w_train.shape)
# print(w_train[0])
print("preparing w_train over!")

print(w_train.shape)
print(w_train[0])
print(w_train[1])


y_train, label_dict_size = build_y_train_publication_all_attribute(titles, authors, journals, years, volumes, pages)

# shuffle here firstly!
data_size = len(w_train)
shuffle_indices = np.random.permutation(np.arange(data_size))
s_w_train = w_train[shuffle_indices]
s_y_train = y_train[shuffle_indices]

print(s_w_train.shape)
print(s_y_train.shape)
print('label_dict_size:', label_dict_size)


embedding_dim = 100
# ===================================
print("Start to train:")
print("Initial TrainCNN: ")
train = TrainCNN_ONEHOT(
                 vocab_size=vocab_size,
                 embedding_dim=embedding_dim,   # 词向量维度,或者embedding的维度
                 sequence_length=max_sample_length,     # padding之后的句子长度
                 num_classes=label_dict_size,
                 )
# Split train/test set, use 10_fold cross_validation
print("k_fold train:")
k_fold = KFold(len(s_w_train), n_folds=5)
for train_indices, test_indices in k_fold:
    w_tr, w_te = s_w_train[train_indices], s_w_train[test_indices]
    y_tr, y_te = s_y_train[train_indices], s_y_train[test_indices]
    train.cnn_train_onehot(w_tr, w_te, y_tr, y_te)
