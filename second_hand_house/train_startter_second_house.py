import sys
sys.path.append('..')
from second_hand_house.toolbox import *
import pymysql
import jieba
from utils import *
from train_cnn_onehot import *
from sklearn.cross_validation import KFold
from train_cnn_pos import *

titles = []
houseIDs = []
publish_times = []
rents = []
charge_methods = []
units = []
rental_models = []
house_types = []
decorations = []
areas = []
orientations = []
floors = []
residential_areas = []
locations = []
configurations = []
contact_persons = []
phone_numbers = []
companies = []
storefronts = []
describes = []
urls = []

connection = pymysql.connect(host="localhost", user="root", password="0099", db='mysql', charset='utf8')
try:

    with connection.cursor() as cursor2:
        # sql = "SELECT 标题,房源编号,发布时间, 租金,押付方式,户型, 租凭方式,房屋类型," \
        #       "装修,面积,朝向,楼层,小区, 位置,配置,联系人,联系方式,公司,店面, URL FROM anjuke WHERE id < 10000"
        sql = "SELECT 标题,发布时间, 租金,押付方式,户型," \
                "面积,楼层,配置 FROM anjuke WHERE id < 200"
        cursor2.execute(sql)
        result = cursor2.fetchall()
        for row in result:
            if row[0] != '':
                titles.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[0]))))
            if row[1] != '':
                publish_times.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[1]))))
            if row[2] != '':
                rents.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[2]))))
            if row[3] != '':
                charge_methods.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[3]))))
            if row[4] != '':
                units.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[4]))))
            if row[5] != '':
                areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[5]))))
            if row[6] != '':
                floors.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[6]))))
            if row[7] != '':
                configurations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[7]))))


finally:
    connection.close()

# x_text = titles + houseIDs + publish_times + rents + charge_methods + units + rental_models+house_types+decorations+\
#        areas+orientations+floors+residential_areas+locations+configurations+contact_persons+phone_numbers+companies+\
#         storefronts+describes+urls

# print(titles)

x_text = titles + publish_times + rents + charge_methods + rental_models + units + house_types + areas + floors + configurations + phone_numbers
print(x_text)
print("Loading data over!")
max_sample_length = max([len(x) for x in x_text])
# max_sample_length = 100
print("max_document_length:", max_sample_length)
# print(x_text)


# tag
pos_tag_list = makePosFeatures(x_text)
# print(pos_tag_list)
max_pos_length = max([len(x) for x in pos_tag_list])
# print(max_pos_length)
# pos_tag_list, _ = makePaddedList(pos_tag_list)

# pos_vocab = makeWordList(pos_tag_list)
# save_dict(pos_vocab, 'pos.pickle')
pos_vocab = load_dict('../publication/pos.pickle')
pos_vocab_size = len(pos_vocab)
print('pos_vocab_size:', pos_vocab_size)

p_train_raw = map_word2index(pos_tag_list, pos_vocab)
P_train = np.array(makePaddedList2(max_sample_length, p_train_raw, 0))     # shape(13035, 98)
print(P_train.shape)


# print("build vocab:")
print('reload vocab:')
vocab = load_dict('second_hand_house_complete_dict.pickle')
vocab_size = len(vocab)
# word_dict = makeWordList(x_text)
# print(word_dict)
# save_dict(word_dict, 'second_hand_house_dict.pickle')
# print("build and save vocab over!")

print("Preparing w_train:")
w_train_raw = map_word2index(x_text, vocab)
w_train = np.array(makePaddedList2(max_sample_length, w_train_raw, 0))
# print(w_train)
print("w_train shape:", w_train.shape)
# print(w_train[0])
print("preparing w_train over!")


print("Preparing y_train:")
y_train, label_dict_size = build_y_train_publication_second_hand_house2(titles, publish_times, rents, charge_methods,
                                                                       units, areas, floors, configurations)
print(y_train)
print("Preparing y_train over!")

# shuffle here firstly!
print("Shuffle data:")
data_size = len(w_train)
shuffle_indices = np.random.permutation(np.arange(data_size))
p_w_train = P_train[shuffle_indices]
s_w_train = w_train[shuffle_indices]
s_y_train = y_train[shuffle_indices]
# print(p_w_train.shape)
# print(s_w_train.shape)
# print(s_y_train.shape)
print('label_dict_size:', label_dict_size)


embedding_dim = 50
pos_emb_dim = 20
# ===================================
print("Start to train:")
print("Initial TrainCNN: ")
train = TrainCNN_POS(
                 vocab_size=vocab_size,
                 embedding_dim=embedding_dim,   # 词向量维度,或者embedding的维度
                 pos_vocab_size=pos_vocab_size,
                 pos_emb_dim=pos_emb_dim,
                 sequence_length=max_sample_length,     # padding之后的句子长度
                 num_classes=label_dict_size,
                 )
# Split train/test set, use 10_fold cross_validation
print("k_fold train:")
k_fold = KFold(len(s_w_train), n_folds=5)
for train_indices, test_indices in k_fold:
    w_tr, w_te = s_w_train[train_indices], s_w_train[test_indices]
    p_tr, p_te = p_w_train[train_indices], p_w_train[test_indices]
    y_tr, y_te = s_y_train[train_indices], s_y_train[test_indices]
    train.cnn_train_pos(w_tr, w_te, p_tr, p_te, y_tr, y_te)






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
k_fold = KFold(len(s_w_train), n_folds=10)
for train_indices, test_indices in k_fold:
    w_tr, w_te = s_w_train[train_indices], s_w_train[test_indices]
    y_tr, y_te = s_y_train[train_indices], s_y_train[test_indices]
    train.cnn_train_onehot(w_tr, w_te, y_tr, y_te)
