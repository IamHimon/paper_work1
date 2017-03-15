from second_hand_house.toolbox import *
import pymysql
import jieba
from utils import *
from train_cnn_onehot import *
from sklearn.cross_validation import KFold

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
        sql = "SELECT 标题,房源编号,发布时间, 租金,押付方式,户型, 租凭方式,房屋类型," \
              "装修,面积,朝向,楼层,小区, 位置,配置,联系人,联系方式,公司,店面, URL FROM anjuke WHERE id < 50000"
        cursor2.execute(sql)
        result = cursor2.fetchall()
        for row in result:
            titles.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[0]))))
            houseIDs.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[1]))))
            publish_times.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[2]))))
            rents.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[3]))))
            charge_methods.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[4]))))
            units.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[5]))))
            rental_models.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[6]))))
            house_types.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[7]))))
            decorations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[8]))))
            areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[9]))))
            orientations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[10]))))
            floors.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[11]))))
            residential_areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[12]))))
            locations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[13]))))
            configurations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[14]))))
            contact_persons.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[15]))))
            phone_numbers.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[16]))))
            companies.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[17]))))
            storefronts.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[18]))))
            urls.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[19]))))
            # describes.append(remove_question_mark(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(row[20])))))
finally:
    connection.close()

x_text = titles + houseIDs + publish_times + rents + charge_methods + units + rental_models+house_types+decorations+\
       areas+orientations+floors+residential_areas+locations+configurations+contact_persons+phone_numbers+companies+\
        storefronts+describes+urls
print("Loading data over!")
max_sample_length = max([len(x) for x in x_text])
# max_sample_length = 100
print("max_document_length:", max_sample_length)

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
y_train, label_dict_size = build_y_train_publication_second_hand_house(titles, houseIDs, publish_times, rents,
            charge_methods, units, rental_models ,house_types ,decorations, areas,orientations,floors,residential_areas,
            locations,configurations,contact_persons,phone_numbers,companies,storefronts,urls)
print(y_train)
print("Preparing y_train over!")

# shuffle here firstly!
print("Shuffle data:")
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
k_fold = KFold(len(s_w_train), n_folds=10)
for train_indices, test_indices in k_fold:
    w_tr, w_te = s_w_train[train_indices], s_w_train[test_indices]
    y_tr, y_te = s_y_train[train_indices], s_y_train[test_indices]
    train.cnn_train_onehot(w_tr, w_te, y_tr, y_te)

