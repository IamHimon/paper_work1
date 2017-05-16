import sys
sys.path.append('..')
import re
import pymysql
import jieba
import pickle
import numpy as np
from utils import *

class_dict = {0:'titles',1:'houseID', 2:'publish_time', 3:'rent',4:'charge_method',5:'unit', 6:'rental_model',
              7:'house_type', 8:'decoration', 9:'area', 10:'orientation', 11:'floor',12:'residential_area',
              13:'location', 14:'configuration', 15:'contact_person', 16:'phone_number', 17:'company', 18:'storefront',
              19:'describe', 20:'url'}

SECOND_HAND_HOUSE = {'titles': 0, 'publish_time': 1, 'rent': 2, 'charge_method': 3, 'unit': 4, 'area': 5, 'floor': 6, 'configuration': 7}


# save dict
def save_dict(word_dict, name):
    with open(name, 'wb') as handle:
        pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(dict_path):
    with open(dict_path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def remove_black_space(a):
    c = []
    for i in a:
        if i != ' ':
            c.append(i)
    return c


def remove_question_mark(a):
    c = []
    for i in a:
        if i != '?':
            c.append(i)
    return c


# pretreatment every sample,strip black space,disperse number with black space
def sample_pretreatment_disperse_number(sample):
    number = re.findall(r'\d+', sample)
    print(number)
    if number:
        for n in number:
            temp = ' ' + ' '.join([c for c in n]) + ' '
            sample = sample.replace(str(n), temp)    # replace only one string
    else:
        return sample.strip()
    return sample.strip()


def sample_pretreatment_disperse_number2(sample):
    add_length = 0
    for m in re.finditer(r'\d+', sample):
        # print('start:',m.start())
        # print('end:', m.end())
        # print('add_length:', add_length)
        sample = replace_by_position(sample, m.start()+add_length, m.end()+add_length)
        add_length += (m.end() - m.start()) + 1
    #     print(sample)
    # print(sample)
    return sample


def load_data(sample_count):
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
                  "装修,面积,朝向,楼层,小区, 位置,配置,联系人,联系方式,公司,店面, 房源描述,URL FROM anjuke WHERE id < %d" % sample_count
            cursor2.execute(sql)
            result = cursor2.fetchall()
            for row in result:
                titles.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[0]))))
                houseIDs.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[1]))))
                publish_times.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[2]))))
                rents.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[3]))))
                charge_methods.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[4]))))
                units.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[5]))))
                rental_models.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[6]))))
                house_types.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[7]))))
                decorations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[8]))))
                areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[9]))))
                orientations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[10]))))
                floors.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[11]))))
                residential_areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[12]))))
                locations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[13]))))
                configurations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[14]))))
                contact_persons.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[15]))))
                phone_numbers.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[16]))))
                companies.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[17]))))
                storefronts.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[18]))))
                describes.append(remove_question_mark(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[19])))))
                urls.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[20]))))
    finally:
        connection.close()

    x_text = titles + houseIDs + publish_times + rents + charge_methods + units + rental_models+house_types+decorations+\
           areas+orientations+floors+residential_areas+locations+configurations+contact_persons+phone_numbers+companies+\
            storefronts+describes+urls

    max_sample_length = max([len(x) for x in x_text])
    print("max_document_length:", max_sample_length)
    return x_text, max_sample_length


def load_all_data():
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
                  "装修,面积,朝向,楼层,小区, 位置,配置,联系人,联系方式,公司,店面, 房源描述,URL FROM anjuke"
            cursor2.execute(sql)
            result = cursor2.fetchall()
            for row in result:
                titles.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[0]))))
                houseIDs.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[1]))))
                publish_times.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[2]))))
                rents.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[3]))))
                charge_methods.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[4]))))
                units.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[5]))))
                rental_models.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[6]))))
                house_types.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[7]))))
                decorations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[8]))))
                areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[9]))))
                orientations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[10]))))
                floors.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[11]))))
                residential_areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[12]))))
                locations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[13]))))
                configurations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[14]))))
                contact_persons.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[15]))))
                phone_numbers.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[16]))))
                companies.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[17]))))
                storefronts.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[18]))))
                describes.append(remove_question_mark(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[19])))))
                urls.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[20]))))
    finally:
        connection.close()

    x_text = titles + houseIDs + publish_times + rents + charge_methods + units + rental_models+house_types+decorations+\
           areas+orientations+floors+residential_areas+locations+configurations+contact_persons+phone_numbers+companies+\
            storefronts+describes+urls
    return x_text


def map_word2index(x_text, word_dict):
    # print("map word to index:")
    w_train = []
    temp = []
    for x in x_text:
        # print(x)
        for w in x:
            if w in word_dict:
                temp.append(word_dict[w])
            else:
                temp.append(0)
        w_train.append(temp)
        temp = []
    return w_train


def build_y_train_publication_second_hand_house(titles, houseIDs, publish_times, rents, charge_methods, units, rental_models ,house_types ,decorations ,\
           areas,orientations,floors,residential_areas,locations,configurations,contact_persons,phone_numbers,companies,\
            storefronts,urls):
    print("Building label dict:")
    title_labels = [0 for i in range(len(titles))]
    houseIDs_labels = [1 for i in range(len(houseIDs))]
    publish_times_labels = [2 for i in range(len(publish_times))]
    rents_labels = [3 for i in range(len(rents))]
    charge_methods_labels = [4 for i in range(len(charge_methods))]
    units_labels = [5 for i in range(len(units))]
    rental_models_labels = [6 for i in range(len(rental_models))]
    house_types_labels = [7 for i in range(len(house_types))]
    decorations_labels = [8 for i in range(len(decorations))]
    areas_labels = [9 for i in range(len(areas))]
    orientations_labels = [10 for i in range(len(orientations))]
    floors_labels = [11 for i in range(len(floors))]
    residential_areas_labels = [12 for i in range(len(residential_areas))]
    locations_labels = [13 for i in range(len(locations))]
    configurations_labels = [14 for i in range(len(configurations))]
    contact_persons_labels = [15 for i in range(len(contact_persons))]
    phone_numbers_labels = [16 for i in range(len(phone_numbers))]
    companies_labels = [17 for i in range(len(companies))]
    storefronts_labels = [18 for i in range(len(storefronts))]
    urls_lebels = [19 for i in range(len(urls))]

    y_t = title_labels + houseIDs_labels + publish_times_labels + rents_labels + charge_methods_labels + units_labels+\
        rental_models_labels + house_types_labels + decorations_labels + areas_labels + orientations_labels + floors_labels +\
        residential_areas_labels + locations_labels + configurations_labels + contact_persons_labels + phone_numbers_labels + \
        companies_labels + storefronts_labels + urls_lebels

    label_dict_size = 20
    y_train = np.zeros((len(y_t), label_dict_size))
    for i in range(len(y_t)):
        y_train[i][y_t[i]] = 1
    print("Preparing y_train over!")
    return y_train, label_dict_size


def build_y_train_publication_second_hand_house2(titles, publish_times, rents, charge_methods, units, areas, floors, configurations):
    print("Building label dict:")
    title_labels = [0 for i in range(len(titles))]
    publish_times_labels = [1 for i in range(len(publish_times))]
    rents_labels = [2 for i in range(len(rents))]
    charge_methods_labels = [3 for i in range(len(charge_methods))]
    units_labels = [4 for i in range(len(units))]
    areas_labels = [5 for i in range(len(areas))]
    floors_labels = [6 for i in range(len(floors))]
    configurations_labels = [7 for i in range(len(configurations))]

    y_t = title_labels + publish_times_labels + rents_labels + charge_methods_labels + units_labels + areas_labels +\
          floors_labels + configurations_labels
    # label_dict_size = len(SECOND_HAND_HOUSE)
    label_dict_size = 8
    y_train = np.zeros((len(y_t), label_dict_size))
    for i in range(len(y_t)):
        y_train[i][y_t[i]] = 1
    print("Preparing y_train over!")
    return y_train, label_dict_size

# use all samples to build a complete vocab
def build_complete_vocab():
    print('vocab')
    all_samples = load_all_data()
    vocab = makeWordList(all_samples)
    save_dict(vocab, 'second_hand_house_complete_dict.pickle')


def load_data4test(model, sample_count):
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
    urls = []
    describes = []

    titles_labels = []
    houseIDs_labels = []
    publish_times_labels = []
    rents_labels = []
    charge_methods_labels = []
    units_labels = []
    rental_models_labels = []
    house_types_labels = []
    decorations_labels = []
    areas_labels = []
    orientations_labels = []
    floors_labels = []
    residential_areas_labels = []
    locations_labels = []
    configurations_labels = []
    contact_persons_labels = []
    phone_numbers_labels = []
    companies_labels = []
    storefronts_labels = []
    urls_labels = []
    # describes_labels = []

    connection = pymysql.connect(host="localhost", user="root", password="0099", db='mysql', charset='utf8')
    try:

        with connection.cursor() as cursor2:
            sql = "SELECT 标题,房源编号,发布时间, 租金,押付方式,户型, 租凭方式,房屋类型," \
                  "装修,面积,朝向,楼层,小区, 位置,配置,联系人,联系方式,公司,店面,URL, 房源描述 FROM anjuke WHERE id>50000 && id < %d" % (50000 + sample_count)
            cursor2.execute(sql)
            result = cursor2.fetchall()
            for row in result:
                if row[0] != '':
                    titles.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[0]))))
                    titles_labels.append(0)
                if row[1] != '':
                    houseIDs.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[1]))))
                    houseIDs_labels.append(1)
                if row[2] != '':
                    publish_times.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[2]))))
                    publish_times_labels.append(2)
                if row[3] != '':
                    rents.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[3]))))
                    rents_labels.append(3)
                if row[4] != '':
                    charge_methods.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[4]))))
                    charge_methods_labels.append(4)
                if row[5] != '':
                    units.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[5]))))
                    units_labels.append(5)
                if row[6] != '':
                    rental_models.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[6]))))
                    rental_models_labels.append(6)
                if row[7] != '':
                    house_types.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[7]))))
                    house_types_labels.append(7)
                if row[8] != '':
                    decorations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[8]))))
                    decorations_labels.append(8)
                if row[9] != '':
                    areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[9]))))
                    areas_labels.append(9)
                if row[10] != '':
                    orientations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[10]))))
                    orientations_labels.append(10)
                if row[11] != '':
                    floors.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[11]))))
                    floors_labels.append(11)
                if row[12] != '':
                    residential_areas.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[12]))))
                    residential_areas_labels.append(12)
                if row[13] != '':
                    locations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[13]))))
                    locations_labels.append(13)
                if row[14] != '':
                    configurations.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[14]))))
                    configurations_labels.append(14)
                if row[15] != '':
                    contact_persons.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[15]))))
                    contact_persons_labels.append(15)
                if row[16] != '':
                    phone_numbers.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[16]))))
                    phone_numbers_labels.append(16)
                if row[17] != '':
                    companies.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[17]))))
                    companies_labels.append(17)
                if row[18] != '':
                    storefronts.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[18]))))
                    storefronts_labels.append(18)
                if row[19] != '':
                    urls.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[19]))))
                    urls_labels.append(19)
                if row[20] != '':
                    describes.append(remove_question_mark(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(row[19])))))
    finally:
        connection.close()

    if model == 0:
        return titles, titles_labels
    if model == 1:
        return houseIDs, houseIDs_labels
    if model == 2:
        return publish_times, publish_times_labels
    if model == 3:
        return rents, rents_labels
    if model ==4:
        return charge_methods, charge_methods_labels
    if model == 5:
        return units, units_labels
    if model == 6:
        return rental_models, rental_models_labels
    if model == 7:
        return house_types, house_types_labels
    if model == 8:
        return decorations, decorations_labels
    if model == 9:
        return areas, areas_labels
    if model == 10:
        return orientations, orientations_labels
    if model == 11:
        return floors, floors_labels
    if model == 12:
        return residential_areas, residential_areas_labels
    if model == 13:
        return locations, locations_labels
    if model == 14:
        return configurations, configurations_labels
    if model == 15:
        return contact_persons, contact_persons_labels
    if model == 16:
        return phone_numbers, phone_numbers_labels
    if model == 17:
        return companies, companies_labels
    if model == 18:
        return storefronts, storefronts_labels
    if model == 19:
        return urls, urls_labels
    if model == 20:
        return describes
    if model == 21:
        return titles, titles_labels,houseIDs, houseIDs_labels,publish_times,publish_times_labels,rents,rents_labels,charge_methods,charge_methods_labels,\
                units,units_labels,rental_models,rental_models_labels,house_types,house_types_labels,decorations,decorations_labels,areas,areas_labels,\
                orientations,orientations_labels,floors,floors_labels,residential_areas,residential_areas_labels,locations,locations_labels,configurations,\
                configurations_labels,contact_persons,contact_persons_labels,phone_numbers,phone_numbers_labels,companies,companies_labels,storefronts,\
                storefronts_labels,urls,urls_labels


def save_experiment_result_secondhand2(result_path, x_raw, y_test, predictions, Accuracy):
    write = open(result_path, 'w+')
    write.write('Classification Accuracy: ' + str(Accuracy)+'\n')
    for i in range(len(x_raw)):
        if y_test[i] != predictions[i]:
            print("label: ", y_test[i])
            print("prediction: ", predictions[i])
            print(''.join(x_raw[i]) + '\t' + class_dict.get(y_test[i]) + '\t' + class_dict.get(predictions[i]))
            write.write(''.join(x_raw[i])+'\t'+class_dict.get(y_test[i])+'\t'+class_dict.get(predictions[i])+'\n')


# save experiment result,only save samples which was categorized wrongly.
def save_experiment_result_secondhand(result_path, x_raw, y_test, predictions, Accuracy):
    write = open(result_path, 'w+')
    size = len(x_raw)
    l = ''
    p = ''
    write.write('Classification Accuracy: ' + str(Accuracy)+'\n')
    for i in range(size):
        label = str(y_test[i])
        prediction = str(predictions[i])
        if label != prediction:
            if prediction == '0':
                l = 'tittle'
            elif prediction == '1':
                l = 'houseID'
            elif prediction == '2':
                l = 'publish_time'
            elif prediction == '3':
                l = 'rent'
            elif prediction == '4':
                l = 'charge_model'
            elif prediction == '5':
                l = 'unit'
            elif prediction == '6':
                l = 'rental_model'
            elif prediction == '7':
                l = 'house_type'
            elif prediction == '8':
                l = 'decoration'
            elif prediction == '9':
                l = 'area'
            elif prediction == '10':
                l = 'orientation'
            elif prediction == '11':
                l = 'floor'
            elif prediction == '12':
                l = 'residential_area'
            elif prediction == '13':
                l = 'location'
            elif prediction == '14':
                l = 'configuration'
            elif prediction == '15':
                l = 'contact_person'
            elif prediction == '16':
                l = 'phone_number'
            elif prediction == '17':
                l = 'company'
            elif prediction == '18':
                l = 'storefront'
            elif prediction == '19':
                l = 'describe'
            elif prediction == '20':
                l = 'url'

            if prediction == '0':
                p = 'tittle'
            elif prediction == '1':
                p = 'houseID'
            elif prediction == '2':
                p = 'publish_time'
            elif prediction == '3':
                p = 'rent'
            elif prediction == '4':
                p = 'charge_model'
            elif prediction == '5':
                p = 'unit'
            elif prediction == '6':
                p = 'rental_model'
            elif prediction == '7':
                p = 'house_type'
            elif prediction == '8':
                p = 'decoration'
            elif prediction == '9':
                p = 'area'
            elif prediction == '10':
                p = 'orientation'
            elif prediction == '11':
                p = 'floor'
            elif prediction == '12':
                p = 'residential_area'
            elif prediction == '13':
                p = 'location'
            elif prediction == '14':
                p = 'configuration'
            elif prediction == '15':
                p = 'contact_person'
            elif prediction == '16':
                p = 'phone_number'
            elif prediction == '17':
                p = 'company'
            elif prediction == '18':
                p = 'storefront'
            elif prediction == '19':
                p = 'describe'
            elif prediction == '20':
                p = 'url'

            print(''.join(x_raw[i])+' '+l+' '+p)
            write.write(''.join(x_raw[i])+'\t'+l+'\t'+p+'\n')
    write.close()


def replace_by_position(str, start, end):
    seg_str = str[start:end]
    temp = ' ' + ' '.join([c for c in seg_str]) + ' '
    str = str[0:start] + temp + str[end:len(str)]
    return str


if __name__ == '__main__':
    print('main')
    # x_text = [['山水','边上', '房主',  '提供', '诗', '所有','<p>', 'asyeurfijdkfh'],
    #           ['号线', '源', '推'],
    #           ['一直', '小户', '最合适', 'belongs']]
    # word_dict = load_dict('second_hand_house_dict.pickle')
    # w_train = map_word2index(x_text, word_dict)
    # print(w_train)
    urls, urls_labels = load_data4test(1, 5)
    print(''.join(urls[0]))

    # s = '急租 盘蠡新村 精装2室 轻轨口 家电齐全 拎包入住 41212770 2015年03月26日 1700元/月 付3押1 2室2厅1卫'
    # s2 = 'hello12345hello456hello'
    # s3 = 'Discovering the Most Influential Sites over Uncertain Data: A Rank Based Approach, K. Zheng, Z. Huang, A. Zhou and X. Zhou, IEEE Transactions on Knowledge and Data Engineering, 24(12),2156-2169, 2012'
    # add_length = 0
    # for m in re.finditer(r'\d+', s):
    #     print('start:',m.start())
    #     print('end:', m.end())
    #     print('add_length:', add_length)
    #     s = replace_by_position(s, m.start()+add_length, m.end()+add_length)
    #     add_length += (m.end() - m.start()) + 1
    #     print(s)
    # print(s)
    # print(sample_pretreatment_disperse_number(s3))
    # print(sample_pretreatment_disperse_number2(s3))
    # seg_s = jieba.lcut(sample_pretreatment_disperse_number(s))
    # print(seg_s)
    # number = re.findall(r'\d+', s)
    # print(number)
    # num = [m.start() for m in re.finditer(r'\d+', s)]
    # print(num)
    # for m in re.finditer(r'\d+', s):
    #     print("start:", m.start())
    #     print("end:", m.end())
    #     print("group:", m.group())
    #     seg_str = s[m.start():m.end()]
    #     print(seg_str)




