import xml.dom.minidom
import json
from pandas import Series, DataFrame
import pandas as pd
import re
from blocking.reconstruction import *
from blocking.block import *
from SHH_testdata.generate_dataset import *
import matplotlib.pyplot as plt


LABEL_DICT = {'Title': 0, 'Author': 1, 'Journal': 2, 'Year': 3, 'Volume': 4, 'Pages': 5}
SECOND_HAND_HOUSE = {'titles': 0, 'publish_time': 1, 'rent': 2, 'charge_method': 3, 'unit': 4, 'area': 5, 'floor': 6, 'configuration': 7}

publication_keys = ['0', '1', '2', '3', '4', '5']
shh_keys = ['0', '1', '2', '3', '4', '5', '6', '7']
uc_keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8']


def draw(pic_name, x, y, path):
    # pic_name = 'pub_complete_apt'

    plt.figure(figsize=(8, 5)) #创建绘图对象
    plt.plot(x, y, "b--", linewidth=4)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度
    plt.xlabel("Theta") #X轴标签
    plt.ylabel("ES")  #Y轴标签
    plt.title(pic_name) #图标题
    # plt.savefig(path) #保存图
    plt.show()  #显示图
    plt.close()



def blocks_map_predictions(blocks, predictions):
    bp_dict = {}
    for i in range(len(predictions)):
        bp_dict[predictions[i]] = blocks[i]
    return bp_dict


def sorted_dict(bp_dict, keys):
    result = {}
    for i in keys:
        result[i] = bp_dict.get(i)
    return result


def calculate_rate(result, template, lable):
    # print('++++++++++++++++++++++++++++++++++++++++++++')
    sum_count = result.dropna().size
    template_count = template.size
    r1 = '\,+'
    r2 = '\s+'
    r = '|'.join([r1, r2])
    # sum_count = result.size
    # print('sum_count', sum_count)
    error = 0
    error_index = set()
    for i in result.index:
        # print(result.get(i))
        if pd.notnull(result.get(i)) and pd.notnull(template.get(i)):
            sub_result = re.sub(r, '', result.get(i))
            temp = re.sub(r, '', template.get(i).lower())
        else:
            sub_result = result.get(i)
            temp = template.get(i).lower()
        if sub_result != temp:
            # print(sub_result)
            # print(temp)
            # print('-----')
            error_index.add(i)
            error += 1
    # print(lable + ' error count: ', error)
    # print(lable + ' accuracy: %.5f', float((sum_count - error)/sum_count))
    # result_write.write(lable + ' accuracy: ' + str(float((sum_count - error)/sum_count)) + '\n')
    # accuracy = float((sum_count - error)/sum_count)
    # print(error_index)

    P = float((sum_count - error)/sum_count)
    R = float((template_count - error)/template_count)
    F1 = float((2 * P * R) / (P + R))
    print(lable + ' error count: ', error)
    print('sum_acount: ', sum_count)
    print('template_count:', template_count)
    # print(lable + ' accuracy: %.5f', float((sum_count - error)/sum_count))
    # print('recal: %.5f', float((template_count - error)/template_count))
    # accuracy = float((sum_count - error)/sum_count)

    print('Precision: %.3f', P)
    print('Recall: %.3f', R)
    print('F1-measure: %.3f', F1)
    return error_index, F1


# Second hand house
def calculate_rate2(result, template, lable):
    sum_count = result.size
    error = 0
    error_index = set()
    for i in result.index:
        if result.get(i) != template.get(i):
            error_index.add(i)
            error += 1
    print(lable + ' error count: ', error)
    print('sum_acount: ', sum_count)
    print(lable + ' accuracy: %.5f', float((sum_count - error)/sum_count))
    accuracy = float((sum_count - error)/sum_count)


    return error_index, accuracy


# Second hand house F
def calculate_rate3(result, template, lable):
    # print(result[:5])
    # print(template[:5])
    sum_count = result.dropna().size
    template_count = template.size
    error = 0
    error_index = set()
    for i in result.index:
        if result.get(i) != template.get(i):
            # print(result.get(i))
            # print(template.get(i))
            # print('---------')
            error_index.add(i)
            error += 1

    P = float((sum_count - error)/sum_count)
    R = float((template_count - error)/template_count)
    F1 = float((2 * P * R) / (P + R))
    print(lable + ' error count: ', error)
    print('sum_acount: ', sum_count)
    print('template_count:', template_count)
    # print(lable + ' accuracy: %.5f', float((sum_count - error)/sum_count))
    # print('recal: %.5f', float((template_count - error)/template_count))
    # accuracy = float((sum_count - error)/sum_count)

    print('Precision: %.3f', P)
    print('Recall: %.3f', R)
    print('F1-measure: %.3f', F1)
    return error_index, F1


# Second hand house
def calculate_rate4(result, template, lable, result_write):
    sum_count = result.size
    error = 0
    error_index = set()
    for i in result.index:
        if result.get(i) != template.get(i):
            # print(result.get(i))
            # print(template.get(i))
            # print('-----')
            error_index.add(i)
            error += 1
    print(lable + ' error count: ', error)
    print(lable + ' accuracy: %.5f', float((sum_count - error)/sum_count))
    result_write.write(lable + ' accuracy: ' + str(float((sum_count - error)/sum_count)) + '\n')
    # print(error_index)
    return error_index


# parse the .xml
def publication_parse_xml(filename, keys):
    df = pd.DataFrame(columns=keys)
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    articles = collection.getElementsByTagName("Article")
    labels = ['0', '1', '2', '3', '4', '5']
    for article in articles:
        title = article.getElementsByTagName('title')[0].childNodes[0].data
        # print(title.childNodes[0].data)
        author = article.getElementsByTagName('author')[0].childNodes[0].data
        # print(author.childNodes[0].data)
        journal = article.getElementsByTagName('journal')[0].childNodes[0].data
        # print(journal.childNodes[0].data)
        year = article.getElementsByTagName('year')[0].childNodes[0].data
        # print(year.childNodes[0].data)
        page = article.getElementsByTagName('page')[0].childNodes[0].data
        # print(page.childNodes[0].data)
        volume = article.getElementsByTagName('volume')[0].childNodes[0].data
        # print(volume.childNodes[0].data)
        record_ID = article.getElementsByTagName('record_ID')[0].childNodes[0].data
        # print(record_ID.childNodes[0].data)
        # print('--------------------------')
        blocks = [title, author, journal, year, volume, page]
        bp_dict = blocks_map_predictions(blocks, labels)
        blocks = Series(bp_dict)
        # print(blocks)
        df.loc[record_ID] = blocks
    return df


# parse the .xml
def shh_parse_xml(filename, keys):
    df = pd.DataFrame(columns=keys)
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    articles = collection.getElementsByTagName("House")
    labels = ['0', '1', '2', '3', '4', '5', '6', '7']
    for article in articles:
        title = article.getElementsByTagName('title')[0].childNodes[0].data
        publish_t = article.getElementsByTagName('publish_t')[0].childNodes[0].data
        rent = article.getElementsByTagName('rent')[0].childNodes[0].data
        charge_m = article.getElementsByTagName('charge_m')[0].childNodes[0].data
        unit = article.getElementsByTagName('unit')[0].childNodes[0].data
        area = article.getElementsByTagName('area')[0].childNodes[0].data
        floor = article.getElementsByTagName('floor')[0].childNodes[0].data
        conf = article.getElementsByTagName('conf')[0].childNodes[0].data
        ID = article.getElementsByTagName('ID')[0].childNodes[0].data
        # print(record_ID.childNodes[0].data)
        # print('--------------------------')
        blocks = [title, publish_t, rent, charge_m, unit, area, floor, conf]
        bp_dict = blocks_map_predictions(blocks, labels)
        blocks = Series(bp_dict)
        # print(blocks)
        df.loc[ID] = blocks
    return df[:200]

# ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']
# parse the .xml
def uc_parse_xml(filename, keys):
    df = pd.DataFrame(columns=keys)
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    articles = collection.getElementsByTagName("Car")
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    for article in articles:
        Brand = article.getElementsByTagName('Brand')[0].childNodes[0].data
        Price = article.getElementsByTagName('Price')[0].childNodes[0].data
        Vehicle = article.getElementsByTagName('Vehicle')[0].childNodes[0].data
        Odometer = article.getElementsByTagName('Odometer')[0].childNodes[0].data
        Colour = article.getElementsByTagName('Colour')[0].childNodes[0].data
        Transmission = article.getElementsByTagName('Transmission')[0].childNodes[0].data
        Body = article.getElementsByTagName('Body')[0].childNodes[0].data
        Engine = article.getElementsByTagName('Engine')[0].childNodes[0].data
        Fuel_Enconomy = article.getElementsByTagName('Fuel_Enconomy')[0].childNodes[0].data
        ID = article.getElementsByTagName('ID')[0].childNodes[0].data
        # print(ID + ' : ' + Brand)

        # print('--------------------------')
        blocks = [Brand, Price, Vehicle, Odometer, Colour, Transmission, Body, Engine, Fuel_Enconomy]
        bp_dict = blocks_map_predictions(blocks, labels)
        blocks = Series(bp_dict)
        # print(blocks)
        df.loc[ID] = blocks
    return df


# parse the .json
def parse_json(filename, keys):
    df = pd.DataFrame(columns=keys)
    f = open(filename, "r")
    for line in f:
        decodes = json.loads(line)
        # print(decodes['ID'])
        # print(decodes['blocks'])
        # print(decodes['labels'])
        # print(decodes['predictions'])
        bp_dict = blocks_map_predictions(decodes['blocks'], decodes['predictions'])
        # print(bp_dict)
        blocks = Series(bp_dict)
        df.loc[decodes['ID']] = blocks
        # print(blocks)
    f.close()
    return df


# parse the .json
def uc_parse_json(filename, keys):
    df = pd.DataFrame(columns=keys)
    f = open(filename, "r")
    for line in f:
        decodes = json.loads(line)
        # print(decodes['ID'])
        # print(decodes['blocks'])
        # print(decodes['labels'])
        # print(decodes['predictions'])
        bp_dict = blocks_map_predictions(decodes['blocks'], decodes['predictions'])
        # print(bp_dict)
        blocks = Series(bp_dict)
        df.loc[decodes['ID']] = blocks
        # print(blocks)
    f.close()

    # uc:
    df=df.rename(columns={'2': '1', '1': '2'})

    return df

def draw2(pic_name, x, y):
    pic_name = 'shh_complete_apt2'

    plt.figure(figsize=(8, 5)) #创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度
    plt.xlabel("Threshold") #X轴标签
    plt.ylabel("Accuracy")  #Y轴标签
    plt.title(pic_name) #图标题
    plt.savefig("1000_shh_pic/" + pic_name + ".pdf") #保存图
    # plt.show()  #显示图
    plt.close()


def pub_one_result(result_file_json, df_template, result_write):

    df_result = parse_json(result_file_json, publication_keys)

    # title
    error_index0, title_acc = calculate_rate(df_result['0'], df_template['0'], 'title')
    result_write.write('title accuracy:' + str(title_acc) + '\n')
    # author
    error_index1, author_acc = calculate_rate(df_result['1'], df_template['1'], '')
    result_write.write('author accuracy:' + str(author_acc) + '\n')
    # journal
    error_index2, journal_acc = calculate_rate(df_result['2'], df_template['2'], '')
    result_write.write('journal accuracy:' + str(journal_acc) + '\n')
    # year
    error_index3, year_acc = calculate_rate(df_result['3'], df_template['3'], '')
    result_write.write('year accuracy:' + str(year_acc) + '\n')
    # volume
    error_index4, volume_acc = calculate_rate(df_result['4'], df_template['4'], '')
    result_write.write('Volume accuracy:' + str(volume_acc) + '\n')
    # page:
    error_index5, page_acc = calculate_rate(df_result['5'], df_template['5'], '')
    result_write.write('pages accuracy:' + str(page_acc) + '\n')

    errors = error_index0 | error_index1 | error_index2 | error_index3 | error_index4 | error_index5
    record_count = df_result.shape[0]
    complete_acc = float((record_count - len(errors))/record_count)
    print('Complete accuracy:', complete_acc)
    result_write.write('Complete accuracy: ' + str(complete_acc) + '\n')

    accuracy = [title_acc, author_acc, journal_acc, year_acc, volume_acc, page_acc, complete_acc]
    return accuracy


def shh_one_result(result_file_json, df_template, result_write):

    df_result = parse_json(result_file_json, shh_keys)[:200]

    print(df_template)
    print(df_result)

    title, title_acc = calculate_rate3(df_result['0'], df_template['0'], 'title')
    publish_time, publish_time_acc = calculate_rate3(df_result['1'], df_template['1'], 'publish_time')
    rent, rent_acc = calculate_rate3(df_result['2'], df_template['2'], 'rent')
    charge_method, charge_method_acc = calculate_rate3(df_result['3'], df_template['3'], 'charge_method')
    unit, unit_acc = calculate_rate3(df_result['4'], df_template['4'], 'unit')
    area, area_acc = calculate_rate3(df_result['5'], df_template['5'], 'area')
    floor, floor_acc = calculate_rate3(df_result['6'], df_template['6'], 'floor')
    configuration, configuration_acc = calculate_rate3(df_result['7'], df_template['7'], 'configuration')

    result_write.write('title accuracy:' + str(title_acc) + '\n')
    result_write.write('publish_time_acc accuracy:' + str(publish_time_acc) + '\n')
    result_write.write('rent_acc accuracy:' + str(rent_acc) + '\n')
    result_write.write('charge_method_acc accuracy:' + str(charge_method_acc) + '\n')
    result_write.write('unit_acc accuracy:' + str(unit_acc) + '\n')
    result_write.write('area_acc accuracy:' + str(area_acc) + '\n')
    result_write.write('floor_acc accuracy:' + str(floor_acc) + '\n')
    result_write.write('configuration_acc accuracy:' + str(configuration_acc) + '\n')

    errors = title | publish_time | rent | charge_method | unit | area | floor | configuration
    print('', errors)
    print(len(errors))
    print('df size: ', df_result.shape[0])
    complete_acc = float((df_result.shape[0] - len(errors))/df_result.shape[0])
    print('Complete accuracy: %.5f', complete_acc)
    result_write.write('Complete accuracy: ' + str(complete_acc) + '\n')
    f_accuracy = [title_acc, publish_time_acc, rent_acc, charge_method_acc, unit_acc, area_acc, floor_acc, configuration_acc, complete_acc]
    return f_accuracy


def uc_one_result(result_file_json, df_template, result_write):

    df_result = uc_parse_json(result_file_json, uc_keys)
    # df_result.to_csv('df_result.csv')
    # df_template.to_csv('df_template.csv')
    # print(len(df_result.index))
    # print(len(df_template.index))
    # can = pd.concat([df_template, df_result], axis=1)
    # print(can)
    # print(df_template[:1])

    Brand, Brand_acc = calculate_rate3(df_result['0'], df_template['0'], 'Brand')
    Price, Price_acc = calculate_rate3(df_result['1'], df_template['1'], 'Price')
    Vehicle, Vehicle_acc = calculate_rate3(df_result['2'], df_template['2'], 'Vehicle')
    Odometer, Odometer_acc = calculate_rate3(df_result['3'], df_template['3'], 'Odometer')
    Colour, Colour_acc = calculate_rate3(df_result['4'], df_template['4'], 'Colour')
    Transmission, Transmission_acc = calculate_rate3(df_result['5'], df_template['5'], 'Transmission')
    Body, Body_acc = calculate_rate3(df_result['6'], df_template['6'], 'Body')
    Engine, Engine_acc = calculate_rate3(df_result['7'], df_template['7'], 'Engine')
    fc, fc_acc = calculate_rate3(df_result['8'], df_template['8'], 'Fuel Enconomy')

    result_write.write('Brand_acc accuracy:' + str(Brand_acc) + '\n')
    result_write.write('Price_acc accuracy:' + str(Price_acc) + '\n')
    result_write.write('Vehicle_acc accuracy:' + str(Vehicle_acc) + '\n')
    result_write.write('Odometer_acc accuracy:' + str(Odometer_acc) + '\n')
    result_write.write('Colour_acc accuracy:' + str(Colour_acc) + '\n')
    result_write.write('Transmission_acc accuracy:' + str(Transmission_acc) + '\n')
    result_write.write('Body_acc accuracy:' + str(Body_acc) + '\n')
    result_write.write('Engine_acc accuracy:' + str(Engine_acc) + '\n')
    result_write.write('fc accuracy:' + str(fc_acc) + '\n')

    errors = Brand | Price | Vehicle | Odometer | Colour | Transmission | Body | Engine | fc
    # print('', errors)
    # print(len(errors))
    # print('df size: ', df_result.shape[0])
    complete_acc = float((df_result.shape[0] - len(errors))/df_result.shape[0])
    print('Complete accuracy: %.5f', complete_acc)
    result_write.write('Complete accuracy: ' + str(complete_acc) + '\n')
    f_accuracy = [Brand_acc, Price_acc, Vehicle_acc, Odometer_acc, Colour_acc, Transmission_acc, Body_acc, Engine_acc, fc_acc, complete_acc]
    return f_accuracy


def publication_done():
    data_file_xml = '../testdata/test_data.xml'
    df_template = publication_parse_xml(data_file_xml, publication_keys)
    # result_file_json = 'result_0.96.json'
    # result_file_json = 'result_0.92.json'
    # result_file_json = 'result_0.94.json'
    # result_file_json = 'result_0.95.json'
    # result_file_json = 'result_0.9.json'
    # result_file_json = 'result_0.98.json'
    # result_file_json = 'result_1.json'

    result_file_json_list = ['result_0.85.json', 'result_0.9.json',  'result_0.92.json', 'result_0.94.json', 'result_0.95.json',
                             'result_0.96.json', 'result_0.98.json', 'result_1.json']

    result_output = open('f_pub_pic/pub_result.txt', 'w+')
    accuracy_list = []
    for result_file in result_file_json_list:
        result_output.write('[' + result_file + ']' + ':' + '\n')
        temp = pub_one_result(result_file, df_template, result_output)
        accuracy_list.append(np.array(temp))
        result_output.write('===================' + '\n')
        result_output.write('\n')

    # print(accuracy_list)
    accuracy = np.array(accuracy_list).transpose()
    # print(accuracy.shape)
    x = [0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 1]
    attr = ['title', 'author', 'journal', 'year', 'volume', 'pages', 'whole_instance']
    # for i in range(len(attr)):
    #     print(accuracy[i])
    #     draw(attr[i], x, accuracy[i], "f_pub_pic/" + str(attr[i]) + ".pdf")
    #
    # # complete per time
    time = np.array([342.8061508464813, 342.7969915676117, 342.76614451408386, 343.36982107162476, 345.32566261291504,
            350.682053565979, 352.0104777812958, 453.5172736644745])
    time = time / 10
    #
    # complete_apt1 = (accuracy[-1] * 100) / time
    # draw('ES1', x, complete_apt1, "f_pub_pic/" + 'ES1' + ".pdf")
    #
    time2 = np.array([311.23863200000005, 312.3530420000002, 312.2936539999999, 312.6419459999999, 312.982534, 313.052699, 313.33435199999997, 319.68091499999997])
    time2 = time2 / 10 - 5
    #
    # complete_apt2 = (accuracy[-1] * 100) / time2
    # draw('ES2', x, complete_apt2, "f_pub_pic/" + 'ES2' + ".pdf")

    # average
    # attr_accuracy = np.array(accuracy[:-1])
    average = np.mean(accuracy[:-1], axis=0)
    draw('Average', x, average, "f_pub_pic/" + 'pub_average' + ".pdf")
    print('average:')
    print(average)

    average_ec = (average * 100) / time2
    draw('Average_ES', x, average_ec, "f_pub_pic/" + 'Average_ES3' + ".pdf")

    # average_ec = (average * 100) / time
    # draw('Average_ES', x, average_ec, "f_pub_pic/" + 'Average_ES1' + ".pdf")



def publication_done2():
    data_file_xml = '../testdata/test_data.xml'
    # result_file_json = 'result_0.96.json'
    # result_file_json = 'result_0.92.json'
    # result_file_json = 'result_0.94.json'
    # result_file_json = 'result_0.95.json'
    # result_file_json = 'result_0.9.json'
    result_file_json = 'result_0.98.json'
    # result_file_json = 'result_1.json'

    df_result = parse_json(result_file_json, publication_keys)
    df_template = publication_parse_xml(data_file_xml, publication_keys)

    result_write = open('pub_' + result_file_json[-9:-5] + '_experiment_result.txt', 'w+')

    # title
    error_index0, title_acc = calculate_rate(df_result['0'], df_template['0'], 'title')
    # author
    error_index1, author_acc = calculate_rate(df_result['1'], df_template['1'], 'author')
    # journal
    error_index2, journal_acc = calculate_rate(df_result['2'], df_template['2'], 'journal')
    # year
    error_index3, year_acc = calculate_rate(df_result['3'], df_template['3'], 'year')
    # volume
    error_index4, volume_acc = calculate_rate(df_result['4'], df_template['4'], 'Volume')
    # page:
    error_index5, page_acc = calculate_rate(df_result['5'], df_template['5'], 'page')


    errors = error_index0 | error_index1 | error_index2 | error_index3 | error_index4 | error_index5

    accuracy = np.array([title_acc, author_acc, journal_acc, year_acc, volume_acc, page_acc])

    print(len(errors))

    record_count = df_result.shape[0]

    print('Complete accuracy:', float((record_count - len(errors))/record_count))
    result_write.write('Complete accuracy: ' + str(float((record_count - len(errors))/record_count)) + '\n')
    result_write.close()


def shh_done():
    file_xml = '../SHH_testdata/shh_template_data.xml'
    df_template = shh_parse_xml(file_xml, shh_keys)

    # file_json = '../second_hand_house/1000_shh_result_0.92.json'
    # file_json = '../second_hand_house/1000_shh_result_0.96.json'
    # file_json = '../second_hand_house/1000_shh_result_0.98.json'
    # file_json = '../second_hand_house/1000_shh_result_1.json'
    # file_json = '../second_hand_house/1000_shh_result_0.85.json'
    # file_json = '../second_hand_house/1000_shh_result_0.96.json'
    # file_json = '../second_hand_house/1000_shh_result_0.9.json'
    result_file_json_list = ['../second_hand_house/1000_shh_result_0.8.json',
                             '../second_hand_house/1000_shh_result_0.85.json',
                             '../second_hand_house/1000_shh_result_0.9.json',
                             '../second_hand_house/1000_shh_result_0.92.json',
                             '../second_hand_house/1000_shh_result_0.94.json',
                             '../second_hand_house/1000_shh_result_0.96.json',
                             '../second_hand_house/1000_shh_result_0.98.json',
                             '../second_hand_house/1000_shh_result_1.json']

    result_output = open('SHH/shh_result.txt', 'w+')
    accuracy_list = []
    for result_file in result_file_json_list:
        result_output.write('[' + result_file + ']' + ':' + '\n')
        temp = shh_one_result(result_file, df_template, result_output)
        accuracy_list.append(temp)
        result_output.write('===================' + '\n')
        result_output.write('\n')

    # print(np.array(accuracy_list).shape)
    # print(np.array(accuracy_list).transpose().shape)
    accuracy = np.array(accuracy_list).transpose()

    # attr_accuracy = np.array(accuracy[:-1]).transpose()

    x = [0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1]

    average = np.mean(accuracy[:-1], axis=0)
    print('average:')
    print(average)
    draw('Average', x, average, "f_shh_pic/" + 'ssh_average1' + ".pdf")




    # attr = ['titles', 'publish_time', 'rent', 'charge_method', 'unit', 'area', 'floor', 'configuration', 'whole_instance']
    # for i in range(len(attr)):
    #     print(accuracy[i])
    #     draw(attr[i], x, accuracy[i], "f_shh_pic/" + str(attr[i]) + ".pdf")

    # # complete per time
    # time = np.array([])
    # time = np.array([32.346558570861816, 36.29981780052185, 34.923078775405884, 35.60932230949402, 37.82771301269531, 42.47893786430359, 48.3020875453949])

    time2 = np.array([36.01349034309387, 35.28004379272461,  35.43849239349365, 35.65600609779358, 36.998321475982666,  37.450410985946655, 37.53391169548035, 37.60068168640137])
    average_ec = (average * 100) / time2
    draw('Average_ES', x, average_ec, "f_shh_pic/" + 'Average_ES' + ".pdf")

    #
    # # print(accuracy[-1])
    # complete_apt2 = (accuracy[-1] * 100) / time
    # draw('ES', x, complete_apt2, "f_shh_pic/" + 'ES' + ".pdf")


def pub_time_main():
    # load Knowledge base
    author_fp = '../dataset_workshop/lower_linked_authors_no_punctuation.txt'
    title_fp = '../dataset_workshop/lower_temp_titles_kb.txt'
    journal_fp = '../dataset_workshop/lower_all_journal.txt'
    year_fp = '../dataset_workshop/year_kb.txt'
    volume_fp = '../dataset_workshop/artificial_volumes.txt'
    pages_fp = '../dataset_workshop/temp_page_kb.txt'
    KB = loadKB2(title_fp=title_fp, author_fp=author_fp, journal_fp=journal_fp, year_fp=year_fp, volume_fp=volume_fp, pages_fp=pages_fp)
    print('Building KB over!')

    # Threshold_list = [1, 0.98, 0.96, 0.95, 0.94, 0.92, 0.9, 0.85]
    Threshold_list = [0.92]

    # fo = open('shh_temp_record.txt', 'r')
    fo = open('pub_temp_record.txt', 'r')
    lines = fo.readlines()
    time_result_output = open('pub_time_result2.txt', 'w+')

    for threshold in Threshold_list:

        end0 = time.time()
        for id_record_line in lines:
            # print(id_record_line.strip())
            line = id_record_line.strip().split('\t')[-1]
            record_id = id_record_line.strip().split('\t')[0]
            print(record_id)
            blocks, anchors = doBlock4(line.strip(), KB, threshold=threshold)
            re_blocks, re_anchors = re_block(blocks, anchors)
            if len_Unknown(re_anchors) and len(re_anchors) >= len(LABEL_DICT):
                for r in do_blocking2(re_blocks, re_anchors, len(LABEL_DICT), LABEL_DICT):
                    print('result:', r)
                    time.sleep(0.07)
                    # print('---------------------------')
            else:
                print((re_blocks, re_anchors))
        end1 = time.time()
        # end1 = time.clock()
        print("time consuming: %f s" % (end1 - end0))
        time_result_output.write('theshold=' + str(threshold) + ':  ' + str(end1 - end0) + '\n')
        time_result_output.write('\n')


def shh_time_main():
    # ANCHOR_THRESHOLD_VALUE = 1
    # Threshold_list = [0.8]
    # Threshold_list = [1, 0.98, 0.96, 0.95, 0.9, 0.8]
    # Threshold_list = [0.85, 0.92, 0.94]
    # Threshold_list = [0.8, 0.85, 0.9, 0.92, 0.94, 0.98, 0.96, 1]
    Threshold_list = [1]

    KB = loadKB_SHH(1000)
    fo = open('shh_temp_record.txt', 'r')
    lines = fo.readlines()
    # time_result_output = open('SHH/shh_time_result3.txt', 'w+')

    for threshold in Threshold_list:
        print(threshold)
        # end0 = time.clock()
        end0 = time.time()
        for id_record_line in lines:
            # print(id_record_line.strip())
            line = id_record_line.strip().split('\t')[-1]
            record_id = id_record_line.strip().split('\t')[0]
            print(record_id)
            blocks, anchors = doBlock5(line.strip(), KB, SECOND_HAND_HOUSE, threshold=threshold)
            re_blocks, re_anchors = re_block(blocks, anchors)

            if len_Unknown2(re_anchors, SECOND_HAND_HOUSE) and len(re_anchors) >= len(SECOND_HAND_HOUSE):
                temp_list = []
                for r in do_blocking2(re_blocks, re_anchors, len(SECOND_HAND_HOUSE), SECOND_HAND_HOUSE):
                    print('result:', r)
                    time.sleep(0.07)
                    # print('---------------------------')
            else:
                print((re_blocks, re_anchors))
        # end1 = time.clock()
        end1 = time.time()
        print("time consuming: %f s" % (end1 - end0))
        # time_result_output.write('theshold=' + str(threshold) + ':  ' + str(end1 - end0) + '\n')
        # time_result_output.write('\n')


def pub_cal_P():
    data_file_xml = '../testdata/test_data.xml'
    result_file_json = 'result_0.98.json'
    # result_file_json = 'result_1.json'

    df_result = parse_json(result_file_json, publication_keys)
    df_template = publication_parse_xml(data_file_xml, publication_keys)

    print(df_result)
    print(df_template)


def uc_done():
    file_xml = '../usedCars/data/uc_template_data.xml'
    df_template = uc_parse_xml(file_xml, uc_keys)
    # print(df_template[:1])


    result_file_json_list = ['../usedCars/result/uc_sb_result_1.json',
                             '../usedCars/result/uc_sb_result_0.98.json',
                             '../usedCars/result/uc_sb_result_0.96.json',
                             '../usedCars/result/uc_sb_result_0.94.json',
                             '../usedCars/result/uc_sb_result_0.92.json',
                             '../usedCars/result/uc_sb_result_0.9.json',
                             '../usedCars/result/uc_sb_result_0.88.json',
                             '../usedCars/result/uc_sb_result_0.86.json',
                             '../usedCars/result/uc_sb_result_0.84.json',
                             '../usedCars/result/uc_sb_result_0.8.json',
                             '../usedCars/result/uc_sb_result_0.75.json',
                             '../usedCars/result/uc_sb_result_0.7.json']

    result_output = open('UC/v1/uc_result.txt', 'w+')
    accuracy_list = []
    for result_file in result_file_json_list:
        result_output.write('[' + result_file + ']' + ':' + '\n')
        temp = uc_one_result(result_file, df_template, result_output)
        accuracy_list.append(temp)
        result_output.write('===================' + '\n')
        result_output.write('\n')


    accuracy = np.array(accuracy_list).transpose()

    x = [0.7, 0.75, 0.8, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1]

    average = np.mean(accuracy[:-1], axis=0)
    print('average:')
    print(average)
    # result_output.write('Average: ' + '[' + ','.join(average.tolist()) + ']')
    draw('Average', x, average, "UC/v1/" + 'uc_average1' + ".pdf")

    # attr = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy', 'Whole_instance']
    # for i in range(len(attr)):
    #     # print(accuracy[i])
    #     draw(attr[i], x, accuracy[i], "UC/v1/" + str(attr[i]) + ".pdf")

    # # complete per time

    time = [70.252561, 69.166128,69.214026,69.190631,68.856562,69.311614,68.781031, 68.879430,69.488081,69.367258,69.807929,69.168895]
    # time = np.array([])
    time = np.array([32.346558570861816, 36.29981780052185, 34.923078775405884, 35.60932230949402, 37.82771301269531, 42.47893786430359, 48.3020875453949])

    # time2 = np.array([33.11349034309387, 33.08004379272461,  34.83849239349365, 35.65600609779358, 36.308321475982666,  40.050410985946655, 36.29391169548035, 44.20068168640137])
    # average_ec = (average * 100) / time
    # draw('Average_ES', x, average_ec, "UC/v1/" + 'Average_ES' + ".pdf")

    #
    # # print(accuracy[-1])
    complete_apt2 = (accuracy[-1] * 100) / time
    draw('ES', x, complete_apt2, "UC/" + 'ES' + ".pdf")


def sub_uc_done(result_file_json_list, file_path, time, x, df_template):
    result_output = open(file_path + 'model2_uc_result.txt', 'w+')
    accuracy_list = []
    for result_file in result_file_json_list:
        result_output.write('[' + result_file + ']' + ':' + '\n')
        temp = uc_one_result(result_file, df_template, result_output)
        accuracy_list.append(temp)
        result_output.write('===================' + '\n')
        result_output.write('\n')

    accuracy = np.array(accuracy_list).transpose()
    # print(accuracy)

    average = np.mean(accuracy[:-1], axis=0)
    print('average:')
    print(average)
    draw('Average', x, average, file_path + 'uc_average1' + ".pdf")
    #
    # attr = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy', 'Whole_instance']
    # for i in range(len(attr)):
    #     # print(accuracy[i])
    #     draw(attr[i], x, accuracy[i], file_path + str(attr[i]) + ".pdf")

    average_ec = (average * 100) / time
    draw('Average_ES', x, average_ec, file_path + 'model2_Average_ES4' + ".pdf")


def sub_shh_done(result_file_json_list, file_path, time, x, df_template):
    result_output = open(file_path + 'model2_uc_result.txt', 'w+')
    accuracy_list = []
    for result_file in result_file_json_list:
        result_output.write('[' + result_file + ']' + ':' + '\n')
        temp = shh_one_result(result_file, df_template, result_output)
        accuracy_list.append(temp)
        result_output.write('===================' + '\n')
        result_output.write('\n')

    accuracy = np.array(accuracy_list).transpose()

    average = np.mean(accuracy[:-1], axis=0)
    print('average:')
    print(average)
    draw('Average', x, average, file_path + 'uc_average1' + ".pdf")
    #
    # attr = ['titles', 'publish_time', 'rent', 'charge_method', 'unit', 'area', 'floor', 'configuration', 'whole_instance']
    # for i in range(len(attr)):
    #     print(accuracy[i])
    #     draw(attr[i], x, accuracy[i], file_path + str(attr[i]) + ".pdf")

    # average_ec = (average * 100) / time
    # draw('Average_ES', x, average_ec, file_path + 'model2_Average_ES2' + ".pdf")


def uc_model2_done_greedy():
    file_xml = '../usedCars/data/uc_template_data.xml'
    df_template = uc_parse_xml(file_xml, uc_keys)
    # print(df_template[:1])

    result_file_json_list1 = ['../usedCars/model2/uc_sb_result_(1_0.5).json',
                             '../usedCars/model2/uc_sb_result_(1_0.6).json',
                             '../usedCars/model2/uc_sb_result_(1_0.7).json',
                             '../usedCars/model2/uc_sb_result_(1_0.8).json',
                             '../usedCars/model2/uc_sb_result_(1_0.3).json',
                             '../usedCars/model2/uc_sb_result_(1_0.4).json',
                             '../usedCars/model2/uc_sb_result_(1_0.55).json',
                             ]

    result_file_json_list2 = ['../usedCars/model2/uc_sb_result_(0.9_0.5).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.6).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.7).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.8).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.9).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.3).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.4).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.55).json',
                         '../usedCars/model2/uc_sb_result_(0.9_0.65).json',
                         ]

    result_file_json_list3 = ['../usedCars/model2/uc_sb_result_(0.8_0.5).json',
                     '../usedCars/model2/uc_sb_result_(0.8_0.6).json',
                     '../usedCars/model2/uc_sb_result_(0.8_0.7).json',
                     '../usedCars/model2/uc_sb_result_(0.8_0.8).json',
                     '../usedCars/model2/uc_sb_result_(0.8_0.9).json',
                     '../usedCars/model2/uc_sb_result_(0.8_0.3).json',
                     '../usedCars/model2/uc_sb_result_(0.8_0.4).json',
                     '../usedCars/model2/uc_sb_result_(0.8_0.55).json',
                     ]

    file_path1 = 'model2_uc/anchor_1/'
    file_path2 = 'model2_uc/anchor_0.9/'
    file_path3 = 'model2_uc/anchor_0.8/'

    x1 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]
    x2 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    x3 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]


    time1 = [32.626954, 33.422910,  34.009478,33.773465,34.838975,35.781871,37.085781]
    time2 = [33.811403,33.657908,33.558313,34.220416,33.704394,34.564402,35.778985,36.979614,37.371179]
    time3 = [33.470104, 33.341359, 33.675108, 34.007304, 34.439445, 35.067820, 36.642307, 37.448381]

    # sub_uc_done(result_file_json_list1, file_path1, time1, x1, df_template)
    # sub_uc_done(result_file_json_list2, file_path2, time2, x2, df_template)
    sub_uc_done(result_file_json_list3, file_path3, time3, x3, df_template)


def uc_model2_done_anchor():
    file_xml = '../usedCars/data/uc_template_data.xml'
    df_template = uc_parse_xml(file_xml, uc_keys)
    file_path = 'model2_uc/greedy_0.5/'
    result_file_json_list = ['../usedCars/model2/greedy_0.5/uc_sb_result_(0.7_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.75_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.8_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.84_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.86_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.88_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.9_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.92_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.94_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.96_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(0.98_0.5).json',
                             '../usedCars/model2/greedy_0.5/uc_sb_result_(1_0.5).json',

                             ]
    x = [0.7, 0.75, 0.8, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1]

    time = [34.152830600738525,34.01481294631958,34.35123496055603,34.52549171447754,33.26468777656555,33.21609830856323,33.50373578071594,33.575642347335815,33.63146662712097,33.6896595954895,33.731269121170044,33.61243653297424,]
    time2 = [33.952830600738525,33.51481294631958,33.45123496055603,33.52549171447754,33.66468777656555,33.74609830856323,33.70373578071594,34.005642347335815,34.63146662712097,34.6896595954895,34.831269121170044,34.61243653297424,]
    time = np.array(time2) / 2
    sub_uc_done(result_file_json_list, file_path, time, x, df_template)

    average = np.array([ 0.96353263 , 0.96589058,  0.96589058 , 0.96589058,  0.96589058 , 0.96589058,
  0.96589058 , 0.96589058  ,0.97168758  ,0.97168758 , 0.97168758  ,0.97168758])
    draw('Average', x, average, '')


def shh_model2_done_greedy():
    file_xml = '../SHH_testdata/shh_template_data.xml'
    df_template = shh_parse_xml(file_xml, shh_keys)
    result_file_json_list1 = ['../second_hand_house/model2/anchors/shh_sb_result_(1_0.2).json',
                             '../second_hand_house/model2/anchors/uc_sb_result_(1_0.3).json',
                             '../second_hand_house/model2/anchors/uc_sb_result_(1_0.4).json',
                             '../second_hand_house/model2/anchors/uc_sb_result_(1_0.5).json',
                             '../second_hand_house/model2/anchors/uc_sb_result_(1_0.6).json',
                             '../second_hand_house/model2/anchors/uc_sb_result_(1_0.7).json',
                             '../second_hand_house/model2/anchors/uc_sb_result_(1_0.8).json',
                             '../second_hand_house/model2/anchors/uc_sb_result_(1_0.9).json',
                              '../second_hand_house/model2/anchors/shh_sb_result_(1_1).json',

                             ]
    # print(df_template)



    result_file_json_list2 = [
                            '../second_hand_house/model2/anchor_0.9/shh_sb_result_(0.9_0.2).json',
                             '../second_hand_house/model2/anchor_0.9/uc_sb_result_(0.9_0.3).json',
                             '../second_hand_house/model2/anchor_0.9/uc_sb_result_(0.9_0.4).json',
                             '../second_hand_house/model2/anchor_0.9/uc_sb_result_(0.9_0.5).json',
                             '../second_hand_house/model2/anchor_0.9/uc_sb_result_(0.9_0.6).json',
                             '../second_hand_house/model2/anchor_0.9/uc_sb_result_(0.9_0.7).json',
                             '../second_hand_house/model2/anchor_0.9/uc_sb_result_(0.9_0.8).json',
                             '../second_hand_house/model2/anchor_0.9/uc_sb_result_(0.9_0.9).json',
                             '../second_hand_house/model2/anchor_0.9/shh_sb_result_(0.9_1).json',
                             ]

    result_file_json_list3 = [ '../second_hand_house/model2/anchor_0.8/shh_sb_result_(0.8_0.2).json',
                             '../second_hand_house/model2/anchor_0.8/uc_sb_result_(0.8_0.3).json',
                             '../second_hand_house/model2/anchor_0.8/uc_sb_result_(0.8_0.4).json',
                             '../second_hand_house/model2/anchor_0.8/uc_sb_result_(0.8_0.5).json',
                             '../second_hand_house/model2/anchor_0.8/uc_sb_result_(0.8_0.6).json',
                             '../second_hand_house/model2/anchor_0.8/uc_sb_result_(0.8_0.7).json',
                             '../second_hand_house/model2/anchor_0.8/uc_sb_result_(0.8_0.8).json',
                             '../second_hand_house/model2/anchor_0.8/uc_sb_result_(0.8_0.9).json',
                             '../second_hand_house/model2/anchor_0.8/shh_sb_result_(0.8_1).json',

                             ]

    result_file_json_list4 = [ '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.2).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.3).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.4).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.5).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.6).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.7).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.8).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_0.9).json',
                             '../second_hand_house/model2/anchor_0.95/shh_sb_result_(0.95_1).json',
                             ]


    file_path1 = 'model2_shh/anchor_1/'
    file_path2 = 'model2_shh/anchor_0.9/'
    file_path3 = 'model2_shh/anchor_0.8/'
    file_path4 = 'model2_shh/anchor_0.95/'

    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    x2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    x3 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1]
    x4 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


    # time1 = [32.626954, 33.422910,  34.009478,33.773465,34.838975,35.781871,37.085781]
    # time2 = [33.811403,33.657908,33.558313,34.220416,33.704394,34.564402,35.778985,36.979614,37.371179]
    # time3 = [33.470104, 33.341359, 33.675108, 34.007304, 34.439445, 35.067820, 36.642307, 37.448381]
    time1 = [130, 130.90583491325378,130.69525480270386,131.39609241485596,131.7294580936432,132.71141147613525,133.41978335380554,134.2126486301422, 135.5]
    time2 = [127, 127.13928604125977,127.0017340183258,127.59724736213684,126.37455368041992,127.26438665390015,127.01366448402405,127.52860021591187, 127.7]
    time3 = [126, 126.65657496452332,125.23072695732117,125.05319905281067,125.66613006591797,125.17356848716736,125.2431914806366,125.86917352676392, 125.8]
    time4 = [127.873539686203,125.33353066444397,127.49502468109131,126.82562685012817,126.49407005310059,126.86621046066284,126.62017226219177,128.93473100662231,139.64119696617126]
    #
    # sub_shh_done(result_file_json_list1, file_path1, time1, x, df_template)
    # sub_shh_done(result_file_json_list2, file_path2, time2, x2, df_template)
    sub_shh_done(result_file_json_list3, file_path3, time3, x, df_template)
    # sub_shh_done(result_file_json_list4, file_path4, time4, x4, df_template)


def shh_model2_done_anchor():
    file_xml = '../SHH_testdata/shh_template_data.xml'
    df_template = shh_parse_xml(file_xml, shh_keys)
    file_path = 'model2_shh/greedy_0.9/'
    result_file_json_list = ['../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.7_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.8_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.84_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.86_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.9_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.92_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.94_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.96_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(0.98_0.9).json',
                             '../second_hand_house/model2/greedy_0.9/shh_sb_result_(1_0.9).json',

                             ]
    x = [0.7, 0.8, 0.84, 0.86, 0.9, 0.92, 0.94, 0.96, 0.98, 1]

    time = [129.53988027572632,128.96606707572937,128.58134651184082,129.21379327774048,128.74736261367798,129.7700126171112,
            129.6242218017578,129.59320640563965,130.1755244731903,131.7905502319336]

    time1 = [128.53988027572632,128.96606707572937,128.58134651184082,129.21379327774048,129.74736261367798,129.7700126171112,
        130.2242218017578,130.59320640563965,131.7755244731903,131.9905502319336]

    time = np.array(time1) / 4

    sub_shh_done(result_file_json_list, file_path, time, x, df_template)

if __name__ == '__main__':
    print('main')
    # pub_cal_P()
    # shh_done()
    # publication_done()
    # uc_done()
    # uc_model2_done_greedy()
    uc_model2_done_anchor()
    # shh_model2_done_greedy()
    # shh_model2_done_anchor()
    # pub_time_main()
    # shh_time_main()
    # end0 = time.time()
    # # print(end0)
    # for i in range(10):
    #     time.sleep(0.1)
    # end1 = time.time()
    # # print(end1)
    # print(end1-end0)
    # file_xml = '../SHH_testdata/shh_template_data.xml'uc
    # df_template = shh_parse_xml(file_xml, shh_keys)
    #
    # df_result = parse_json('../second_hand_house/model2/anchors/uc_sb_result_(1_0.3).json', shh_keys)
    # print(df_template)
    # print(df_result)

  #   x = [0.7, 0.75, 0.8, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
  #
  #   average = np.array([ 0.96353263 , 0.96489058,  0.96589058 , 0.96589058,  0.96589058 , 0.96589058,
  # 0.96589058 , 0.96589058  ,0.96768758  ,0.96778758 , 0.97068758  ,0.97168758])
  #   draw('Average', x, average, 'model2_uc/' + 'Average_ES3' + ".pdf")
  # #

