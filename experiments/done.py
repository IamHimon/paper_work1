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


def draw(pic_name, x, y, path):
    # pic_name = 'pub_complete_apt'

    plt.figure(figsize=(8, 5)) #创建绘图对象
    plt.plot(x, y, "b--", linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度
    plt.xlabel("Threshold") #X轴标签
    plt.ylabel("Accuracy")  #Y轴标签
    plt.title(pic_name) #图标题
    plt.savefig(path) #保存图
    # plt.show()  #显示图
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
    r1 = '\,+'
    r2 = '\s+'
    r = '|'.join([r1, r2])
    sum_count = result.size
    print('sum_count', sum_count)
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
    print(lable + ' accuracy: %.5f', float((sum_count - error)/sum_count))
    # result_write.write(lable + ' accuracy: ' + str(float((sum_count - error)/sum_count)) + '\n')
    accuracy = float((sum_count - error)/sum_count)
    # print(error_index)
    return error_index, accuracy


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


# Second hand house
def calculate_rate3(result, template, lable, result_write):
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
    return df


# parse the .json
def parse_json(filename, keys):
    df = pd.DataFrame(columns=keys)
    f = open(filename, "r")
    for line in f:
        decodes = json.loads(line)
        # print(decodes['blocks'])
        # print(decodes['labels'])
        # print(decodes['predictions'])
        # print(decodes['ID'])
        bp_dict = blocks_map_predictions(decodes['blocks'], decodes['predictions'])
        # print(bp_dict)
        blocks = Series(bp_dict)
        df.loc[decodes['ID']] = blocks
        # print(blocks)
    f.close()
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

    df_result = parse_json(result_file_json, shh_keys)
    title, title_acc = calculate_rate2(df_result['0'], df_template['0'], 'title')
    publish_time, publish_time_acc = calculate_rate2(df_result['1'], df_template['1'], 'publish_time')
    rent, rent_acc = calculate_rate2(df_result['2'], df_template['2'], 'rent')
    charge_method, charge_method_acc = calculate_rate2(df_result['3'], df_template['3'], 'charge_method')
    unit, unit_acc = calculate_rate2(df_result['4'], df_template['4'], 'unit')
    area, area_acc = calculate_rate2(df_result['5'], df_template['5'], 'area')
    floor, floor_acc = calculate_rate2(df_result['6'], df_template['6'], 'floor')
    configuration, configuration_acc = calculate_rate2(df_result['7'], df_template['7'], 'configuration')

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
    accuracy = [title_acc, publish_time_acc, rent_acc, charge_method_acc, unit_acc, area_acc, floor_acc, configuration_acc, complete_acc]
    return accuracy


def publication_done():
    data_file_xml = '../testdata/test_data.xml'
    df_template = publication_parse_xml(data_file_xml, publication_keys)
    # result_file_json = 'result_0.96.json'
    # result_file_json = 'result_0.92.json'
    # result_file_json = 'result_0.94.json'
    # result_file_json = 'result_0.95.json'
    # result_file_json = 'result_0.9.json'
    result_file_json = 'result_0.98.json'
    # result_file_json = 'result_1.json'

    result_file_json_list = ['result_0.85.json', 'result_0.9.json',  'result_0.92.json', 'result_0.94.json', 'result_0.95.json',
                             'result_0.96.json', 'result_0.98.json', 'result_1.json']

    result_output = open('pub_result.txt', 'w+')
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
    # attr = ['title', 'author', 'journal', 'year', 'volume', 'pages', 'complete']
    # for i in range(len(attr)):
    #     print(accuracy[i])
    #     draw(attr[i], x, accuracy[i], "pub_pic2/" + str(attr[i]) + ".pdf")

    # complete per time
    time = np.array([335.3761508464813, 342.3169915676117, 342.76614451408386, 343.36982107162476, 345.32566261291504,
            350.682053565979, 369.0104777812958, 432.5172736644745])

    complete_apt1 = (accuracy[-1] * 100) / time
    draw('complete_apt1', x, complete_apt1, "pub_pic2/" + 'complete_apt1' + ".pdf")

    time2 = np.array([311.23863200000005, 312.3530420000002, 311.4936539999999, 312.4419459999999, 311.582534, 311.152699, 311.33435199999997, 311.68091499999997])

    complete_apt2 = (accuracy[-1] * 100) / time2
    draw('complete_apt2', x, complete_apt2, "pub_pic2/" + 'complete_apt2' + ".pdf")


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
    file_json = '../second_hand_house/1000_shh_result_0.9.json'
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

    x = [0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
    attr = ['titles', 'publish_time', 'rent', 'charge_method', 'unit', 'area', 'floor', 'configuration', 'complete']
    for i in range(len(attr)):
        print(accuracy[i])
        draw(attr[i], x, accuracy[i], "1000_shh_pic/" + str(attr[i]) + ".pdf")

    # # complete per time
    # time = np.array([])
    time = np.array([32.346558570861816, 36.29981780052185, 34.923078775405884, 35.60932230949402, 37.82771301269531, 42.47893786430359, 48.3020875453949])

    time2 = np.array([33.11349034309387, 33.08004379272461,  34.83849239349365, 35.65600609779358, 36.308321475982666,  40.050410985946655, 36.29391169548035, 44.20068168640137])

    # print(accuracy[-1])
    complete_apt2 = (accuracy[-1] * 100) / time2
    draw('whole_apt', x, complete_apt2, "1000_shh_pic/" + 'whole_apt' + ".pdf")


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


if __name__ == '__main__':
    print('main')
    # pub_cal_P()
    # shh_done()
    # publication_done()
    # pub_time_main()
    shh_time_main()
    # end0 = time.time()
    # # print(end0)
    # for i in range(10):
    #     time.sleep(0.1)
    # end1 = time.time()
    # # print(end1)
    # print(end1-end0)

