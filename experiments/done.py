import xml.dom.minidom
import json
from pandas import Series, DataFrame
import pandas as pd
import re
LABEL_DICT = {'Title': 0, 'Author': 1, 'Journal': 2, 'Year': 3, 'Volume': 4, 'Pages': 5}
keys = ['0', '1', '2', '3', '4', '5']


def blocks_map_predictions(blocks, predictions):
    bp_dict = {}
    for i in range(len(predictions)):
        bp_dict[predictions[i]] = blocks[i]
    return bp_dict


def sorted_dict(bp_dict):
    result = {}
    for i in keys:
        result[i] = bp_dict.get(i)
    return result


def calculate_rate(result, template, lable):
    print('++++++++++++++++++++++++++++++++++++++++++++')
    r1 = '\,+'
    r2 = '\s+'
    r = '|'.join([r1, r2])
    sum_count = result.size
    error = 0
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
            error += 1
    print(lable + ' error count: ', error)
    print(lable + ' accuracy: %.5f', float((sum_count - error)/sum_count))


# parse the .xml
def parse_xml(filename):
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


# parse the .json
def parse_json(filename):
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
    return df


if __name__ == '__main__':
    data_fale_xml = '../testdata/test_data.xml'
    result_file_json = 'result_0.96.json'
    df_result = parse_json(result_file_json)
    print(df_result)
    # print(df_result['5'])
    df_template = parse_xml(data_fale_xml)
    print(df_template)
    # page:
    #calculate_rate(df_result['5'], df_template['5'], 'page')
    # year
    #calculate_rate(df_result['3'], df_template['3'], 'year')
    # page
    #calculate_rate(df_result['4'], df_template['4'], 'page')
    # title
    #calculate_rate(df_result['0'], df_template['0'], 'title')
    # author
    #calculate_rate(df_result['1'], df_template['1'], 'author')
    # journal
    #calculate_rate(df_result['2'], df_template['2'], 'journal')



