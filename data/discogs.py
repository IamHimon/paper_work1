import xml.etree.ElementTree as ET
from lxml import etree

CATEGORIES = {'master'}
DATA_ITEMS1 = {'year', 'artists', 'styles', 'main_release', 'genres', 'title', 'data_quality'}
DATA_ITEMS = {'artist', 'name', 'genre', 'year', 'style', 'title', 'main_release'}
DISCOGS_DB_DUMP = '/home/himon/Jobs/paper_work1/dataset/discogs_20120301_masters.xml'


def clear_element(element):
    element.clear()
    while element.getprevious() is not None:
        del element.getparent()[0]


def extract_paper_elements(context):
    for event, element in context:
        if element.tag in CATEGORIES:
            yield element
            clear_element(element)


def fast_iter(context):
    for paperCounter, element in enumerate(extract_paper_elements(context)):
        print('paperCounter:', paperCounter)
        # print(element.tag)

        artist = [artist.text for artist in element.findall('artists')]
        print(artist)

        artists = {}
        genres = {}
        styles = {}
        master = {}
        for data_item in DATA_ITEMS1:
            data = element.find(data_item)
            if data is not None:
                master[data_item] = data.text

        print(master)



        if paperCounter > 200:
            return

            # authors = [author.text for author in element.findall("author")]
            # # 定义词典
            # paper = {
            #     'element': element.tag,
            #     'mdate': element.get("mdate"),
            #     'dblpkey': element.get('key')
            # }
            # for data_item in DATA_ITEMS:
            #     data = element.find(data_item)
            #     if data is not None:
            #         paper[data_item] = data  # 词典中加入新元素
            #
            # if (paper['element'] not in SKIP_CATEGORIES) and ("journal" in paper.keys())and("title" in paper.keys()):
            #     print(paper["title"].text)
            #     print(paper["journal"].text)
            #     print(authors)
            # print(paperCounter)


def main():
    # output = open("temp_title_author_journal.txt", 'w+')
    # infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    # infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(DISCOGS_DB_DUMP, events=("end",), load_dtd=True)  # 生成迭代器
    fast_iter(context)


if __name__ == '__main__':
    main()
#

# e = ET.parse(DISCOGS_DB_DUMP).getroot()
#
#
# print(e.tag)
# for child in e:
#     print(child.tag, child.attrib)