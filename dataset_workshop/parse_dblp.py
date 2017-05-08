from lxml import etree
import random

CATEGORIES = {'article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', "mastersthesis", "www"}
SKIP_CATEGORIES = {'phdthesis', 'mastersthesis', 'www'}
DATA_ITEMS = ["title", "year", "journal", "ee", "year", "volume", "pages", "number"]


def clear_element(element):
    element.clear()
    while element.getprevious() is not None:
        del element.getparent()[0]


def extract_paper_elements(context):
    for event, element in context:
        if element.tag in CATEGORIES:
            yield element
            clear_element(element)


# build test dataset according dblp,
# one record:"title,author1,author2,author3,journal,year,volume,pages"
def build_samples(context, output, boundary1, boundary2, boundary3):
    record_count = 0
    author_part = ''
    linked_author_set = set()
    title_set = set()
    author_set = set()
    journal_set = set()
    pages_set = set()
    for paperCounter, element in enumerate(extract_paper_elements(context)):
        authors = [author.text for author in element.findall("author")]
        # 定义词典
        paper = {
            'element': element.tag,
            'mdate': element.get("mdate"),
            'dblpkey': element.get('key')
        }
        for data_item in DATA_ITEMS:
            data = element.find(data_item)
            if data is not None:
                paper[data_item] = data  # 词典中加入新元素

        if (paper['element'] not in SKIP_CATEGORIES)and ("journal" in paper.keys())and("title" in paper.keys())and ("year" in paper.keys())and("volume" in paper.keys())\
                    and("pages" in paper.keys())and(paper["journal"].text is not None)and (paper["title"].text is not None)\
                and(paper["year"].text is not None)and(paper["volume"].text is not None)and(paper["pages"].text is not None):
            record_count += 1
            print('paperCounter', paperCounter)
            print('record_count:', record_count)
            if record_count <= boundary2:  # 10000
                title_set.add(paper["title"].text.strip('.').lower())
                journal_set.add(paper["journal"].text.lower())
                pages_set.add(paper["pages"].text.lower())
                # for author in authors:
                #     author_set.add(author.lower())
                #
                # authors = [author.text for author in element.findall("author")]
                if authors:
                    if len(authors) == 1:
                        linked_author_set.add(authors[0].lower())
                        print(authors[0].lower())
                    else:
                        linked_author_set.add((' '.join(authors[:-1]) + ' and ' + authors[-1]).lower())
                        print((' '.join(authors[:-1]) + ' and ' + authors[-1]).lower())
            # 保留30%的sample

            if record_count >= boundary1:   # 7000
                for author in authors:
                    author_part += author + ','
                sample = author_part + paper["title"].text + ',' + paper["journal"].text + ',' + paper["year"].text \
                         + ',' + paper["volume"].text + ',' + paper["pages"].text
                print(sample)
                output.write(sample + '\n')
                author_part = ''
            if record_count == boundary3: # 20000
                break
    return title_set, linked_author_set, journal_set, pages_set


# build test data that have no cover
def build_samples2(context, boundary1, boundary2):
    record_count = 0
    linked_author_set = set()
    title_set = set()
    pages_set = set()
    for paperCounter, element in enumerate(extract_paper_elements(context)):
        authors = [author.text for author in element.findall("author")]
        # 定义词典
        paper = {
            'element': element.tag,
            'mdate': element.get("mdate"),
            'dblpkey': element.get('key')
        }
        for data_item in DATA_ITEMS:
            data = element.find(data_item)
            if data is not None:
                paper[data_item] = data  # 词典中加入新元素

        if (paper['element'] not in SKIP_CATEGORIES)and("title" in paper.keys())and ("year" in paper.keys())and("volume" in paper.keys())\
                    and("pages" in paper.keys())and (paper["title"].text is not None)\
                and(paper["year"].text is not None)and(paper["volume"].text is not None)and(paper["pages"].text is not None):
            record_count += 1
            print('paperCounter', paperCounter)
            print('record_count:', record_count)
            if record_count >= boundary1:  # 20000
                title_set.add(paper["title"].text.strip('.').lower())
                pages_set.add(paper["pages"].text.lower())
                if authors:
                    if len(authors) == 1:
                        linked_author_set.add(authors[0].lower())
                        print(authors[0].lower())
                    else:
                        linked_author_set.add((' '.join(authors[:-1]) + ' and ' + authors[-1]).lower())
                        print((' '.join(authors[:-1]) + ' and ' + authors[-1]).lower())

            if record_count == boundary2:    # 30000
                break
    return title_set, linked_author_set, pages_set


# build dataset which titles,authors and journals stored separately.
def fast_iter(context, t_output, a_output, j_output):
    for paperCounter, element in enumerate(extract_paper_elements(context)):
            authors = [author.text for author in element.findall("author")]
            # 定义词典
            paper = {
                'element': element.tag,
                'mdate': element.get("mdate"),
                'dblpkey': element.get('key')
            }
            for data_item in DATA_ITEMS:
                data = element.find(data_item)
                if data is not None:
                    paper[data_item] = data  # 词典中加入新元素

            if (paper['element'] not in SKIP_CATEGORIES) and ("journal" in paper.keys())and("title" in paper.keys()):
                print(paper["title"].text)
                print(paper["journal"].text)
                print(authors)
                if(authors is not None) and (paper["title"].text is not None) and (paper["journal"].text is not None):
                    t_output.write(paper["title"].text+'\n')
                    for author in authors:
                        if author is not None:
                            a_output.write(author+"\n")
                    j_output.write(paper["journal"].text+'\n')
            print(paperCounter)


# build dataset which year,volume and pages stored separately.
def fast_iter5(context, y_output, v_output, p_output):
    for paperCounter, element in enumerate(extract_paper_elements(context)):
            # 定义词典
            paper = {
                'element': element.tag,
                'mdate': element.get("mdate"),
                'dblpkey': element.get('key')
            }
            for data_item in DATA_ITEMS:
                data = element.find(data_item)
                if data is not None:
                    paper[data_item] = data  # 词典中加入新元素

            if (paper['element'] not in SKIP_CATEGORIES) and ("year" in paper.keys())and("volume" in paper.keys())\
                    and("pages" in paper.keys()):
                print(paper["year"].text)
                print(paper["volume"].text)
                print(paper["pages"].text)
                if(paper["year"].text is not None) and (paper["volume"].text is not None)and (paper["pages"].text is not None):
                    y_output.write(paper["year"].text+'\n')
                    v_output.write(paper["volume"].text+'\n')
                    p_output.write(paper["pages"].text+'\n')
            print(paperCounter)


# store:"title#&author#&journal"
def fast_iter2(context, output):
    for paperCounter, element in enumerate(extract_paper_elements(context)):
            authors = [author.text for author in element.findall("author")]
            # 定义词典
            paper = {
                'element': element.tag,
                'mdate': element.get("mdate"),
                'dblpkey': element.get('key')
            }
            for data_item in DATA_ITEMS:
                data = element.find(data_item)
                if data is not None:
                    paper[data_item] = data  # 词典中加入新元素

            if (paper['element'] not in SKIP_CATEGORIES) and ("journal" in paper.keys())and("title" in paper.keys()):
                print(paper)
                print(paper["title"].text)
                print(paper["journal"].text)
                print(authors)
                if(authors is not None) and (paper["title"].text is not None) and (paper["journal"].text is not None):
                    authors_copy = [author for author in authors if author is not None]
                    print(paper)
                    print(paper["title"].text)
                    print(paper["journal"].text)
                    print(authors_copy)
                    print(paper["title"].text+"|"+",".join(authors_copy)+"|"+paper["journal"].text)
                    output.write(paper["title"].text+"#$"+",".join(authors_copy)+"#$"+paper["journal"].text)
                    output.write('\n')
            print(paperCounter)


# build corpus for word2vec,store title,author,journal per line.
def fast_iter3(context, output):
    for paperCounter, element in enumerate(extract_paper_elements(context)):
            authors = [author.text for author in element.findall("author")]
            # 定义词典
            paper = {
                'element': element.tag,
                'mdate': element.get("mdate"),
                'dblpkey': element.get('key')
            }
            for data_item in DATA_ITEMS:
                data = element.find(data_item)
                if data is not None:
                    paper[data_item] = data  # 词典中加入新元素

            if (paper['element'] not in SKIP_CATEGORIES) and ("journal" in paper.keys())and("title" in paper.keys()):
                if(authors is not None) and (paper["title"].text is not None) and (paper["journal"].text is not None):
                    authors_copy = [author for author in authors if author is not None]
                    print(paper["title"].text)
                    print(paper["journal"].text)
                    print(authors_copy)
                    output.write(paper["title"].text)
                    output.write('\n')
                    for a in authors_copy:
                        output.write(a)
                        output.write('\n')
                    output.write(paper["journal"].text)
                    output.write('\n')
            print(paperCounter)


# 返回不重复的set,然后持久化,用来训练word2vec,title去掉'.'
def fast_iter4(context, output):
    for paperCounter, element in enumerate(extract_paper_elements(context)):
            authors = [author.text for author in element.findall("author")]
            # 定义词典
            paper = {
                'element': element.tag,
                'mdate': element.get("mdate"),
                'dblpkey': element.get('key')
            }
            for data_item in DATA_ITEMS:
                data = element.find(data_item)
                if data is not None:
                    paper[data_item] = data  # 词典中加入新元素

            if (paper['element'] not in SKIP_CATEGORIES) and ("journal" in paper.keys())and("title" in paper.keys()):
                if(authors is not None) and (paper["title"].text is not None) and (paper["journal"].text is not None):
                    authors_copy = [author for author in authors if author is not None]
                    print(paper["title"].text)
                    print(paper["journal"].text)
                    print(authors_copy)
                    output.write(paper["title"].text)
                    output.write('\n')
                    for a in authors_copy:
                        output.write(a)
                        output.write('\n')
                    output.write(paper["journal"].text)
                    output.write('\n')
            print(paperCounter)


def fast_iter6(context):
    linked_author_set = set()
    for paperCounter, element in enumerate(extract_paper_elements(context)):
        print(paperCounter)
        authors = [author.text for author in element.findall("author")]
        if authors:
            if len(authors) == 1:
                linked_author_set.add(authors[0].lower())
                print(authors[0])
            else:
                linked_author_set.add((' '.join(authors[:-1]) + ' and ' + authors[-1]).lower())
                print((' '.join(authors[:-1]) + ' and ' + authors[-1]).lower())
                # if paperCounter % 2 == 0:
                #     linked_author_set.add(' '.join(authors))
                #     print(' '.join(authors))
                # else:
                #     linked_author_set.add(' '.join(authors[:-1]) + ' and ' + authors[-1])
                #     print(' '.join(authors[:-1]) + ' and ' + authors[-1])
    return linked_author_set


def fast_iter7(context):
    all_pages_set = set()
    for paperCounter, element in enumerate(extract_paper_elements(context)):
        print(paperCounter)
        # 定义词典
        paper = {
            'element': element.tag,
            'mdate': element.get("mdate"),
            'dblpkey': element.get('key')
        }
        for data_item in DATA_ITEMS:
            data = element.find(data_item)
            if data is not None:
                paper[data_item] = data  # 词典中加入新元素

        if (paper['element'] not in SKIP_CATEGORIES) and ("pages" in paper.keys())and (paper["pages"].text is not None):
            print(paper["pages"].text)
            all_pages_set.add(paper["pages"].text)

    return all_pages_set


# author1,author2,author3 and author4, title, journal, year,volume(number),pages
def fast_iter8(context, output, boundary1, boundary2):
    for paperCounter, element in enumerate(extract_paper_elements(context)):
        if paperCounter >= boundary1:
            taj_record_list = []
            yvp_record_list = []

            # 定义词典
            paper = {
                'element': element.tag,
                'mdate': element.get("mdate"),
                'dblpkey': element.get('key')
            }
            for data_item in DATA_ITEMS:
                data = element.find(data_item)
                if data is not None:
                    paper[data_item] = data  # 词典中加入新元素
            # print(paper.keys())
            # print(paper['element'])

            authors = [author.text for author in element.findall("author")]
            if authors:
                    if len(authors) == 1:
                        taj_record_list.append(authors[0])
                    else:
                        taj_record_list.append(','.join(authors[:-1]) + ' and ' + authors[-1])

            if ('title' in paper.keys()) and (paper['title'].text is not None):
                # print(paper['title'].text)
                taj_record_list.append(paper['title'].text.strip('.'))

            if ("journal" in paper.keys()) and (paper["journal"].text is not None):
                # journal = paper["journal"].text
                taj_record_list.append(paper["journal"].text.strip('.'))

            if ('year' in paper.keys()) and (paper['year'].text is not None):
                # year = paper['year'].text
                yvp_record_list.append(paper['year'].text)

            if ('volume' in paper.keys()) and ('number' in paper.keys()) and (paper['volume'].text is not None) and \
                    (paper['number'].text is not None):
                # volume_number = paper['volume'].text + '(' + paper['number'].text + ')'
                yvp_record_list.append(paper['volume'].text + '(' + paper['number'].text + ')')
            elif ('volume' in paper.keys()) and (paper['volume'].text is not None):
                # volume_number = paper['volume'].text
                yvp_record_list.append(paper['volume'].text)

            if ('pages' in paper.keys()) and (paper['pages'].text is not None):
                # pages = paper['pages'].text
                yvp_record_list.append(paper['pages'].text)

            # print(taj_record_list)
            # print(yvp_record_list)
            random.shuffle(taj_record_list)
            random.shuffle(yvp_record_list)
            result_record = taj_record_list + yvp_record_list
            print(result_record)
            print(','.join(result_record))
            output.write(','.join(result_record) + '\n')
            print(paperCounter)
        if paperCounter == boundary2:
            break


def fast_iter9(context, output, boundary1, boundary2):
    record_count = 0
    for paperCounter, element in enumerate(extract_paper_elements(context)):
        if paperCounter >= boundary1:
            taj_record_list = []
            yvp_record_list = []

            # 定义词典
            paper = {
                'element': element.tag,
                'mdate': element.get("mdate"),
                'dblpkey': element.get('key')
            }
            for data_item in DATA_ITEMS:
                data = element.find(data_item)
                if data is not None:
                    paper[data_item] = data  # 词典中加入新元素
            # print(paper.keys())
            # print(paper['element'])

            authors = [author.text for author in element.findall("author")]
            if authors and ('title' in paper.keys()) and (paper['title'].text is not None) and ("journal" in paper.keys()) and (paper["journal"].text is not None)\
                    and ('year' in paper.keys()) and (paper['year'].text is not None) and ('volume' in paper.keys()) and ('number' in paper.keys()) and (paper['volume'].text is not None) and \
                    (paper['number'].text is not None) and ('pages' in paper.keys()) and (paper['pages'].text is not None)\
                    and ('e' not in paper['pages'].text):
                record_count += 1
                if len(authors) == 1:
                    taj_record_list.append(authors[0])
                else:
                    taj_record_list.append(','.join(authors[:-1]) + ' and ' + authors[-1])
                taj_record_list.append(paper['title'].text.strip('.'))
                taj_record_list.append(paper["journal"].text.strip('.'))
                yvp_record_list.append(paper['year'].text)
                yvp_record_list.append(paper['volume'].text + '(' + paper['number'].text + ')')
                yvp_record_list.append(paper['pages'].text)
                # print(taj_record_list)
                # print(yvp_record_list)
                random.shuffle(taj_record_list)
                random.shuffle(yvp_record_list)
                result_record = taj_record_list + yvp_record_list
                print(result_record)
                print(','.join(result_record))
                output.write(','.join(result_record) + '\n')
            print(record_count)
        if paperCounter == boundary2:
            break

def main():
    output = open("temp_title_author_journal.txt", 'w+')
    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile, events=("end",), load_dtd=True)  # 生成迭代器
    fast_iter2(context, output)
    output.close()


def buildcorpus4word2vec():
    output = open("corpus4word2vec.txt", 'w+')
    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile2, events=("end",), load_dtd=True)  # 生成迭代器
    fast_iter3(context, output)
    output.close()


def build():
    t_output = open("temp_titles.txt", 'w+')
    a_output = open('temp_authors.txt', 'w+')
    j_output = open('temp_journals.txt', 'w+')
    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile, events=("end",), load_dtd=True)  # 生成迭代器
    fast_iter(context, t_output, a_output, j_output)
    t_output.close()
    a_output.close()
    j_output.close()


def build2():
    y_output = open("temp_year.txt", 'w+')
    v_output = open('temp_volume.txt', 'w+')
    p_output = open('temp_page.txt', 'w+')
    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile, events=("end",), load_dtd=True)  # 生成迭代器
    fast_iter5(context, y_output, v_output, p_output)
    y_output.close()
    v_output.close()
    p_output.close()


def build_samples_main():
    # t_output = open("lower_temp_titles_kb.txt", 'w+')
    a_output = open('lower_temp_linked_authors_kb.txt', 'w+')
    # j_output = open('lower_temp_journals_kb.txt', 'w+')
    # y_output = open('temp_year_kb.txt', 'w+')
    # v_output = open('temp_volume_kb.txt', 'w+')
    # p_output = open('lower_temp_page.txt_kb', 'w+')
    output = open('temp_dataset3.txt', 'w+')

    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile2, events=("end",), load_dtd=True)
    title_set, author_set, journal_set, pages_set = build_samples(context, output, 7000, 10000, 20000)
    # save_data(title_set, t_output)
    save_data(author_set, a_output)
    # save_data(journal_set, j_output)
    # save_data(pages_set, p_output)
    #
    output.close()


# have no cover
def build_samples_main2():
    t_output = open("../testdata/no_cover_lower_temp_titles_kb.txt", 'w+')
    a_output = open('../testdata/no_cover_lower_temp_linked_authors_kb.txt', 'w+')
    # y_output = open('temp_year_kb.txt', 'w+')
    # v_output = open('temp_volume_kb.txt', 'w+')
    p_output = open('../testdata/no_cover_lower_temp_page.txt_kb', 'w+')

    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile2, events=("end",), load_dtd=True)
    title_set, author_set, pages_set = build_samples2(context, 20000, 24000)
    save_data(title_set, t_output)
    save_data(author_set, a_output)
    save_data(pages_set, p_output)


def build_cleared_corpus4word2vec():
    output = open("cleared_corpus4word2vec.txt", 'w+')
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile2, events=("end",), load_dtd=True)  # 生成迭代器
    title_set = set()
    author_set = set()
    journal_set = set()


def build_year4KB():
    file_name = 'year_kb.txt'
    fw = open(file_name, 'w+')
    for i in range(1900, 2017):
        print(i)
        fw.write(str(i) + '\n')
    fw.close()


def build_volume4KB():
    file_name = 'volume_kb.txt'
    fw = open(file_name, 'w+')
    for i in range(200):
        fw.write(str(i) + '\n')
    fw.close()


def save_data(data, output):
    for d in data:
        output.write(d + '\n')
    output.close()


def build_linkedauthor():
    la_output = open('lower_linked_authors_no_punctuation.txt', 'w+')
    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile, events=("end",), load_dtd=True)
    lined_authors_set = fast_iter6(context)

    save_data(lined_authors_set, la_output)


def build_all_pages():
    p_output = open('all_pages.txt', 'w+')
    infile = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile, events=("end",), load_dtd=True)
    lined_authors_set = fast_iter7(context)
    save_data(lined_authors_set, p_output)


# volume(number)
def build_dataset2():
    print('hehe')
    output = open('../testdata/temp_combined_data6.txt', 'w+')
    infile = '/home/himon/PycharmProjects/paper_work1/dataset_workshop/dblp_temp.xml'
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile2, events=("end",), load_dtd=True)
    fast_iter9(context, output, 24000, 28000)
    # fast_iter9(context, output, 0, 1000)
    output.close()

if __name__ == '__main__':
    # build_all_pages()
    # main()
    # buildcorpus4word2vec()
    # build2()
    # build_year4KB()
    # build_volume4KB()
    # build_samples_main()
    # build()
    # build_linkedauthor()
    build_dataset2()
    # build_samples_main2()
