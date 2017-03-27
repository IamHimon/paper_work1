from lxml import etree

CATEGORIES = {'article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', "mastersthesis", "www"}
SKIP_CATEGORIES = {'phdthesis', 'mastersthesis', 'www'}
DATA_ITEMS = ["title", "year", "journal", "ee", "year", "volume", "pages"]


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
            if record_count <= boundary2:
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

            if record_count >= boundary1:
                for author in authors:
                    author_part += author + ','
                sample = author_part + paper["title"].text + ',' + paper["journal"].text + ',' + paper["year"].text \
                         + ',' + paper["volume"].text + ',' + paper["pages"].text
                print(sample)
                output.write(sample + '\n')
                author_part = ''
            if record_count == boundary3:
                break
    return title_set, linked_author_set, journal_set, pages_set


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


if __name__ == '__main__':
    build_all_pages()
    # main()
    # buildcorpus4word2vec()
    # build2()
    # build_year4KB()
    # build_volume4KB()
    # build_samples_main()
    # build()
    # build_linkedauthor()
