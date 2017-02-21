from lxml import etree

CATEGORIES = {'article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', "mastersthesis", "www"}
SKIP_CATEGORIES = {'phdthesis', 'mastersthesis', 'www'}
DATA_ITEMS = ["title", "year", "journal", "ee"]


def clear_element(element):
    element.clear()
    while element.getprevious() is not None:
        del element.getparent()[0]


def extract_paper_elements(context):
    for event, element in context:
        if element.tag in CATEGORIES:
            yield element
            clear_element(element)


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

# def split_all_taj():


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


def build_cleared_corpus4word2vec():
    output = open("cleared_corpus4word2vec.txt", 'w+')
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    context = etree.iterparse(infile2, events=("end",), load_dtd=True)  # 生成迭代器
    title_set = set()
    author_set = set()
    journal_set = set()


if __name__ == '__main__':
    # main()
    # buildcorpus4word2vec()
    build()
