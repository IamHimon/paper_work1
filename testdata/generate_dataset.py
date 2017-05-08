from lxml import etree
import xml.dom.minidom
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


def fast_iter(context, xml_write, output, boundary1, boundary2):
    # create xml
    doc = xml.dom.minidom.Document()
    root = doc.createElement('Articles')
    root.setAttribute('Source', 'dblp')
    doc.appendChild(root)
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

            authors = [author.text for author in element.findall("author")]
            if authors and ('title' in paper.keys()) and (paper['title'].text is not None) and ("journal" in paper.keys()) and (paper["journal"].text is not None)\
                    and ('year' in paper.keys()) and (paper['year'].text is not None) and ('volume' in paper.keys()) and ('number' in paper.keys()) and (paper['volume'].text is not None) and \
                    (paper['number'].text is not None) and ('pages' in paper.keys()) and (paper['pages'].text is not None)\
                    and ('e' not in paper['pages'].text):
                record_count += 1
                nodeArticle = doc.createElement('Article')
                node_title = doc.createElement('title')
                node_author = doc.createElement('author')
                node_journal = doc.createElement('journal')
                node_year = doc.createElement('year')
                node_page = doc.createElement('page')
                node_volume = doc.createElement('volume')
                node_record_ID = doc.createElement('record_ID')

                if len(authors) == 1:
                    taj_record_list.append(authors[0])
                    node_author.appendChild(doc.createTextNode(authors[0]))
                else:
                    taj_record_list.append(','.join(authors[:-1]) + ' and ' + authors[-1])
                    node_author.appendChild(doc.createTextNode(','.join(authors[:-1]) + ' and ' + authors[-1]))

                taj_record_list.append(paper['title'].text.strip('.'))
                node_title.appendChild(doc.createTextNode(paper['title'].text.strip('.')))
                taj_record_list.append(paper["journal"].text.strip('.'))
                node_journal.appendChild(doc.createTextNode(paper["journal"].text.strip('.')))
                yvp_record_list.append(paper['year'].text)
                node_year.appendChild(doc.createTextNode(paper['year'].text))

                yvp_record_list.append(paper['volume'].text + '(' + paper['number'].text + ')')
                node_volume.appendChild(doc.createTextNode(paper['volume'].text + '(' + paper['number'].text + ')'))
                yvp_record_list.append(paper['pages'].text)
                node_page.appendChild(doc.createTextNode(paper['pages'].text))

                node_record_ID.appendChild(doc.createTextNode(str(record_count)))

                nodeArticle.appendChild(node_title)
                nodeArticle.appendChild(node_author)
                nodeArticle.appendChild(node_journal)
                nodeArticle.appendChild(node_year)
                nodeArticle.appendChild(node_page)
                nodeArticle.appendChild(node_volume)
                nodeArticle.appendChild(node_record_ID)
                root.appendChild(nodeArticle)

                random.shuffle(taj_record_list)
                random.shuffle(yvp_record_list)
                result_record = taj_record_list + yvp_record_list
                print(','.join(result_record))
                # print(record_count)
                output.write(str(record_count) + '\t' + ','.join(result_record) + '\n')

        if paperCounter == boundary2:
            break

    doc.writexml(xml_write, indent='\t', addindent='\t', newl='\n', encoding='utf-8')


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


def build_dataset():
    print('hehe')
    output = open('../testdata/temp_combined_data7.txt', 'w+')
    infile2 = '/home/himon/Jobs/paper_work1/dblp.xml'
    xml_write = open('test_data.xml', 'w')

    context = etree.iterparse(infile2, events=("end",), load_dtd=True)
    # fast_iter9(context, output, 24000, 25500)
    fast_iter(context, xml_write, output, 24000, 25500)

    output.close()
    xml_write.close()


if __name__ == '__main__':
    # build_dataset()
    fo = open('../testdata/temp_combined_data7.txt', 'r')
    lines = fo.readlines()
    for line in lines:
        # print(line.strip())
        # l1 = line.strip().split('\t')[-1]
        # l0 = line.strip().split('\t')[0]
        # print(l1)
        # print(l0)
        print(len(line.strip().split('\t')))
        if len(line.strip().split('\t')) < 2:
            print('eroo!@!!!!!!!!!!!!!')
            break
