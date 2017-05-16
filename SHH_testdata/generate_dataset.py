from second_hand_house.toolbox import *
import pymysql
import jieba
import xml.dom.minidom


def write_set(filename, my_set):
    fw = open(filename, 'w+')
    for i in my_set:
        fw.write(i + '\n')
    fw.close()


def loadKB_SHH(COUNT=10000):
    titles = set()
    publish_times = set()
    rents = set()
    charge_methods = set()
    units = set()
    areas = set()
    floors = set()
    configurations = set()
    connection = pymysql.connect(host="localhost", user="root", password="0099", db='mysql', charset='utf8')
    try:

        with connection.cursor() as cursor2:
            sql = "SELECT 标题,发布时间, 租金,押付方式,户型," \
                    "面积,楼层,配置 FROM anjuke WHERE id < %d" % COUNT
            cursor2.execute(sql)
            result = cursor2.fetchall()
            for row in result:
                if row[0] != '':
                    titles.add(' '.join(remove_black_space(jieba.lcut(row[0]))))
                if row[1] != '':
                    publish_times.add(' '.join(remove_black_space(jieba.lcut(row[1]))))
                if row[2] != '':
                    rents.add(' '.join(remove_black_space(jieba.lcut(row[2]))))
                if row[3] != '':
                    charge_methods.add(' '.join(remove_black_space(jieba.lcut(row[3]))))
                if row[4] != '':
                    units.add(' '.join(remove_black_space(jieba.lcut(row[4]))))
                if row[5] != '':
                    areas.add(' '.join(remove_black_space(jieba.lcut(row[5]))))
                if row[6] != '':
                    floors.add(' '.join(remove_black_space(jieba.lcut(row[6]))))
                if row[7] != '':
                    configurations.add(' '.join(remove_black_space(jieba.lcut(row[7]))))
    finally:
        connection.close()
    KB = {'titles': titles, 'publish_time': publish_times, 'rent': rents, 'charge_method': charge_methods,
          'unit': units, 'area': areas, 'floor': floors, 'configuration': configurations}

    write_set('title_set.txt', titles)
    return KB


def fast_iter(dataset_output, xml_output, count):
    # CREATE XML
    doc = xml.dom.minidom.Document()
    root = doc.createElement('Houses')
    root.setAttribute('source', 'anjuke')
    doc.appendChild(root)

    connection = pymysql.connect(host="localhost", user="root", password="0099", db='mysql', charset='utf8')
    record_count = 0
    try:
        with connection.cursor() as cursor2:
            sql = "SELECT 标题,发布时间, 租金,押付方式,户型," \
                    "面积,楼层,配置 FROM anjuke WHERE id >= 30000"
            cursor2.execute(sql)
            result = cursor2.fetchall()
            for row in result:
                temp = []
                if (row[0] != '') and (row[1] != '') and (row[2] != '') and (row[3] != '') and (row[4] != '') \
                        and (row[5] != '') and (row[6] != '') and (row[7] != ''):
                    record_count += 1
                    # create combined text record
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[0]))))
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[1]))))
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[2]))))
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[3]))))
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[4]))))
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[5]))))
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[6]))))
                    temp.append(' '.join(remove_black_space(jieba.lcut(row[7]))))
                    print(','.join(temp))
                    dataset_output.write(str(record_count) + '\t' + ','.join(temp) + '\n')

                    # create template
                    nodeHouse = doc.createElement('House')
                    node_title = doc.createElement('title')
                    node_publish_t = doc.createElement('publish_t')
                    node_rent = doc.createElement('rent')
                    node_charge_m = doc.createElement('charge_m')
                    node_unit = doc.createElement('unit')
                    node_area = doc.createElement('area')
                    node_floor = doc.createElement('floor')
                    node_conf = doc.createElement('conf')
                    node_ID = doc.createElement('ID')

                    node_title.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[0])))))
                    node_publish_t.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[1])))))
                    node_rent.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[2])))))
                    node_charge_m.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[3])))))
                    node_unit.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[4])))))
                    node_area.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[5])))))
                    node_floor.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[6])))))
                    node_conf.appendChild(doc.createTextNode(' '.join(remove_black_space(jieba.lcut(row[7])))))
                    node_ID.appendChild(doc.createTextNode(str(record_count)))

                    nodeHouse.appendChild(node_title)
                    nodeHouse.appendChild(node_publish_t)
                    nodeHouse.appendChild(node_rent)
                    nodeHouse.appendChild(node_charge_m)
                    nodeHouse.appendChild(node_unit)
                    nodeHouse.appendChild(node_area)
                    nodeHouse.appendChild(node_floor)
                    nodeHouse.appendChild(node_conf)
                    nodeHouse.appendChild(node_ID)
                    root.appendChild(nodeHouse)

                if record_count > count:
                    break
    finally:
        connection.close()

    doc.writexml(xml_output, indent='\t', addindent='\t', newl='\n', encoding='utf-8')


def build_dataset():
    print('hehe')
    dataset_output = open('shh_combined_data1.txt', 'w+')
    xml_write = open('shh_template_data.xml', 'w')
    fast_iter(dataset_output, xml_write, 1500)
    dataset_output.close()
    xml_write.close()

if __name__ == '__main__':
    KB = loadKB_SHH()
    print(KB)
    # build_dataset()
