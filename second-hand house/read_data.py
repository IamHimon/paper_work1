import pymysql
import jieba
connection = pymysql.connect(host="localhost", user="root", password="0099", db='mysql', charset='utf8')
try:

    with connection.cursor() as cursor2:
        sql = "SELECT 标题, 租金, 户型, 位置, 房源描述 FROM anjuke WHERE id < 10"
        cursor2.execute(sql)
        result = cursor2.fetchall()
        for row in result:
            source = row[0]
            rent = row[1]
            unit = row[2]
            location = row[3]
            Describe = row[4]
            # Title = row[0]
            # HouseID = row[3]
            # Fetch_time = row[4]
            # Publish_time = row[5]
            # area = row[12]
            # URL = row[25]
            print("source=%s, rent=%s, unit=%s, location=%s, Describe=%s" % (source, rent, unit, location, Describe))
finally:
    connection.close()

str = "合景二期毛坯三房低价出租 房间方正干净整洁 周边生活设施齐全,"
seg_list = jieba.cut(str.replace(' ', ''))
print("Full Mode: " + "/ ".join(seg_list))
