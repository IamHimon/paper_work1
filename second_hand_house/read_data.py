import pymysql
import jieba
import re
from second_hand_house.toolbox import *


def load_data2():
    sources = []
    titles = []
    houseIDs = []
    publish_times = []
    rents = []
    charge_methods = []
    units = []
    rental_models = []
    house_types = []
    decorations = []
    areas = []
    orientations = []
    floors = []
    residential_areas = []
    locations = []
    configurations = []
    contact_persons = []
    phone_numbers = []
    companies = []
    storefronts = []
    describes = []
    urls = []

    connection = pymysql.connect(host="localhost", user="root", password="0099", db='mysql', charset='utf8')
    try:

        with connection.cursor() as cursor2:
            sql = "SELECT 来源,标题,房源编号,发布时间, 租金,押付方式,户型, 租凭方式,房屋类型," \
                  "装修,面积,朝向,楼层,小区, 位置,配置,联系人,联系方式,公司,店面, 房源描述,URL FROM anjuke WHERE id = 68277"
            cursor2.execute(sql)
            result = cursor2.fetchall()
            for row in result:
                if row[0] != '':
                    sources.append(row[0])
                if row[1] != '':
                    titles.append(row[1])
                if row[2] != '':
                    houseIDs.append(row[2])
                if row[3] != '':
                    publish_times.append(row[3])
                if row[4] != '':
                    rents.append(row[4])
                if row[5] != '':
                    charge_methods.append(row[5])
                if row[6] != '':
                    units.append(row[6])
                if row[7] != '':
                    rental_models.append(row[7])
                if row[8] != '':
                    house_types.append(row[8])
                if row[9] != '':
                    decorations.append(row[9])
                if row[10] != '':
                    areas.append(row[10])
                if row[11] != '':
                    orientations.append(row[11])
                if row[12] != '':
                    floors.append(row[12])
                if row[13] != '':
                    residential_areas.append(row[13])
                if row[14] != '':
                    locations.append(row[14])
                if row[15] != '':
                    configurations.append(row[15])
                if row[16] != '':
                    contact_persons.append(row[16])
                if row[17] != '':
                    phone_numbers.append(row[17])
                if row[18] != '':
                    companies.append(row[18])
                if row[19] != '':
                    storefronts.append(row[19])
                if row[20] != '':
                    describes.append(row[20])
                if row[21] != '':
                    urls.append(row[21])
    finally:
        connection.close()

    x_text = titles + houseIDs + publish_times + rents + charge_methods + units + rental_models+house_types+decorations+\
   areas+orientations+floors+residential_areas+locations+configurations+contact_persons+phone_numbers+companies+\
    storefronts+describes+urls

    return x_text


if __name__ == '__main__':
    print('main')
    # x_text = load_data2()
    # print(x_text)
    # print(len(x_text))

