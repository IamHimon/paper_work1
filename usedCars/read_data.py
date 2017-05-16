import sys
sys.path.append('..')
import json
import pandas as pd
from pandas import DataFrame, Series
from blocking.block import *
from usedCars.tools import *

label = ['Vehicle', 'Price', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']


def load_car_data(filename):
    fo = open(filename, 'r')
    lines = fo.readlines()
    temp_label = ''
    records = []
    temp_dict = {}
    for line in lines:
        # print(line)

        if line.strip() == '':
            # print(temp_dict)
            if temp_dict:
                records.append(temp_dict)
            # count += 1
            temp_dict = {}
            pass
        else:
            if line.strip() in label:
                temp_label = line.strip()
            else:
                if temp_label == '':
                    # print('Brand: ' + line.strip())
                    temp_dict['Brand'] = line.strip()
                    # print(line.strip())
                else:
                    # print(temp_label + ': ' + line.strip())
                    temp_dict[temp_label] = line.strip()
                    # print(line.strip())
                    temp_label = ''
    return records




if __name__ == '__main__':
    KB = load_kb_us()
    filename = 'cars3.txt'
    filename2 = 'data/data_bmw.txt'
    records = load_car_data(filename2)
    names = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']
    df = pd.DataFrame(records, columns=names).dropna()
    test_samples = []
    count = 0
    for i in df.values:
        count += 1
        # print(str(count) + '\t' + ','.join(i))

        record = ','.join(i)
        blocks, anchors = doBlock5(record, KB, USED_CAR_DICT, threshold=0.5)
        print(blocks)
        print(anchors)
        re_blocks, re_anchors = re_block(blocks, anchors)
        print(re_blocks)
        print(re_anchors)
        if len_Unknown2(re_anchors, USED_CAR_DICT):
                for result in do_blocking2(re_blocks, re_anchors, len(USED_CAR_DICT), USED_CAR_DICT):
                    print('result:', result)
        else:
            print((re_blocks, re_anchors))

        # test_samples.append(str(count) + '\t' + ','.join(i))

    # df.to_csv('cars.csv')
# print(records)

    # for r in records:
    #     print(r)

# print(len(records))
# print(json)