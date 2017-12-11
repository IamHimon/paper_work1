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


def data_aug_fe():
    fe = 0.0
    for i in range(150):
        fe += 0.1
        print(str('%.1f' % (fe)) + ' (L/100km)')


def my_filter(x):
    return str(x).replace(',', '')


def split_brand(x):
    if pd.isnull(x):
        return x
    else:
        return ' '.join(x.split()[:2])


def build_test_train_data():

    filename = 'data/cars3.txt'
    # filename2 = 'data/data_bmw.txt'
    records = load_car_data(filename)
    names = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']
    df = pd.DataFrame(records, columns=names)
    shuffle_df = df.sample(frac=1).reset_index(drop=True)

    shuffle_df['Price'] = shuffle_df['Price'].apply(lambda x: str(x).replace(',', ''))
    # print(shuffle_df['Brand'])
    shuffle_df['Brand'] = shuffle_df['Brand'].apply(lambda x: split_brand(x))
    # print(shuffle_df['Odometer'])
    # shuffle_df['Odometer'] = shuffle_df['Odometer'].dropna().apply(lambda x: int(x))
    # print(shuffle_df['Odometer'])


    test_df = shuffle_df[:100]
    print(test_df)
    train_df = shuffle_df[100:]
    print(len(test_df.index))
    print(len(train_df.index))
    #
    test_df.to_csv('data/test_data_split_brand.txt', index=False, header=False)
    train_df.to_csv('data/train_data_split_brand.txt', index=False, header=False)


if __name__ == '__main__':
    print(build_test_train_data())
    # KB = load_kb_us()
    # # print(KB)
    # names = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']
    # print(KB.get('Odometer'))



    # 测试 block 用test_data
    #
    # test_df = pd.read_csv('data/test_data_split_brand.txt', names=names).dropna()
    # # print(test_df[:1])
    # test_df['Odometer'] = test_df['Odometer'].apply(lambda x: str(x))
    # # print(test_df[:1])
    # # train_df = pd.read_csv('data/train_data.txt')
    # # print(test_df)
    # # print(train_df)
    # # df = pd.DataFrame(records, columns=names).dropna()
    # test_samples = []
    # count = 0
    # for i in test_df.values:
    #     count += 1
    #     # print(i)
    #     # print(str(count) + '\t' + ','.join(i))
    #     record = ','.join(i)
    #     print(record)
    #     blocks, anchors = doBlock5(record, KB, USED_CAR_DICT, threshold=0.75)
    #     print(blocks)
    #     print(anchors)
    #     re_blocks, re_anchors = re_block(blocks, anchors)
    #     print(re_blocks)
    #     print(re_anchors)
    #     # if len_Unknown2(re_anchors, USED_CAR_DICT):
        #        for result in do_blocking2(re_blocks, re_anchors, len(USED_CAR_DICT), USED_CAR_DICT):
        #             print('result:', result)
        # else:
        #    print((re_blocks, re_anchors))

        # test_samples.append(str(count) + '\t' + ','.join(i))
