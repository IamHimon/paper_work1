import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
from blocking.block import *

# Brand,Price,Vehicle,Odometer,Colour,Transmission,Body,Engine,Fuel Enconomy

USED_CAR_DICT = {'Brand': 0, 'Price': 1, 'Vehicle': 2, 'Odometer': 3, 'Color': 4, 'Transmission': 5, 'Body': 6, 'Engine': 7, 'Fuel_enconomy': 8}
label = ['Vehicle', 'Price', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']


def build_y_train_used_car_all_attribute(Brand, Price, Vehicle,  Odometer, Colour, Transmission, Body, Engine, Fuel_enconomy):
    # print("Building label dict:")
    Brand_labels = [0 for i in range(len(Brand))]
    Price_labels = [1 for i in range(len(Price))]
    Vehicle_labels = [2 for i in range(len(Vehicle))]
    Odometer_labels = [3 for i in range(len(Odometer))]
    Colour_labels = [4 for i in range(len(Colour))]
    Transmission_labels = [5 for i in range(len(Transmission))]
    Body_labels = [6 for i in range(len(Body))]
    Engine_labels = [7 for i in range(len(Engine))]
    Fuel_enconomy_labels = [8 for i in range(len(Fuel_enconomy))]

    y_t = Brand_labels + Price_labels + Vehicle_labels + Odometer_labels + Colour_labels + Transmission_labels + \
          Body_labels + Engine_labels + Fuel_enconomy_labels
    label_dict_size = len(USED_CAR_DICT)

    y_train = np.zeros((len(y_t), label_dict_size))
    for i in range(len(y_t)):
        y_train[i][y_t[i]] = 1
    # print("Preparing y_train over!")
    return y_train, label_dict_size


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


def load_kb_us():
    filename = 'cars3.txt'
    records = load_car_data(filename)
    names = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']
    df = pd.DataFrame(records, columns=names)
    # print(df)
    Brand = df['Brand'].dropna().values.tolist()
    Price = df['Price'].dropna().values.tolist()
    Vehicle = df['Vehicle'].dropna().values.tolist()
    Odometer = df['Odometer'].dropna().values.tolist()
    Colour = df['Colour'].dropna().values.tolist()
    Transmission = df['Transmission'].dropna().values.tolist()
    Body = df['Body'].dropna().values.tolist()
    Engine = df['Engine'].dropna().values.tolist()
    Fuel_enconomy = df['Fuel Enconomy'].dropna().values.tolist()

    KB = {'Brand': Brand, 'Price': Price, 'Vehicle': Vehicle, 'Odometer': Odometer, 'Colour': Colour, 'Transmission': Transmission,
          'Body': Body, 'Engine': Engine, 'Fuel Enconomy': Fuel_enconomy}
    # print(KB)
    return KB


if __name__ == '__main__':
    # print(len(USED_CAR_DICT))
    KB = load_kb_us()
    # print(KB)
    l1 = '2016 Mazda 3 Sedan Touring,$29335,BN Series Touring Sedan 4dr SKYACTIV-Drive 6sp 2.0i [May],0,Soul Red,6 speed Automatic,4 doors 5 seats Sedan,4 cylinder Petrol - Unleaded ULP Aspirated AspiratedL,5.7 (L/100km)'
    l2 = '2016 Mazda 3 Sedan Touring'
    blocks, anchors = doBlock5(l1, KB, USED_CAR_DICT, threshold=0.95)
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

