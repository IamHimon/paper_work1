import numpy as np
from numpy import random
# label = ['Brand', 'Vehicle', 'Price', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']


USED_CAR_DICT = {'Brand': 0, 'Price': 2, 'Vehicle': 1, 'Odometer': 3, 'Colour': 4, 'Transmission': 5, 'Body': 6, 'Engine': 7, 'Fuel Enconomy': 8}


def value_get_key(value, m_dict):
    for i, v in m_dict.items():
        if v == value:
            return i


def greedy_labeling(label, softmax_loss, y, DICT=USED_CAR_DICT):
    # print('label:')
    # print(label)
    candidate_result = {}
    unknown_indexes = []

    for i, l in enumerate(label):
        if l == 'Unknown':
            unknown_indexes.append(i)
    # print('unknown_indexes:')
    # print(unknown_indexes)

    other_label = []
    for k in DICT.keys():
        if k not in label:
            other_label.append(k)

    # print(other_label)
    right_label = ''
    left_label = ''

    for index in unknown_indexes:
        # print(index)
        p = []
        right_index = index
        while right_index < len(label):
            # print(label[right_index])
            if label[right_index] in DICT.keys():
                # print(label[right_index])
                right_label = label[right_index]
                break
            else:
                right_index += 1
        # print('right label:', right_label)
        p.append(right_label)

        left_index = index
        while left_index >= 0:
            # print(label[right_index])
            if label[left_index] in DICT.keys():
                # print(label[right_index])
                left_label = label[left_index]
                break
            else:
                left_index -= 1
        # print('left label:', left_label)
        p.append(left_label)
        p += other_label

        # print(p)
        p = [i for i in p if i != '']
        # print(p)

        candidate_label = sorted([DICT.get(l) for l in p])
        candidate_result[index] = candidate_label

        # print(candidate_label)
    # print(candidate_result)
    # 递归
    if 'Unknown' in label:
        label = got_probality(candidate_result, softmax_loss, label, y, DICT)
        return greedy_labeling(label, softmax_loss, y, DICT)
    else:
        print('It is over!')
        return label


def get_inde_in_list(value, m_list):
    for i, v in enumerate(m_list):
        if v == value:
            return i


def got_probality(candidate_result, softmax_loss, label, y, DICT=USED_CAR_DICT):

    # print(softmax_loss)

    probility_dict = {}
    for index, key in candidate_result.items():
        sum = np.sum(softmax_loss[index][[k for k in key]])
        probility_dict[index] = softmax_loss[index][[k for k in key]] / sum

    # print(probility_dict)



    max_key = 0
    max_value = 0
    filter_num = 0
    for i, k in probility_dict.items():
        if max(k) <= y:
            filter_num += 1
            label[i] = 'low_quality_label' + str(filter_num)
        if max_value < max(k):
            max_value = max(k)
            max_key = i

    attr_index = max_key
    label_index = candidate_result.get(max_key)[get_inde_in_list(max_value, probility_dict.get(max_key))]
    # print('attr_index:', attr_index)
    # print('label_index:', label_index)
    # print('max_value: ', max_value)

    # print(value_get_key(label_index, USED_CAR_DICT))
    defined_label = value_get_key(label_index, DICT)
    # print(defined_label)

    label[attr_index] = defined_label

    return label


def re_construct_block(block, greedy_label):
    all_sinks = []

    # 产生这样的结果: [[0], [1, 3], [4], [5], [6, 7], [8], [9], [10], [11]]
    i = 0
    while i < len(greedy_label):
        sink = [i]
        j = len(greedy_label)-1
        while j >= i:
            if greedy_label[i] == greedy_label[j]:
                if i != j:
                    sink.append(j)
                    i = j
                    all_sinks.append(sink)
                else:
                    all_sinks.append(sink)
                break
            j -= 1
        if i == j:
            i += 1
    # [[0], [1, 2, 3], [4], [5], [6, 7], [8], [9], [10], [11]]
    all_com_sinks = [[i for i in range(sink[0], sink[-1] + 1)] for sink in all_sinks]

    re_blocks = []
    re_labels = []
    for sink in all_com_sinks:
        b_temp = [block[i] for i in sink]
        re_blocks.append(' '.join(b_temp))
        l_temp = [greedy_label[i] for i in sink]
        re_labels.append(l_temp[0])

    return re_blocks, re_labels


if __name__ == '__main__':

    blcok = ['2015 Audi', '$48999', '8V', 'Cabriolet', '2dr S tronic 7sp 1.4T (CoD) [MY17]', '4200', 'Mythos', 'Black', '7 speed Automatic', '2 doors 4 seats Convertible', '4 cylinder Petrol - Premium ULP Turbo Intercooled Turbo IntercooledL', '5.1 (L/100km)']
    label = ['Unknown', 'Brand', 'Unknown', 'Unknown', 'Vehicle', 'Unknown', 'Unknown', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']

    g_label1 = ['Brand', 'Price', 'low_quality_label1', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']
    g_label2 = ['Brand', 'Price', 'low_quality_label1', 'low_quality_label2', 'Vehicle', 'Odometer', 'Colour', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']

    blcok_label_result = re_construct_block(blcok, g_label2)
    print(blcok_label_result)
    softmax_loss = random.rand(9, 9)
    result_label = greedy_labeling(label, softmax_loss, 0.3)
    print(result_label)




    # label = got_probality(candidate_result, softmax_loss, label, 0.8)
    # print(label)


