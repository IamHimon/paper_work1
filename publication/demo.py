from second_hand_house.toolbox import *
# import tensorflow as tf
# from publication.tools import *
import re
import numpy as np
from utils import *

def group_by_element(lst):
    '''基本思路是先取得不同元素起始的索引值，
    再按照这个索引值取切片
    '''
    index = []
    result = []
    for i, _ in enumerate(lst):
        if i < len(lst) - 1 and lst[i + 1] != lst[i]:
            index.append(i + 1)

    result.append(lst[:index[0]])
    for i, item in enumerate(index):
        if i < len(index) - 1:
            result.append(lst[index[i]:index[i + 1]])
    result.append(lst[item:])
    return result


def my_index(item, my_list):
    indexes = []
    for i, e in enumerate(my_list):
        if item == e:
            indexes.append(i)
    return indexes


# block = np.arrange(len(blocks))
def all_revise_predictions2(predictions, loss, block):
    copy_predictions = prediction.copy()
    # print(copy_predictions)
    max_temp = -10000
    max_index = 0
    for p in predictions:
        # print(predictions)
        # print(p)
        index = np.where(copy_predictions == p)
        # print(index[0])
        if len(index[0]) > 1:
            for i in index[0]:
                if max_temp < loss[i][p]:
                    max_temp = loss[i][p]
                    max_index = i
            # print('max_index:', str(max_index))
            # print('rest label:')
            # print(make_rest_label(copy_predictions, block))
            for i in index[0]:
                if i != max_index:
                    copy_predictions[i] = make_rest_label(copy_predictions, block)[0]
        # else:
        #     continue
        # print(copy_predictions)
        if (np.sort(copy_predictions) == block).all():
            return copy_predictions


def make_rest_label(predictions, block):
    rest_label = []
    for b in block:
        if b not in predictions:
            rest_label.append(b)
    return rest_label


if __name__ == '__main__':

    loss  =  [[4.09746729e-03,  9.95899260e-01,  2.57667261e-06,  1.68470446e-08,    4.67520467e-09,   7.21266986e-07],
             [1.26743168e-01,  1.01604406e-02,  8.62519741e-01,  7.18588126e-05,    7.47017839e-06,   4.97338886e-04],
             [9.15050405e-05,  9.99908090e-01,  1.11127729e-08,  3.90859363e-08,    1.72063281e-07,   1.75976069e-07],
             [2.25420056e-08,  1.21055148e-08,  3.49254492e-20,  1.36265155e-16,    1.00000000e+00,   3.04593195e-09],
             [2.43220999e-10,  1.32002658e-13,  5.45930668e-17,  2.76123450e-16,    1.05543933e-11,   1.00000000e+00],
             [3.65529024e-10,  1.47016982e-11,  9.13290575e-14,  2.33751241e-10,    9.99999881e-01,    1.36062013e-07]]

    print('hehe')
    #         #       0                   1               2                   3               4                   5
    loss1  =  [[3.14047247e-01,   6.85949445e-01,   4.89102518e-08,   1.79532847e-07,   3.04309106e-06,   5.45771712e-08],
             [1.43391253e-06,   9.99998093e-01,   4.57024470e-07,   2.42702886e-10,   9.54997970e-10,   7.36000094e-10],
             [1.26743168e-01,   1.01604406e-02,   8.62519741e-01,   7.18588126e-05,   7.47017839e-06,   4.97338886e-04],
             [9.77953096e-09,   6.51602217e-10,   4.93727021e-12,   9.99995947e-01,   6.12361051e-09,   4.01748457e-06],
             [2.54595820e-02,   2.62733679e-02,   8.40937018e-01,   2.79799942e-02,   2.03690249e-02,   5.89810945e-02],
             [9.84512055e-11,   1.57443954e-15,   5.78188501e-16,   8.24614931e-20,   1.16112892e-12,   1.00000000e+00]]

    loss2 = [[  3.18797166e-09,   1.00000000e+00,   2.22347489e-17,   5.99956528e-15,
    3.14311269e-16,   6.81803247e-15],
 [  9.51207112e-05,   9.99902487e-01,   1.66898099e-06,   1.55570504e-07,
    2.61838295e-07,   3.61113251e-07],
 [  1.49044703e-04,   5.33596847e-08,   9.99850988e-01,   9.09468001e-10,
    1.17529253e-09,   4.40743548e-08],
 [  8.74410677e-09,   1.52488440e-08,   8.83775690e-19,   1.80787324e-16,
    1.00000000e+00,   3.91094018e-10],
 [  2.55180165e-12,   2.78238196e-12,   2.03912247e-17,   1.00000000e+00,
    6.31268041e-14,   6.24503249e-09],
 [  1.11689336e-09,   5.54886595e-13,   4.90244035e-15,   2.24908279e-18,
    5.66776417e-11,   1.00000000e+00]]

    block = np.arange(6)
    # print(block)

    loss = np.array(loss2)
    # score = tf.reduce_sum(loss_max)
    # t_p = np.transpose(p)
    # print(t_p)
    # t_prediction = t_p.argmax(1)
    # t_score = t_p.max(1)
    # print(t_prediction)
    # print(t_score)

    # print(make_rest_label(prediction, block))

    # print(prediction)
    revise_predictions, score = greddy_predictions(loss, block)
    # revise_pre = [str(p) for p in revise_predictions]
    print(revise_predictions)
    # print('revise:'+'\t'+'[' + ', '.join(revise_predictions) + ']' + '\n')
    print(score)
    print(np.sum(score))

