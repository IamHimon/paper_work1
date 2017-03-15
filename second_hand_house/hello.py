
from second_hand_house.toolbox import *
import re


def for_long():
    print('hello')
    l = ['1 AS','1 EF','2 DF','2 IJ','2 FSA','3 OK', '3 DK', '3 IK','4 DK', '4 IJ']
    print(l)
    temp = []
    result = []
    ID = '1'
    for i in range(len(l)):
        if l[i][0] == ID:
            temp.append(l[i])
        else:
            ID = l[i][0]
            result.append(temp)
            temp = []
            temp.append(l[i])
    result.append(temp)
    print("RESULT:",result)


def clear_null(seg_list):
    result = []
    for s in seg_list:
        if s:
            result.append(s)
    return result


def split_list(str_list, stride):
    result = []
    list_length = len(str_list)
    for l in range(list_length):
        if l + stride - 1 == list_length:
            break
        result.append(str_list[l: l + stride])
    return result


def build_all_windows(sample):
    r1 = '\s+'
    r2 = '\,'
    r3 = '\.'
    r = "|".join([r1, r2, r3])
    windows = []
    result = []
    seg_list = clear_null(re.split(r, sample))
    for l in range(1, len(seg_list)):
        temp = split_list(seg_list, l)
        windows.append(temp)

    for win in windows:
        for w in win:
            result.append(w)
    result.append(seg_list)
    return result


def build_all_windows2(sample):
    r1 = '\s+'
    r2 = '\,'
    r3 = '\.'
    r4 = '\#'
    r = "|".join([r1, r2, r3, r4])
    windows = []
    result = []
    str_result = []
    seg_list = clear_null(re.split(r, sample))
    for l in range(1, len(seg_list)):
        temp = split_list(seg_list, l)
        windows.append(temp)

    for win in windows:
        for w in win:
            result.append(w)
    result.append(seg_list)
    # print(result)

    for r in result:
        if len(r) != 1:
            str_result.append(' '.join(r))
        else:
            str_result.append(r[0])
    return str_result


if __name__ == '__main__':
    print('main')
    l = "急租 盘蠡新村 精装2室 轻轨口 家电齐全 拎包入住#41212770#2015年03月26日#1700元/月#付3押1#2室2厅1卫#整租#普通住宅#精装修#80平米#南北#5/5#水香七村#苏州-吴中-龙西#床空调电视冰箱洗衣机热水器宽带可做饭独立卫生间阳台#鲍张洋#137 7191 7123#万腾房产#先奇店#http://su.zu.anjuke.com/fangyuan/41212770?from=Filter_1"
    l2 = l.replace('#', ' ')
    print(l2)
    t = sample_pretreatment_disperse_number2(l2)

    r_raw = remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(l2)))
    print(r_raw)
    wins = build_all_windows2(' '.join(r_raw))
    print(len(wins))
    print(wins)

