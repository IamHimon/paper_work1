import jieba
import re
from second_hand_house.toolbox import *

if __name__ == '__main__':

    str9 = '2015年03月30日'
    str0 = 'http://su.zu.anjuke.com/fangyuan/40379293'
    str1 = '2015-03-30'
    str2 = '41435034'
    str3 = '1 1 /2 4 '
    str4 = '18862148294'
    str5 = '75平米'
    str7 = '苏州-沧浪-友新 '
    str8 = '8000朗诗四房含物业含车位真实房源，年后诚意出租'
    r = sample_pretreatment_disperse_number(str8)
    seg_list = jieba.lcut(r)
    print(remove_black_space(seg_list))
    # seg_list = jieba.lcut(str0.strip())
    # print(seg_list)
    # print("Full Mode: " + "  ".join(seg_list))
    # disperse_number('2015-03-30')
    # disperse_number('41435034')

    # l = [c for c in str2]
    # print(' '+' '.join(l)+' ')
    # r = sample_pretreatment_disperse_number(str0)
    # print(r)
    # seg_list = jieba.lcut(r)
    # print(seg_list)
    # print()
    #
    #
    # print(jieba.lcut(str0))
    #
    # a = ['a', 'b', 'c', 'd', ' ', ' ', 'd']
    # c = []
    # for i in a:
    #     if i != ' ':
    #         c.append(i)
    # print(c)
    # ra = a.remove('d')
    # print(ra)
