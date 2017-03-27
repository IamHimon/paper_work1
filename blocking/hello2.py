from itertools import combinations, permutations
# from blocking.reconstruction import *


def get_block_value(s, blocks):
    result = [blocks[i] for i in range(len(blocks)) if i in s]
    return ' '.join(result)


def get_label_value(s, labels, back_sink_sign):
    # print(s)
    if s in back_sink_sign.values():
        return vale_find_key(s, back_sink_sign)
    else:
        return labels[s[0]]


def vale_find_key(value, my_dict):
    for k, v in my_dict.items():
        if value == v:
            return k


def re_organize_bolckandlabel(blocks, labels, rebuild_sink, rest_backup_sink, sign_label_dict):
    # 链接成一个记录
    re_block_index = sorted(rebuild_sink + [rest_backup_sink])
    # print(re_block_index)
    # 构造backup_sink和sign_label的dict
    re_block = []
    re_label = []
    for bi in re_block_index:
        # print(get_label_value2(bi, labels, back_sink_sign))
        re_block.append(get_block_value(bi, blocks))
        re_label.append(get_label_value(bi, labels, sign_label_dict))
    return re_block, re_label


if __name__ == '__main__':
    blocks = ['Dominique Fournier', 'Crémilleux', 'A quality', 'pruning.', 'Knowl.-Based Syst.', '2002', '15', '37-43']
    labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Year', 'Unknown', 'Pages']

    l = [[0], [3], [5], [7]]
    l2 = [[0], [3], [4], [5], [7]]
    backup_sinks = [[1, 2], [4], [6]]
    backup_sinks2 = [[1, 2], [6]]

    rebuild_block = []
    sign = 0
    sign_label = [str(i) + '_Backup_Unknown' for i in range(6 - len(l))]
    # print(sign_label)
    for bs in combinations(backup_sinks, 6 - len(l)):
        print('backup_sink:', bs)
        rest_backup_sink = [b for b in backup_sinks if b not in bs]
        # print('rest_backup_sink:', rest_backup_sink)
        rebuild_sink = l + [b for b in bs]
        print((rebuild_sink, sorted(sum(rest_backup_sink, []))))
        # print(sorted(sum(rest_backup_sink, [])))
        sign_label_dict = {}
        for i in range(len(sign_label)):
            sign_label_dict[sign_label[i]] = bs[i]
        # print(sign_label_dict)
        re_block, re_label = re_organize_bolckandlabel(blocks, labels, rebuild_sink, sorted(sum(rest_backup_sink, [])), sign_label_dict)
        print(re_block)
        print(re_label)


        print('===========')


