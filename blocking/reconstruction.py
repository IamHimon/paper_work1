import operator
import math
from blocking.block import *
from itertools import combinations, permutations


#判断是不是labels只有LABEL_DICT中的key
def len_ex_Unknown(labels):
    count = 0
    for l in labels:
        if l in LABEL_DICT.keys():
            count += 1
    return count


def remove_duplicate(combined_sinks):
    news_ids = []
    for id in combined_sinks:
        if id not in news_ids:
            news_ids.append(id)
    return news_ids


# sinks:[[[],[],[]...], [[],[],[],..], [[],[],[],,]]
def combine_all_sinks(u, all_sinks):
    # print('u:', u)
    # print('all_sinks:', all_sinks)
    combined_sink = []
    for sinks in all_sinks:
        available_sink_index = []
        for s in range(len(sinks)):
            # print(s)
            # print(sinks[s])
            if judge_if_neighbour(u, sinks[s]):
                available_sink_index.append(s)
        # print('available_sink_index:', available_sink_index)
        temp = []
        for i in available_sink_index:
            for s in range(len(sinks)):
                if s == i:
                    temp.append(sorted(sinks[s] + [u]))
                else:
                    temp.append(sinks[s])
            combined_sink.append(temp)
            temp = []

    return combined_sink


# sink: [1,2]
def judge_if_neighbour(u, sink):
    # print('judge:')
    # print(u)
    # print(sink)
    # print(sink[0])
    # print(sink[-1])
    if (u not in sink) and (abs(sink[0]-u) == 1) or (abs(sink[-1]-u) == 1):
        return True
    else:
        return False


def remove_Unknown(label, unknown_list):
    return [l for l in label if l not in unknown_list]


def reblock_according_sinks(blocks, labels, sinks, unknown_list):
    # print('reblock_according_sinks blocks:', blocks)
    # print('reblock_according_sinks label:', labels)
    # print('reblock_according_sinks sinks:', sinks)
    re_blocks = []
    re_labels = []
    for sink in sinks:
        # print(sink)
        b_temp = [blocks[i] for i in sink]
        # print(b_temp)
        re_blocks.append(' '.join(b_temp))

        l_temp = [labels[i] for i in sink]
        re_labels.append(remove_Unknown(l_temp, unknown_list)[0])
        # print(b_temp)
        # print(l_temp)
    return re_blocks, re_labels


def normal_reblock_and_relabel(blocks, labels, unknown_list):
    # print(labels)
    nlabel_dict = {}
    unknown_indexes = []
    for i in range(len(labels)):
        if labels[i] in unknown_list:
            unknown_indexes.append(i)
        else:
            nlabel_dict[labels[i]] = i

    # print(nlabel_dict)

    sorted_x = sorted(nlabel_dict.items(), key=operator.itemgetter(1))
    # print(sorted_x)
    anchor_indexes = [anchor[1] for anchor in sorted_x]
    # print(anchor_indexes)
    # print('unknown_indexes:', unknown_indexes)

    all_sinks = [[[a] for a in anchor_indexes]]
    # print('all_sinks:', all_sinks)
    # reblock_and_relabel(blocks, labels, all_sinks, unknown_indexes)

    combined_sinks1 = all_sinks
    combined_sinks2 = all_sinks
    for u in unknown_indexes:
        combined_sinks1 = combine_all_sinks(u, combined_sinks1)
    for u in reversed(unknown_indexes):
        combined_sinks2 = combine_all_sinks(u, combined_sinks2)

    combined_sinks = remove_duplicate(combined_sinks1 + combined_sinks2)

    # print('combined_sinks:', combined_sinks)
    # yield all block and label according combined_sinks_index
    result = []
    for sinks in combined_sinks:
        # print(sinks)
        re_blocks, re_labels = reblock_according_sinks(blocks, labels, sinks, unknown_list)
        # print(re_blocks)
        # print(re_labels)
        result.append((re_blocks, re_labels))
        # yield (re_blocks, re_labels)
    return result


# unknown_indexes: [1, 2, 6]
# all_sinks: [[0], [3], [4], [5], [7]]
def reblock_and_relabel(blocks, labels, all_sinks, unknown_indexes):
    # print('blocks:', blocks)
    # print('labels:', labels)
    # print('re all_sinks:', all_sinks)
    # print('re unknown_indexes:', unknown_indexes)
    combined_sinks1 = all_sinks
    combined_sinks2 = all_sinks
    for u in unknown_indexes:
        combined_sinks1 = combine_all_sinks(u, combined_sinks1)
    for u in reversed(unknown_indexes):
        combined_sinks2 = combine_all_sinks(u, combined_sinks2)

    combined_sinks = remove_duplicate(combined_sinks1 + combined_sinks2)
    print('combined_sinks:', combined_sinks)

    # yield all block and label according combined_sinks_index
    for sinks in combined_sinks:
        # print(sinks)
        re_blocks, re_labels = reblock_according_sinks(blocks, labels, sinks, [])
        # print(re_blocks)
        # print(re_labels)
        yield (re_blocks, re_labels)


def get_label_value(s, labels, back_sink_sign):
    if s in back_sink_sign.values():
        return vale_find_key(s, back_sink_sign)
    else:
        return labels[s[0]]


def vale_find_key(value, my_dict):
    for k, v in my_dict.items():
        if value == v:
            return k


# just reconstruct label sequence
def re_organize_label(labels, rebuild_sink, rest_backup_sink, sign_label_dict):
    rest_backup_sink = sorted(sum(rest_backup_sink, []))
    # 链接成一个记录
    # re_block_index = []
    # print(rebuild_sink)
    # print(rest_backup_sink)
    if rest_backup_sink:
        re_block_index = sorted(rebuild_sink + [[r] for r in rest_backup_sink])
    else:
        re_block_index = sorted(rebuild_sink)
    # print(re_block_index)
    # 构造backup_sink和sign_label的dict
    re_label = []
    for bi in re_block_index:
        # print(get_label_value2(bi, labels, back_sink_sign))
        re_label.append(get_label_value(bi, labels, sign_label_dict))
    return re_label


# unlabeled_block_count:在LABEL_DICT中的个数
def do_blocking(blocks, labels, target_block_count):
    # print('target_block_count:', target_block_count)
    do_blocking_result = []
    label_dict = {}
    unknown_indexes = []
    unknown_list = []
    for i, l in enumerate(labels):
        if l not in LABEL_DICT.keys():
            unknown_list.append(l)
            unknown_indexes.append(i)
        else:
            label_dict[labels[i]] = i

    # print('label_dict:', label_dict)
    sorted_x = sorted(label_dict.items(), key=operator.itemgetter(1))
    # print('sorted_x:', sorted_x)
    anchor_indexes = [anchor[1] for anchor in sorted_x]
    # print('anchor_indexes:', anchor_indexes)
    # print('unknown_indexes:', unknown_indexes)   # 值肯定是递增排序的
    # print('unknown_list:', unknown_list)

    all_sinks = [[a] for a in anchor_indexes]
    # print('all_sinks:', all_sinks)

    # bachkup_Unknown
    sign_label = [str(i) + '_Backup_' + unknown_list[i] for i in range(target_block_count - len(all_sinks))]
    # print(sign_label)
    if len(unknown_indexes) == 0:
        return
    if len(all_sinks) < target_block_count:
        backup_sinks = [[u] for u in unknown_indexes]
        # print('backup_sinks', backup_sinks)

        for bs in combinations(backup_sinks, 6 - len(all_sinks)):
            # print('backup_sink:', bs)
            rest_backup_sink = [b for b in backup_sinks if b not in bs]
            # print('rest_backup_sink:', rest_backup_sink)
            # print('rest_backup_sink:', sorted(sum(rest_backup_sink, [])))
            rebuild_sink = all_sinks + [b for b in bs]
            # print('rebuild_sink', sorted(rebuild_sink))
            # print(sorted(sum(rest_backup_sink, [])))
            sign_label_dict = {}
            for i in range(len(sign_label)):
                sign_label_dict[sign_label[i]] = bs[i]
            # print(sign_label_dict)
            # print(blocks)
            # print(labels)
            # re_blocks, re_labels = re_organize_bolckandlabel(blocks, labels, rebuild_sink, sorted(sum(rest_backup_sink, [])), sign_label_dict)
            re_labels = re_organize_label(labels, rebuild_sink, rest_backup_sink, sign_label_dict)
            # print(blocks)
            # print(re_labels)
            # print([sorted(rebuild_sink)])
            # print('-------------back to normal:')
            re_result = normal_reblock_and_relabel(blocks, re_labels, unknown_list)
            for nr in re_result:
                yield (nr[0], nr[1])
                # print('result:')
                # print(nr[0], nr[1])
                # do_blocking_result.append((nr[0], nr[1]))
                yield from do_blocking(nr[0], nr[1], target_block_count - 1)
    else:
        re_result = normal_reblock_and_relabel(blocks, labels, unknown_list)
        for nr in re_result:
            yield (nr[0], nr[1])
            # print('normal result:')
            # print(nr[0], nr[1])
            # do_blocking_result.append((nr[0], nr[1]))
            yield from do_blocking(nr[0], nr[1], target_block_count - 1)

    return do_blocking_result

if __name__ == '__main__':
    blocks = ['Dominique Fournier', 'Crémilleux', 'A quality', 'pruning.', 'Knowl.-Based Syst', '2002', '15', '37-43']
    # labels = ['Author', 'Unknown', 'Title', 'Unknown',  'Journal','Year', 'Volume', 'Pages']
    labels = ['Journal', 'Unknown', 'Unknown', 'Title', 'Unknown', 'Volume', 'Unknown', 'Pages']
    # labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Unknown', 'Year', 'Pages']
    # labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Volume', 'Journal', 'Pages']
    # blocks = ['Dominique Fournier Crémilleux A quality', 'pruning.', 'Knowl.-Based Syst', '2002', '15', '37-43']
    # labels = ['Author', 'Title', 'Journal', '0_Backup_Unknown', 'Year', 'Pages']

    # blocks = ['towards effective indexing for large video sequence', 'data', 'h. t. shen b. c. ooi x. zhou and z. huang', 'sigmod', '2005']
    # labels = ['Title', 'Unknown', 'Author', 'Journal', 'Year']

    start = time.clock()
    # do_blocking_result = do_blocking(blocks, labels, 6)
    # # print(do_blocking_result)
    # if do_blocking_result:
    #     for r in do_blocking_result:
    #         print('result:', r)
    for result in do_blocking(blocks, labels, 6):
        print("main result:", result)
    end = time.clock()
    print('time consuming: %f s' % (end - start))
    # for l in do_blocking(blocks, labels):
    #     print(l)
    # labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Year', 'Volume', 'Pages']

    # blocks = ['Dominique Fournier', 'Crémilleux A quality', 'pruning.', 'Knowl.-Based Syst.', '2002', '15', '37-43']
    # labels = ['Author', '0_Backup_Unknown', 'Title', '1_Backup_Unknown', 'Journal', 'Unknown', 'Pages']
    # for r in normal_reblock_and_relabel(blocks, labels):
    #     print(r)
    # do_blocking_result = do_blocking(blocks, labels)
    # print(do_blocking_result)
    # print(len(do_blocking_result))
