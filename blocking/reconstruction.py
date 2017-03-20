import operator
import math
from blocking.block import *
from itertools import combinations, permutations



# sinks:[[[],[],[]...], [[],[],[],..], [[],[],[],,]]
def combine_all_sinks(u, all_sinks):
    # print('u:', u)
    # print('all_sinks:', all_sinks)
    combined_sink = []
    for sinks in all_sinks:
        # print('sinks:', sinks)
        available_sink_index = []
        for s in range(len(sinks)):
            if judge_if_neighbour(u, sinks[s]):
                available_sink_index.append(s)
        print('available_sink_index:', available_sink_index)
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
    if (u not in sink) and (abs(sink[0]-u) == 1) or (abs(sink[-1]-u) == 1):
        return True
    else:
        return False


def remove_Unknown(label):
    return [l for l in label if l != 'Unknown']


def reblock_according_sinks(blocks, labels, sinks):
    re_blocks = []
    re_labels = []
    for sink in sinks:
        # print(sink)
        b_temp = [blocks[i] for i in sink]
        re_blocks.append(' '.join(b_temp))

        l_temp = [labels[i] for i in sink]
        re_labels.append(remove_Unknown(l_temp)[0])
        # print(b_temp)
        # print(l_temp)
    return re_blocks, re_labels


def normal_reblock_and_relabel(blocks, labels):
    label_dict = {}
    unknown_indexes = []
    for i in range(len(labels)):
        if labels[i] == 'Unknown':
            unknown_indexes.append(i)
        else:
            label_dict[labels[i]] = i
    # print(label_dict)

    sorted_x = sorted(label_dict.items(), key=operator.itemgetter(1))
    print(sorted_x)
    anchor_indexes = [anchor[1] for anchor in sorted_x]
    print(anchor_indexes)
    print(unknown_indexes)

    all_sinks = [[[a] for a in anchor_indexes]]
    print(all_sinks)
    combined_sinks = all_sinks

    for u in unknown_indexes:
        combined_sinks = combine_all_sinks(u, combined_sinks)
        print(combined_sinks)
    # yield all block and label according combined_sinks_index
    for sinks in combined_sinks:
        # print(sinks)
        re_blocks, re_labels = reblock_according_sinks(blocks, labels, sinks)
        # print(re_blocks)
        # print(re_labels)
        yield (re_blocks, re_labels)


def unnormal_reblock_and_relabel(blocks, labels, all_sinks, unknown_indexes):
    combined_sinks = all_sinks
    for u in unknown_indexes:
        combined_sinks = combine_all_sinks(u, combined_sinks)
        print(combined_sinks)
    # yield all block and label according combined_sinks_index
    for sinks in combined_sinks:
        # print(sinks)
        re_blocks, re_labels = reblock_according_sinks(blocks, labels, sinks)
        # print(re_blocks)
        # print(re_labels)
        yield (re_blocks, re_labels)


def get_block_value(s, blocks):
    result = [blocks[i] for i in range(len(blocks)) if i in s]
    return ' '.join(result)


def get_label_value(s, labels, Backup_Unknown):
    for i in range(len(labels)):
        if i in s:
             if len(s) > 1:
                return Backup_Unknown
             else:
                return labels[i]

if __name__ == '__main__':
    blocks = ['Dominique Fournier', 'Crémilleux', 'A quality', 'pruning.', 'Knowl.-Based Syst.', '2002', '15', '37-43']
    # labels = ['Author', 'Unknown', 'Title', 'Unknown',  'Journal', 'Unknown','Year', 'Volume', 'Pages', 'Unknown']
    labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Year', 'Unknown', 'Pages']
    labels2 = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Year', 'Unknown', 'Pages']
    # labels2 = ['Author', 'Unknown', 'Title', 'Unknown', 'Journal', 'Unknown', 'Unknown', 'Pages']

    # print(len_ex_Unknown(labels))


    # for b in normal_reblock_and_relabel(blocks, labels2):
    #     print(b)

    # sinks = [[0, 1], [2, 3], [4], [5], [6], [7]]
    # re_blocks, re_labels = reblock_according_sinks(blocks, labels, sinks)
    # print(re_blocks)
    # print(re_labels)
    #
    label_dict = {}
    # for i in range(len(labels)):
    #     label_dict[labels[i]] = i

    unknown_indexes = []
    for i in range(len(labels)):
        if labels[i] == 'Unknown':
            unknown_indexes.append(i)
        else:
            label_dict[labels[i]] = i


    # print(label_dict)

    sorted_x = sorted(label_dict.items(), key=operator.itemgetter(1))
    print('sorted_x:', sorted_x)
    anchor_indexes = [anchor[1] for anchor in sorted_x]
    print('anchor_indexes:', anchor_indexes)
    print('unknown_indexes:', unknown_indexes)   # 值肯定是递增排序的

    all_sinks = [[a] for a in anchor_indexes]
    print('all_sinks:', all_sinks)


    # bachkup_Unknown
    backup_sinks = []
    if len_ex_Unknown(labels) < 6:
        i = 0
        j = 0
        while i < len(unknown_indexes):
            temp = [unknown_indexes[i]]
            j = i+1
            while j < len(unknown_indexes):
                if judge_if_neighbour(unknown_indexes[j], [unknown_indexes[i]]):
                    temp += [unknown_indexes[j]]
                    # print(temp)
                    j += 1
                else:
                    break
            i = j
            # print(temp)
            backup_sinks.append(temp)

        print('backup_sinks', backup_sinks)

        rebuild_block = []
        for bs in permutations(backup_sinks, 6 - len(all_sinks)):
            rest = [b for b in backup_sinks if b not in bs]
            for s in bs:
                rebuild_block = sorted(all_sinks + [s])
            print((rebuild_block, sum(rest, [])))


'''
        for b_s in backup_sinks:
            print(b_s)

        # for b in backup_sinks:
        #     l2.append(b)
        #     # print(l2)
        #     print(sorted(l2))
        #     l2 = l

        # 根据backup_sinks ,重构block和anchor,
        print('blocks:', blocks)
        print('labels', labels)

        for b_sink in backup_sinks:
            # print(b)
            b_temp = []
            l_temp = []
            for i in b_sink:
                b_temp.append(blocks[i])
                l_temp.append(labels[i])
                # print(blocks[i])
            print(' '.join(b_temp))
            print(l_temp)

    combined_sinks = all_sinks
    print(combined_sinks)

    for u in unknown_indexes:
        # print(u)
        combined_sinks = combine_all_sinks(u, combined_sinks)
    print(combined_sinks)

    # yield all block and label according combined_sinks_index
    for sinks in combined_sinks:
        # print(sinks)
        re_blocks, re_labels = reblock_according_sinks(blocks, labels, sinks)
        print(re_blocks)
        print(re_labels)
'''
