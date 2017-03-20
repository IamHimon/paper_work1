from itertools import combinations, permutations
from blocking.reconstruction import *

def infinity(start):
    if start <= 20:
        yield start
        yield from infinity(start + 1)
    else:
        return
# inf = infinity(10)
# print(next(i))
# print(next(i))
# print(next(i))
# for i in infinity(10):
#     print(i)


# unknown_index = [1, 3]
# anchors = [[0], [2], [4], [5], [6], [7]]

def insert_sink(l1, l2):
    result = sorted(l2 + [l1])
    return result


blocks = ['Dominique Fournier', 'Crémilleux', 'A quality', 'pruning.', 'Knowl.-Based Syst.', '2002', '15', '37-43']
labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Year', 'Unknown', 'Pages']

t1 = ([[0], [1, 2], [3], [5], [7]], [6])
s = '0_Backup_Unknown'
# print(t1[0])
# for sink in t1[0]:
#     print(sink)
#     print(labels[sink[0]])
# print(t1[-1])

t2 = ([[0], [3], [5], [6], [7]], [1, 2])



l = [[0], [3], [5], [7]]
l2 = [[0], [3], [4], [5], [7]]
backup_sinks = [[1, 2], [4], [6]]
backup_sinks2 = [[1, 2], [6]]

rebuild_block = []
sign = 0
sing_label = [str(i) + '_Backup_Unknown' for i in range(6 - len(l2))]
print(sing_label)
for bs in permutations(backup_sinks2, 6 - len(l2)):
    rest = [b for b in backup_sinks2 if b not in bs]
    # print('backup_sink:', bs)
    for i in range(len(bs)):

        print(sing_label[sign])
        print('add_backup_sink:', bs[i])
        rest_bs = [x for x in bs if x != bs[i]]
        rebuild_block = sorted(l2 + [bs[i]])
        print((rebuild_block, sorted(sum(rest + rest_bs, []))))
        # 重构blocks和labels
        re_block_index = sorted(rebuild_block + [sorted(sum(rest + rest_bs, []))])
        # print(re_block_index)

        re_blocks = []
        re_labels = []

        for bi in re_block_index:
            re_blocks.append(get_block_value(bi, blocks))
            re_labels.append(get_label_value(bi, labels, sing_label[sign]))
        print(re_blocks)
        print(re_labels)

    print('===========')


