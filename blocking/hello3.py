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

    re_block_index = [[0], [1, 2, 3], [4], [5], [6], [7]]
    Backup_Unknown = '0_Backup_Unknown'
    blocks = ['Dominique Fournier', 'Crémilleux', 'A quality', 'pruning.', 'Knowl.-Based Syst.', '2002', '15', '37-43']
    labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Year', 'Unknown', 'Pages']

    # s = [1, 2]
    re_block = []
    re_label = []

    for bi in re_block_index:
        # print(bi)
        # print(get_label_value(bi, labels, Backup_Unknown))
        re_block.append(get_block_value(bi, blocks))
        re_label.append(get_label_value(bi, labels, Backup_Unknown))

    print(re_block)
    print(re_label)