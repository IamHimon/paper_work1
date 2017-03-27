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


def get_label_value2(s, labels, back_sink_sign):
    # print(s)
    if s in back_sink_sign.values():
        return vale_find_key(s, back_sink_sign)
    else:
        return labels[s[0]]


def vale_find_key(value, my_dict):
    for k, v in my_dict.items():
        if value == v:
            return k

if __name__ == '__main__':
    # re_block_index = [[0], [1, 2, 3], [4], [5], [6], [7]]
    # Backup_Unknown = '0_Backup_Unknown'
    # blocks = ['Dominique Fournier', 'Crémilleux', 'A quality', 'pruning.', 'Knowl.-Based Syst.', '2002', '15', '37-43']
    # labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Journal', 'Year', 'Unknown', 'Pages']
    # # s = [1, 2]
    # re_block = []
    # re_label = []
    # for bi in re_block_index:
    #     # print(bi)
    #     # print(get_label_value(bi, labels, Backup_Unknown))
    #     re_block.append(get_block_value(bi, blocks))
    #     re_label.append(get_label_value(bi, labels, Backup_Unknown))
    # print(re_block)
    # print(re_label)

    # backup_sink = ([1, 2], [4])
    # back_sink_sign = {'0_Backup_Unknown': [1, 2], '1_Backup_Unknown': [4]}
    # rest_backup_sink = [[6]]
    # l = ([[0], [1, 2], [3], [4], [5], [7]], [6])
    # re_block_index = [[0], [1, 2], [3], [4], [5], [6], [7]]
    # sing_label = ['0_Backup_Unknown', '1_Backup_Unknown']
    # blocks = ['Dominique Fournier', 'Crémilleux', 'A quality', 'pruning.', 'Knowl.-Based Syst.', '2002', '15', '37-43']
    # labels = ['Author', 'Unknown', 'Unknown', 'Title', 'Unknown', 'Year', 'Unknown', 'Pages']
    #
    # re_block = []
    # re_label = []
    #
    # for bi in re_block_index:
    #     # print(get_label_value2(bi, labels, back_sink_sign))
    #     re_block.append(get_block_value(bi, blocks))
    #     re_label.append(get_label_value2(bi, labels, back_sink_sign))
    #
    # print(re_block)
    # print(re_label)

    combined_sinks = [[[0], [1], [2, 3], [4], [5], [6]], [[0], [1], [2], [3, 4], [5], [6]], [[0], [1], [2, 3], [4], [5], [6]], [[0], [1], [2], [3, 4], [5], [6]]]

    # print(len(combined_sinks))
    # print(set(combined_sinks))
    for c in combined_sinks:
        print(c)
    news_ids = []
    for id in combined_sinks:
        if id not in news_ids:
            news_ids.append(id)
    print(news_ids)