import random
import math
def add2dict(my_dict, key, value):
    if key in my_dict.keys():
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
    return my_dict


# l1 = [1,2,3]
# l2 = [3,4,5]
# l3 = [5,6,7]
l4 = [7,8,9,1,2,3,4,5,6]
# print(l4[:3])
#
# d = {}
# d = add2dict(d, 'one', l1)
# d = add2dict(d, 'tow', l2)
# d = add2dict(d, 'tow', l4)
# d = add2dict(d, 'three', l3)
# d = add2dict(d, 'three', l4)
#
# print(d.values())
# #
# for key in d.keys():
#     for value in d[key]:
#         print(value)
#         # print(key + ':'+' '.join(value))

# print(d)
# d = {'one':(l1,l3), 'tow':l2, 'three':l3}

# d['one']=l3
# d.setdefault('one', [])
# d['one'].append(l4)
#
# print(d)
# print(d['one'])

# l = ['Author', 'Title', 'Title', 'Title', 'Title', None]
# l2 = ['Title']
# label = ''
# i = 0
# print(int(7/2))

# p5 = 'Calibrating Trajectory Data for Similarity-based Analysis, H. Su, K. Zheng, H. Wang, J. Huang , X. Zhou, SIGMOD, 2013'
# b = 'H. Su, K. Zheng'
# print(p5.index(b))
# print(p5.index('J. Huang'))

def selectMinKey(a, i, length):
    k = i
    for j in range(i+1, length):
        if a[k] > a[j]:
            k = j
    return k


def selectSort(record, block, label):
    length = len(block)
    indexes = []
    for i in range(len(block)):
        index = record.index(block[i])
        indexes.append(index)

    for i in range(length):
        key = selectMinKey(indexes, i, length)
        if key != 1:
            tmp = indexes[i]
            indexes[i] = indexes[key]
            indexes[key] = tmp

            tmp = block[i]
            block[i] = block[key]
            block[key] = tmp

            tmp = label[i]
            label[i] = label[key]
            label[key] = tmp




# p3 = 'Nuno Sepúlveda,Carlos Daniel Paulino,Carlos Penha Gonçalves,Bayesian analysis of allelic penetrance models for complex binary traits.,Computational Statistics & Data Analysis,2009,53,1271-1283'
#
# label = ['Title', 'Title', 'Title', 'Title', 'Title', 'Journal', 'Unknow', 'Unknow', 'Unknow', 'Unknow', 'Unknow', 'Unknow', 'Volume', 'Author', 'Author', 'Author', 'Author', 'Pages', 'Year']
# block = ['Bayesian', 'analysis of', 'models for', 'complex', 'binary traits', 'Computational Statistics & Data Analysis', 'Sepúlveda', 'Paulino', 'Penha', 'Gonçalves', 'allelic', 'penetrance', '53', 'Nuno', 'Carlos', 'Daniel', 'Carlos', '1271-1283', '2009']
# selectSort(p3, block, label)
# print(block)
# print(label)
#
# random.shuffle(block)
# print(block)
# d1 = {'a': 1, 'b': 2}
# d2 = {'c': 3, 'd': 4}
# d3 = {'e': 5, 'f': 6}
# l = []
# l.append(d1)
# l.append(d2)
# l.append(d3)
# print(l)

print(math.exp(34))