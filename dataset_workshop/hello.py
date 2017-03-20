
print(44 % 2)
print(45 % 2)
l = [1,2,3,4]
print(l[:-1])
l2 = [1,2]
print(l2[:-1])
print(l2[-1])


def contains(small, big):
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return True
    return False


def my_in(block, l):
    return contains(block.split(), l.split())
l = 'Isospectral-like.flows and eigenvalue problem.'
print('Isos' in l)
print(my_in('Isos', l))
# print(l.strip('.'))
#
# print('Math' in l)
# print(my_in('Math', l))
# print(my_in('Mathematics and', l))
# print(my_in('Isospectral-like flows and eigenvalue problem', l))
# ls = [{}]
# ls2 = [{'Author': 739}, {}]
#
# if ls2[0]:
#     print('T')
# else:
#     print('F')