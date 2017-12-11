import numpy as np
#
# l = [1,2,3,4,5,6,7,8]
# l2 = [2,3,4,5,6,7,8,9]
# l3 = [20,3,4,5,6,7,8,9]
# n = np.array([l, l2, l3])
# # print(n)
# # print(n[:-1])
# # print(n.transpose())
#
# accu = n.transpose()
# print(accu)
#
# me_n = np.mean(accu[:-1], axis=0)
# print(me_n)


# print()
# me_n = np.mean(n[:-1].transpose(), axis=0)
# print(me_n)
#

time2 = np.array([311.23863200000005, 312.3530420000002, 312.2936539999999, 312.6419459999999, 312.982534, 313.052699, 313.33435199999997, 319.68091499999997])
time2 = time2 / 10 - 5

print(time2)