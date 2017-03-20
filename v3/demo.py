from decimal import *
from hello import *
getcontext().prec = 6
# print(Decimal(1) / Decimal(7))
#
# f = 0.00123
#
# s = str('%.4f' % f)
# print(s)

a = Decimal(0.009999)
b = Decimal(0.000001)
c = a + b
print(c)
c += Decimal(0.000001)
print(c)

print(float('%.3f' % (1/3)))
