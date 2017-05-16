import string
import re
# s1 = "an approximate max-flow min-cut relation for unidirected multicommodity flow with applications"
# s2 = "an approximate max-flow min-cut relation for unidirected multicommodity flow  with applications"
#
#
# print(s1 == s2)
# r1 = '\s+'
# s1 = re.sub(r1, '', s1)
# s2 = re.sub(r1, '', s2)
# print(s1)
# print(s2)
# print(s1 == s2)

s1 = {'24', '546', '532'}
s2 = {'24', '780', '1031', '253'}
s3 = {'1', '23', '34', '555', '1031'}

# s = set()
# s.add(s1)
# s.add(s2)
# print(s)
print(s1 | s2 | s3)
