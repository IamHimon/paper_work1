import string
import re
s1 = "an approximate max-flow min-cut relation for unidirected multicommodity flow with applications"
s2 = "an approximate max-flow min-cut relation for unidirected multicommodity flow  with applications"


print(s1 == s2)
r1 = '\s+'
s1 = re.sub(r1, '', s1)
s2 = re.sub(r1, '', s2)
print(s1)
print(s2)
print(s1 == s2)