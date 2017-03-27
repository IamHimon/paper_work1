from second_hand_house.toolbox import *
s3 = 'Discovering the Most Influential Sites over Uncertain Data: A Rank Based Approach, K. Zheng, Z. Huang, A. Zhou and X. Zhou, IEEE Transactions on Knowledge and Data Engineering, 24(12),2156-2169, 2012'
s4 = '20-48'
s5 = '99(35)'
print(sample_pretreatment_disperse_number2(s3))
print(sample_pretreatment_disperse_number2(s4).strip())
print(sample_pretreatment_disperse_number2(s5).strip())
