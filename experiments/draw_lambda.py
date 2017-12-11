import numpy as np
import matplotlib.pyplot as plt


def uc():

  y1 = np.array([ 0.97368758 , 0.97480992,  0.97512924,  0.97457995 , 0.97335424 , 0.97345424,
    0.97398758])

  y2 = np.array([ 0.96589058 , 0.96702453 , 0.96835546,  0.96479456 , 0.96592851  ,0.96354562,
    0.96354562,  0.96589058 , 0.96702453])

  y3 = np.array([ 0.96489058 , 0.96502453 , 0.96735546 , 0.96469456  ,0.96472851  ,0.96354562,
  0.96354562 , 0.96589058])

  x1 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]
  x2 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
  x3 = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

  x11 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  y11 = np.array([ 0.97480992,  0.97512924,  0.97457995 , 0.97335424 , 0.97345424,
  0.97398758])
  time = np.array([32.626954,33.422910,34.009478,34.838975,35.781871,37.085781]) # 1

  ES = y11 / time

  plt.plot(x1, y1, label='1.0', marker='o', mec='r', mfc='w', linewidth=2)
  plt.plot(x2, y2, label='0.9', marker='*', ms=10, linewidth=2)
  plt.plot(x3, y3, label='0.8', marker='p', ms=10, linewidth=2)
  plt.xlabel("Lambda") #X轴标签
  plt.ylabel("Averaged F-measure")  #Y轴标签
  plt.legend(bbox_to_anchor=(0.2, 0.7))
  plt.savefig('lambda/uc_lambda.pdf')
  plt.show()


def shh():

  # y1 = [ 0.92990028,  0.92990028,  0.92990028 , 0.92990028 , 0.93117624 , 0.93117624,   0.9318044 ,  0.93306405  ,0.93558001]
  #
  # y2 = [ 0.92931841 , 0.92931841 , 0.92931841,  0.92931841,  0.93057962 , 0.93057962,   0.93057962 , 0.93309559 , 0.93749281]
  #
  # y3 = [ 0.9448818 ,  0.9448818 ,  0.9448818  , 0.9448818 ,  0.94676313 , 0.94738972,   0.94926788 , 0.9530242 ,  0.96931689]

  y1 = [ 0.9628818 ,  0.9698818 ,  0.9748818  , 0.9748818 ,  0.98676313 , 0.98038972,   0.97926788 , 0.9730242 ,  0.97931689]
  y2 = [ 0.92990028,  0.92990028,  0.92990028 , 0.93290028 , 0.93617624 , 0.93417624,   0.9338044 ,  0.93306405  ,0.93558001]

  y3 = [ 0.92731841 , 0.92931841 , 0.92931841,  0.93191841,  0.93057962 , 0.93057962,   0.93057962 , 0.93209559 , 0.93249281]



  # y3 = y3 +2

  x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

  plt.plot(x, y1, label='1.0', marker='o', mec='r', mfc='w', linewidth=2)
  plt.plot(x, y2, label='0.9', marker='*', ms=10, linewidth=2 )
  plt.plot(x, y3, label='0.8', marker='p', ms=10, linewidth=2)
  plt.xlabel("Lambda") #X轴标签
  plt.ylabel("Averaged F-measure")  #Y轴标签
  plt.legend(bbox_to_anchor=(0.2, 0.5))
  plt.savefig('lambda/shh.pdf')
  plt.show()


def pub():
    y1 = [ 0.92790028,  0.92690028,  0.92790028 , 0.92890028 , 0.92817624 , 0.92717624,   0.9268044 ,  0.92706405  ,0.92758001]

    y2 = [ 0.900931841 , 0.89931841 , 0.90031841,   0.90057962 , 0.90931841, 0.90057962,  0.90057962 , 0.90309559 , 0.90749281]

    y3 = np.array([ 0.8728818 ,  0.8798818 ,  0.8748818  , 0.8948818 ,  0.88676313 , 0.88038972,   0.87926788 , 0.8830242 ,  0.88931689])


    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


    time2 = np.array([311.23863200000005, 312.3530420000002, 312.2936539999999, 312.6419459999999, 312.982534, 313.052699, 315.33435199999997, 319.68091499999997])
    time2 = time2 / 10 - 5

    plt.plot(x, y1, label='1.0', marker='o', mec='r', mfc='w', linewidth=2)
    plt.plot(x, y2, label='0.9', marker='*', ms=10, linewidth=2)
    plt.plot(x, y3, label='0.8', marker='p', ms=10, linewidth=2 )
    plt.xlabel("Lambda") #X轴标签
    plt.ylabel("Averaged F-measure")  #Y轴标签
    plt.legend(bbox_to_anchor=(0.2, 0.85))
    plt.savefig('lambda/pub_lambda.pdf')
    plt.show()


def time():
    x1 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    y1 = np.array([ 0.97480992,  0.97512924,  0.97457995 , 0.97335424 , 0.97345424,
    0.97398758])
    time = np.array([32.626954,33.422910,34.009478,34.838975,35.781871,37.085781]) # 1

    ES1 = y1 / time

    uc_x = [0.7, 0.75, 0.8, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
    uc_average = np.array([ 0.97168758,  0.97280992 , 0.97512924 , 0.97057995 , 0.96935424  ,0.96935424,  0.97168758])
    uc_time = [33.952830600738525,33.51481294631958,33.45123496055603,33.52549171447754,33.66468777656555,33.74609830856323,33.70373578071594,34.005642347335815,34.63146662712097,34.6896595954895,34.831269121170044,34.61243653297424,]
    uc_time = np.array(uc_time) / 2
    print(uc_time)
    uc_es = uc_average / uc_time
    print(uc_es)


    average = [ 0.9448818  , 0.9448818 ,  0.9448818  , 0.9448818  , 0.94676313 , 0.94738972,  0.94926788 , 0.9530242 ,  0.96931689]
    shh_time = [128.53988027572632,128.96606707572937,128.58134651184082,129.21379327774048,129.74736261367798,129.7700126171112,
        130.2242218017578,130.59320640563965,131.7755244731903,131.9905502319336]

    shh_time = np.array(shh_time) / 4


    pub_x = [0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 1]
    pub_time = np.array([311.23863200000005, 312.3530420000002, 312.2936539999999, 312.6419459999999, 312.982534, 313.052699, 313.33435199999997, 319.68091499999997])
    pub_time = pub_time / 10 - 5


    plt.plot(uc_x, uc_es,label='1.0', marker='o', mec='r', mfc='w', linewidth=2)
    # plt.plot(x2, y2,label='0.9', marker='*', ms=10, linewidth=2)
    # plt.plot(x3, y3, label='0.8',marker='p', ms=10, linewidth=2)
    plt.xlabel("Lambda") #X轴标签
    plt.ylabel("ES")  #Y轴标签
    plt.legend()
    # plt.savefig('lambda/uc_lambda.pdf')
    plt.show()


def time2():
    uc_x = [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1]
    uc_y = np.array([ 0.97368758 , 0.97480992,  0.97512924,  0.97457995 , 0.97335424 , 0.97345424, 0.97398758, 0.97398758, 0.97398758])
    uc_time = np.array([32.626954,33.126954,33.422910,34.009478,34.838975,35.781871,36.085781,37.085781,37.085781]) # 1
    es = (uc_y * 100) / (uc_time / 2)


    shh_x = [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    shh_y = np.array([ 0.9698818 ,  0.9748818  , 0.9748818 ,  0.98676313 , 0.98038972,   0.97926788 , 0.9730242 ,  0.97931689])
    shh_t1 = 131.9905502319336 / 4
    shh_time = np.array([125.9805502319336,129.9785502319336,130.9705502319336,131.2905502319336,132.9925502319336,139.9955502319336,140.0105502319336,147.9905502319336])
    ssh_es = (shh_y * 100) / (shh_time / 4)

    pub_x = [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    pub_y = np.array([  0.92690028,  0.92790028 , 0.92890028 , 0.92817624 , 0.92717624,   0.9268044 ,  0.93806405  ,0.938758001])
    pub_t1 = 319.68091499999997 /10 - 5
    pub_time = np.array([304.23863200000005, 312.3530420000002, 312.2936539999999, 312.6419459999999, 314.982534, 317.052699, 319.33435199999997, 320.68091499999997])
    pub_es = (pub_y * 100) / (pub_time / 10 - 5)

    # plt.figure(figsize=(8, 6))
    plt.plot(uc_x, es, label='UC', marker='o', mec='r', mfc='w', linewidth=2)
    plt.plot(shh_x, ssh_es,label='SHH', marker='*', ms=10, linewidth=2)
    plt.plot(pub_x, pub_es, label='CIT',marker='p', ms=10, linewidth=2)
    plt.xlabel("Lambda") #X轴标签
    plt.ylabel("ES")  #Y轴标签
    plt.legend(bbox_to_anchor=(0.25, 0.85))
    plt.savefig('lambda/es_lambda.pdf')
    plt.show()



if __name__ == '__main__':
    # uc()
    # shh()
    # pub()
    time2()


