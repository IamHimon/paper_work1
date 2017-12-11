import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

def draw_f(x, y, path, xlabel, y_label):
    # fig = plt.figure(figsize=(3, 2),  facecolor="white")
    # axes = plt.subplot(111)
    # axes.cla()#清空坐标轴内的所有内容
    fig, ax = plt.subplots()




    # 配置一下坐标刻度等
    ax=plt.gca()
    # ax.set_xticks(np.linspace(0,1,9))
    # ax.set_xticklabels( ('275', '280', '285', '290', '295',  '300',  '305',  '310', '315'))
    # ax.set_yticks(np.linspace(0,1,8))
    # ax.set_yticklabels( ('0.80', '0.85', '0.90', '0.95', '1.0'))


    plt.tick_params(axis='both', which='major', labelsize=15)
    # plt.rc('font', size=SMALL_SIZE)
    plt.ylim(0.9, 1.0)
    plt.plot(x, y, "b--", linewidth=2)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度
    plt.xlabel(xlabel, fontsize=20) #X轴标签
    plt.ylabel(y_label, fontsize=20)  #Y轴标签
    plt.savefig(path) #保存图
    plt.show()  #显示图
    plt.close()


def draw_f2(x, y, path, xlabel, y_label):
    # fig = plt.figure(figsize=(10, 8),  facecolor="white")
    fig = plt.figure(figsize=(4,4), dpi=100)
    axes = plt.subplot(111)
    axes.cla()#清空坐标轴内的所有内容
    ax = plt.gca()
    # fig.add_axes([ 0.4, 0.8,1])
    # locator= ['0.4','0.8','0.1']

    # 配置一下坐标刻度等
    axis = plt.gca().xaxis
    # yaxis = plt.gca().yaxis
    #
    # print(axis.get_ticklocs())
    # print(axis.get_ticklabels())
    # axis.set_major_locator( MultipleLocator(1/4) )

    # print(ax1.xaxis.get_view_interval())
    # print(ax1.xaxis.get_major_locator())
    # ax1.xaxis.set_major_locator(eval(locator))
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(25)
    #
    # for tick in axis.get_major_ticks():
    #     tick.label.set_fontsize(25)
    #
    # for tick in yaxis.get_major_ticks():
    #     tick.label.set_fontsize(25)

    # ax.set_xticks(np.linspace(0,1,9))
    # ax.set_xticklabels( ('275', '280', '285', '290', '295',  '300',  '305',  '310', '315'))
    # ax.set_yticks(np.linspace(0,1,8))
    # ax.set_yticklabels( ('0.80', '0.85', '0.90', '0.95', '1.0'))

    # plt.tick_params(axis='both', which='major', labelsize=15)
    # plt.rc('font', size=SMALL_SIZE)
    plt.ylim(0.9, 1.0)
    # plt.xlim(0.4, 1.0)
    plt.plot(x, y, "b--", linewidth=2)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度
    plt.xlabel(r"\lambda", fontsize=25) #X轴标签
    plt.ylabel(y_label, fontsize=25)  #Y轴标签
    plt.savefig(path) #保存图
    plt.show()  #显示图
    plt.close()


def draw_f3(x, y,path, xlabel, y_label):
    # ymajorLocator   = MultipleLocator(0.02) #将y轴主刻度标签设置为0.5的倍数
    # ax.yaxis.set_major_locator(ymajorLocator)
    # axis = plt.gca().xaxis
    # yaxis = plt.gca().yaxis
    fig, ax = plt.subplots(1,1,figsize=(8, 3))

    ax.set_ylim([0.88, 0.98])
    ax.yaxis.labelpad = 0.02

    # for tick in axis.get_major_ticks():
    #     tick.label.set_fontsize(6)
    #
    # for tick in yaxis.get_major_ticks():
    #     tick.label.set_fontsize(6)

    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    # ax.set_ylabel(r'$\theta$', fontsize=18)
    fig.subplots_adjust(left=0.45, right=0.7, bottom=0.3, top=0.65)
    plt.savefig(path) #保存图
    plt.show()
    plt.close()


def draw_e(x, y, path, xlabel, y_label):
    fig, ax = plt.subplots(1,1,figsize=(8, 3))
    ymajorLocator   = MultipleLocator(100) #将y轴主刻度标签设置为0.5的倍数

    # ax.set_ylim([0.15, 0.45])
    # ax.yaxis.labelpad = 0.05
    ax.set_ylim([0, 500])
    # ax.yaxis.labelpad = 100
    ax.yaxis.set_major_locator(ymajorLocator)
    group_labels = ['0', '100', '200','300','400','500']

    # for tick in axis.get_major_ticks():
    #     tick.label.set_fontsize(6)
    #
    # for tick in yaxis.get_major_ticks():
    #     tick.label.set_fontsize(6)

    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(r'$\theta$', fontsize=12)
    ax.set_ylabel('Time(ms)', fontsize=12)
    # ax.set_ylabel(r'$\theta$', fontsize=18)
    fig.subplots_adjust(left=0.45, right=0.7, bottom=0.3, top=0.65)
    # plt.yticks(y, group_labels, rotation=0)
    plt.savefig(path) #保存图
    plt.show()
    plt.close()


def draw_kb(x, cnn_y, ondux_y, path):
    fig, ax = plt.subplots(1,1,figsize=(8, 3))
    ax.set_ylim([0.70, 1.0])
    ax.yaxis.labelpad = 0.04

    ax.plot(x, cnn_y, label='CNN-IETS', marker='x', ms=4,  linewidth=1)
    ax.plot(x, ondux_y, label='ONDEX',  marker='v', ms=4, linewidth=1)
    plt.legend(bbox_to_anchor=(0.9, 0.35), fontsize=4)
    ax.set_xlabel('KB', fontsize=12)
    ax.set_ylabel('F-measure', fontsize=12)
    # ax.set_ylabel(r'$\theta$', fontsize=18)
    fig.subplots_adjust(left=0.45, right=0.7, bottom=0.3, top=0.65)
    plt.savefig(path) #保存图
    plt.show()


def draw_labmda(x, y, path):
    ymajorLocator   = MultipleLocator(0.2) #将y轴主刻度标签设置为0.5的倍数
    fig, ax = plt.subplots(1,1,figsize=(8, 3))
    ax.set_ylim([0.7, 1.0])
    ax.yaxis.labelpad = 0.05

    ax.set_xlim([0.2, 1.0])
    ax.xaxis.labelpad = 0.2
    ax.xaxis.set_major_locator(ymajorLocator)

    # for tick in axis.get_major_ticks():
    #     tick.label.set_fontsize(6)
    #
    # for tick in yaxis.get_major_ticks():
    #     tick.label.set_fontsize(6)

    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(r'$\lambda$', fontsize=12)
    ax.set_ylabel('F-measure', fontsize=12)
    # ax.set_ylabel(r'$\theta$', fontsize=18)
    fig.subplots_adjust(left=0.45, right=0.7, bottom=0.3, top=0.65)
    plt.savefig(path) #保存图
    plt.show()

def pub_f():
    x = [0.4,0.5,0.6,0.7,0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 1]
    average = [ 0.88016946 ,0.88416946 ,0.88816946 ,0.90116946 ,0.91716946 , 0.9293022  , 0.9277704 ,  0.92456182 , 0.920956177 , 0.92092808,  0.919743771 , 0.910895121]
    draw_f3(x, average, 'bib_f.pdf', 'lambda', 'F-measure')


def pub_e():
    # x = [0.4,0.5,0.6,0.7,0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 1]
    # time = np.array([311.23863200000005, 312.3530420000002, 312.2936539999999, 312.6419459999999, 312.982534, 313.052699, 313.33435199999997, 319.68091499999997])
    # time = time / 10 - 5
    x = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    # time = np.array(sorted([32.55,31.89,30.89,28.44,25.33,24.33,21.45]))/100

    time = np.array([0.047, 0.081, 0.091, 0.100,0.100, 0.151, 0.202, 0.256]) * 1000
    draw_e(x, time,'bib_e.pdf', 'lambda', 'Time')


def shh_f():
    x = [0.4,0.5,0.6,0.7,0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
    average = [0.90009324,0.91009324,0.91009324,0.92009324,0.95909324, 0.97309324, 0.9790037, 0.9740037,  0.96597037, 0.96097037, 0.96136585, 0.95448273]
    draw_f3(x, average,'house_f.pdf', 'lambda', 'F-measure')

def shh_e():
    x = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
    # time = [44.9,43.89,42.89,39.39,35.33,36.33,24.45]

    time = np.array([0.107,0.124, 0.230,0.231,0.321,0.340,0.399, 0.432])*1000

    draw_e(x, time,'house_e.pdf', 'lambda', 'Time')


def uc_f():
    x = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
    average = np.array([0.90353263, 0.91353263, 0.92353263, 0.95353263 , 0.96089058,  0.96189058 , 0.96489058,  0.96489058 , 0.96589058,  0.96889058 , 0.96589058  ,0.96468758  ,0.96078758 , 0.95468758  ,0.95068758])

    # draw_f2(x, average, 'uc_f.pdf', 'lambda', 'F-measure')
    draw_f3(x, average,'car_f.pdf', 'lambda', 'F-measure')

def uc_e():
    # time = [33.952830600738525, 33.51481294631958, 33.45123496055603, 33.52549171447754,33.66468777656555,33.74609830856323,33.70373578071594,34.005642347335815,34.63146662712097,34.6896595954895,34.831269121170044,34.61243653297424,]
    x = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,  1]
    # time = [26.9,24.89,23.89,20.39,17.93,17.33,15.45]
    # time = [0.107,0.124, 0.230,0.231,0.321,0.340,0.399, 0.462]
    time = np.array([0.093, 0.125, 0.135, 0.1458, 0.195,0.214, 0.281, 0.346]) * 1000

    draw_e(x, time, 'car_e.pdf', 'lambda', 'Time')

def bib_kb():
    kb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    c_f = [0.874, 0.901, 0.929, 0.942,0.959, 0.979]
    o_f = [0.704, 0.741, 0.810, 0.839,0.870, 0.881]
    draw_kb(kb, c_f, o_f, 'bib_kb.pdf')


def house_kb():
    kb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    c_f = [0.924, 0.951, 0.979, 0.983, 0.988, 0.990]
    o_f = [0.723, 0.795, 0.845, 0.865, 0.885, 0.901]
    draw_kb(kb, c_f, o_f, 'house_kb.pdf')


def car_kb():
    kb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    c_f = [0.894, 0.931, 0.968, 0.978, 0.985, 0.992]
    o_f = [0.723, 0.765, 0.861, 0.895, 0.905, 0.901]
    draw_kb(kb, c_f, o_f, 'car_kb.pdf')


def bib_lambda():
    y1 = [ 0.81790028,  0.84690028,  0.87790028 , 0.90890028 , 0.92817624 , 0.92817624,   0.8568044 ,  0.81706405  ,0.80758001]
    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # y1 = [ 0.81790028,   0.87790028 ,  0.91817624 ,   0.8968044  ,0.87758001]
    # x = [0.2, 0.4, 0.6, 0.8,  1]
    draw_labmda(x, y1, 'bib_lambda.pdf')

def house_lambda():
    y1 = [ 0.8528818 ,  0.8698818 ,  0.9348818  , 0.9748818 ,  0.98676313 , 0.94038972,   0.90926788 , 0.9030242 ,  0.89931689]
    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    draw_labmda(x, y1, 'house_lambda.pdf')


def car_lambda():
    y1 = np.array([0.83368758 , 0.86368758 , 0.91480992,  0.93512924,  0.97457995 , 0.96835424 , 0.94345424,  0.93398758,0.90398758])
    x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
    draw_labmda(x, y1, 'car_lambda.pdf')
    # data4gunplot(x, y1, '../dat4gnuplot/car_lambda.dat')


def data4gunplot(x, y, filepath):
    fw = open(filepath, 'w+')
    for i in range(len(x)):
        fw.write(str(x[i]) + ' ' + str(y[i]))
        fw.write('\n')
    fw.close()


if __name__ == '__main__':
    print('λ')
    pub_f()
    # shh_f()
    # uc_f()
    # pub_e()
    # shh_e()
    # uc_e()
    bib_kb()
    house_kb()
    car_kb()
    # bib_lambda()
    # house_lambda()
    # car_lambda()


