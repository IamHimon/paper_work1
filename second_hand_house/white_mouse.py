from SHH_testdata.generate_dataset import *
from blocking.block import *
from blocking.reconstruction import *
from second_hand_house.toolbox import *

if __name__ == '__main__':
    KB = loadKB_SHH()

    r = '东环 沿线 东环 * 村好 房出租 干净 清爽 看房 方便,2015 年 02 月 26 日,1800 元 / 月,付 3 押 1,2 室 1 厅 1 卫,75 平米,4 / 6,床 空调 电视 冰箱 洗衣机 热水器 宽带 阳台 地铁 信息 ： 紧邻 1 号线 东环路 站'
    r1 = '凯悦 大厦 家电 齐全 看房 随时 一室 精装 紧靠 吴中 汽车站 靠近 地铁口,2015 年 03 月 09 日,1650 元 / 月,付 3 押 1,1 室 1 厅 1 卫,48 平米,17 / 26,床 空调 电视 冰箱 洗衣机 热水器 可做饭 独立 卫生间 阳台'
    r2 = '园区 CBD 东环 沿线 恒润后 街 酒店式 公寓 精装 一房 拎包 入住 真实'
    r3 = '御窑 花园 顶 带阁 出租 家电 一房 拎包 入住 真实 齐全 安静 舒适 随时 看房 拎包 入住'
    r4 = '精装 两房 宝带 商圈 好 房子 不 二家 限时 秒杀 独家 房源,2015 年 02 月 26 日,3500 元 / 月,付 3 押 1,2 室 2 厅 1 卫,102 平米,25 / 25,床 空调 电视 冰箱 洗衣机 热水器 宽带 可做饭 独立 卫生间 阳台'
    blocks, anchors = doBlock5(r4, KB, SECOND_HAND_HOUSE, threshold=0.95)
    print(blocks)
    print(anchors)
    re_blocks, re_anchors = re_block(blocks, anchors)
    print(re_blocks)
    print(re_anchors)
    if len_Unknown2(re_anchors, SECOND_HAND_HOUSE):
            for result in do_blocking2(re_blocks, re_anchors, len(SECOND_HAND_HOUSE), SECOND_HAND_HOUSE):
                print('result:', result)
    else:
        print((re_blocks, re_anchors))

