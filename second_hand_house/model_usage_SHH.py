import tensorflow as tf
from second_hand_house.toolbox import *
from second_hand_house.hello import *


l = "急租 盘蠡新村 精装2室 轻轨口 家电齐全 拎包入住#41212770#2015年03月26日#1700元/月#付3押1#2室2厅1卫#整租#普通住宅#精装修#" \
    "80平米#南北#5/5#水香七村#苏州-吴中-龙西#床空调电视冰箱洗衣机热水器宽带可做饭独立卫生间阳台#鲍张洋#137 7191 7123#万腾房产#先奇店#" \
    "http://su.zu.anjuke.com/fangyuan/41212770?from=Filter_1"
y_test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
des = '#房源亮点1、此房为5/5层2室2厅1卫，2房朝南二个房间和客厅都有空调（共3个空调），温馨装修，干净清爽，布艺沙发，液晶电视，独立卧室，独立厨卫，房间都带有窗；2、自住精装修，品牌家具家电，温馨格局，色调柔和，拎包入住；3、现在空关出租，找爱干净的朋友入住4、此房价格低于市场价格，性价比高，房东诚心出租，找爱家的人；个人介绍万腾房产（苏州）-鲍叶春倾情为您推荐随时恭候您的来电！！！24小时咨询电话：13771801606QQ/微信：13771801606'
x = l.split('#')
# x = l.replace('#', ' ')
# x_raw = remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(x)))
# print(x_raw)

#
# l3 = "急租 盘蠡新村 精装2室 轻轨口 家电齐全 拎包入住#41212770#2015年03月26日#1700元/月#付3押1#2室2厅1卫#整租#普通住宅#精装修#80平米#南北#5/5#水香七村#苏州-吴中-龙西#床空调电视冰箱洗衣机热水器宽带可做饭独立卫生间阳台#鲍张洋#137 7191 7123#万腾房产#先奇店#http://su.zu.anjuke.com/fangyuan/41212770?from=Filter_1"
# l2 = l3.replace('#', ' ')
#
# r_raw2 = remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(l2)))
# wins = build_all_windows2(' '.join(r_raw2))     # wins == x
# print(wins)
# print(len(wins))
x_raw = build_all_windows2(l)

# x_raw = []
# for i in x:
#     x_raw.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number2(i))))
#
print(x_raw)
# print(len(x_raw))


# reload vocab
vocab = load_dict('second_hand_house_complete_dict.pickle')

# Parameters
# ==================================================
checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/second_hand_house/runs/1488294385/checkpoints'
max_sample_length = 29

input_samples = map_word2index(x_raw, vocab)
print(input_samples)
input_padding_samples = makePaddedList2(max_sample_length, input_samples, 0)
# print(input_padding_samples)


# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        sess.run(tf.all_variables())

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        loss = graph.get_operation_by_name("output/scores").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        feed_dict = {
            input_x: input_padding_samples,
            dropout_keep_prob: 1.0,     # set 0.5 at train step
        }

        loss = sess.run(loss, feed_dict=feed_dict)
        print("loss:", loss)
        softmax_loss = tf.nn.softmax(loss)
        print("softmax loss:", sess.run(softmax_loss))
        predictions = sess.run(predictions, feed_dict=feed_dict)
        print("predictions:", predictions)

    # if y_test is not None:
    #     correct_predictions = float(sum(predictions == y_test))
    #     print("Total number of test examples: {}".format(len(y_test)))
    #     Accuracy = correct_predictions/float(len(y_test))
    #     print("Accuracy:", Accuracy)
        # save result
        # print(str(i) + '   ' + class_dict.get(i))
        # result_path = 'result/'+class_dict.get(i)+'_result_ont-hot.txt'
        # save_experiment_result_secondhand2(result_path, x_raw, y_test, predictions, Accuracy)
