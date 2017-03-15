import tensorflow as tf
from second_hand_house.toolbox import *
# x = ["山水华庭 交通便利 豪华婚房 首次出租 只求爱干净人士入住",	"41020070",	"2015年03月22日", "2700元/月",
#     "付3押1","2室2厅1卫",	"整租","普通住宅","精装修","100平米	", "南北","10/11","山水华庭",	"苏州-吴中-木渎",
#      "床空调电视冰箱洗衣机热水器宽带可做饭独立卫生间阳台 地铁信息： 紧邻1号线玉山路站","郭梦慧","152 6235 1319",
#      "感谢您光临克尔达地产山水华庭板块网店，本人郑重承诺此房源真实有效一经出租，本人将以第一时间下架1.山水华庭位于长江路南段2、"
#      "室内设施齐全，装修新颖，洗衣机、冰箱、精品衣柜、宽带、数字电视和热水器。房子为1室，适合家庭或者单位同事合租。3、房东诚心出租，"
#      "望有意者与我联系。如果以上房子不能满足您的需求请您联系我，我会按照您的需求以第一时间找到您需要的房子克尔达地产郭梦慧 15262351319",
#      "http://su.zu.anjuke.com/fangyuan/41020070"]
#
# y_test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#
# l = "急租 盘蠡新村 精装2室 轻轨口 家电齐全 拎包入住#41212770#2015年03月26日#1700元/月#付3押1#2室2厅1卫#整租#普通住宅#精装修#80平米#南北#5/5#水香七村#苏州-吴中-龙西#床空调电视冰箱洗衣机热水器宽带可做饭独立卫生间阳台#鲍张洋#137 7191 7123#万腾房产#先奇店#房源亮点1、此房为5/5层2室2厅1卫，2房朝南二个房间和客厅都有空调（共3个空调），温馨装修，干净清爽，布艺沙发，液晶电视，独立卧室，独立厨卫，房间都带有窗；2、自住精装修，品牌家具家电，温馨格局，色调柔和，拎包入住；3、现在空关出租，找爱干净的朋友入住4、此房价格低于市场价格，性价比高，房东诚心出租，找爱家的人；个人介绍万腾房产（苏州）-鲍叶春倾情为您推荐随时恭候您的来电！！！24小时咨询电话：13771801606QQ/微信：13771801606#http://su.zu.anjuke.com/fangyuan/41212770?from=Filter_1"
# y_test = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# x = l.split('#')
#
#
#
# x_raw = []
# for i in x:
#     x_raw.append(remove_black_space(jieba.lcut(sample_pretreatment_disperse_number(i))))
#
# print(x_raw)
# print(len(x_raw))

# Parameters
# ==================================================
checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/second_hand_house/runs/1488245487/checkpoints'
max_sample_length = 29

# reload vocab
vocab = load_dict('second_hand_house_complete_dict.pickle')

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

        titles, titles_labels,houseIDs, houseIDs_labels,publish_times,publish_times_labels,rents,rents_labels,charge_methods,charge_methods_labels,\
        units,units_labels,rental_models,rental_models_labels,house_types,house_types_labels,decorations,decorations_labels,areas,areas_labels,\
        orientations,orientations_labels,floors,floors_labels,residential_areas,residential_areas_labels,locations,locations_labels,configurations,\
        configurations_labels,contact_persons,contact_persons_labels,phone_numbers,phone_numbers_labels,companies,companies_labels,storefronts,\
        storefronts_labels,urls,urls_labels = load_data4test(21, 10)

        all_sample = [ titles, titles_labels,houseIDs, houseIDs_labels,publish_times,publish_times_labels,rents,rents_labels,charge_methods,charge_methods_labels,
        units,units_labels,rental_models,rental_models_labels,house_types,house_types_labels,decorations,decorations_labels,areas,areas_labels,
        orientations,orientations_labels,floors,floors_labels,residential_areas,residential_areas_labels,locations,locations_labels,configurations,
        configurations_labels,contact_persons,contact_persons_labels,phone_numbers,phone_numbers_labels,companies,companies_labels,storefronts,
        storefronts_labels,urls,urls_labels]

        print("================================")

        for i in range(20):
            x_raw = all_sample[2 * i]
            y_test = all_sample[2 * i + 1]
            # print('x_raw:', x_raw)
            # print('y_test:', y_test)

            input_samples = map_word2index(x_raw, vocab)
            # print(input_samples)
            input_padding_samples = makePaddedList2(max_sample_length, input_samples, 0)
            # print(input_padding_samples)

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

            if y_test is not None:
                correct_predictions = float(sum(predictions == y_test))
                print("Total number of test examples: {}".format(len(y_test)))
                Accuracy = correct_predictions/float(len(y_test))
                print("Accuracy:", Accuracy)
                # save result
                # print(str(i) + '   ' + class_dict.get(i))
                # result_path = 'result/'+class_dict.get(i)+'_result_ont-hot.txt'
                # save_experiment_result_secondhand2(result_path, x_raw, y_test, predictions, Accuracy)

            loss = graph.get_operation_by_name("output/scores").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
