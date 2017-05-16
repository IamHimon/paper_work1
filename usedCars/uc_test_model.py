from SHH_testdata.generate_dataset import *
import tensorflow as tf
from publication.tools import *
from blocking.block import *
from usedCars.tools import *

ANCHOR_THRESHOLD_VALUE = 1

KB = load_kb_us()

# print("build vocab:")
print('reload vocab:')
vocab = load_dict('used_car_complete_dict.pickle')
pos_vocab = load_dict('pos.pickle')
print('load vocab over!')

checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/usedCars/runs/1494942170/checkpoints'
max_length = 21

# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        device_count={"CPU": 4},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        with tf.device('/gpu:0'):
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            sess.run(tf.all_variables())

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_pos = graph.get_operation_by_name("input_pos").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            loss = graph.get_operation_by_name("output/scores").outputs[0]
            cnn_predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            print('Reading data:')
            # for line in lines:
            # line = '[ 我 爱 我家 全优 房源 ] 金枫苑 ， 精装 2 房 ， 自 住 ， 首次 出 ...,2015 年 03 月 16 日,2800 元 / 月,面议,1 室 2 厅 1 卫,71 平米,1 / 15,床 空调 电视 冰箱 洗衣机 热水器 宽带 地铁 信息 ： 紧邻 1 号线 滨河路 站'
            line = '2016 Mazda 3 Sedan Touring,$29335,BN Series Touring Sedan 4dr SKYACTIV-Drive 6sp 2.0i [May],0,Soul Red,6 speed Automatic,4 doors 5 seats Sedan,4 cylinder Petrol - Unleaded ULP Aspirated AspiratedL,5.7 (L/100km)'
            line2 ='2017 Subaru Impreza Hatch 2.0i,$25190,G5 2.0i Hatchback 5dr CVT 7sp AWD [MY17],0,Crystal White,7 speed Automatic,5 doors 5 seats Hatch,4 cylinder Petrol - Unleaded ULP Aspirated AspiratedL,6.6 (L/100km)'
            l2 = '2016 Mazda 3 Sedan Touring'
            l3 = '$29335'
            blocks, anchors = doBlock5(line2, KB, USED_CAR_DICT, threshold=0.95)
            print(blocks)
            print(anchors)
            re_blocks, re_anchors = re_block(blocks, anchors)
            print(re_blocks)
            print(re_anchors)

            if len_Unknown2(re_anchors, USED_CAR_DICT) and len(re_anchors) >= len(USED_CAR_DICT):
                temp_list = []
                for r in do_blocking2(re_blocks, re_anchors, len(USED_CAR_DICT), USED_CAR_DICT):
                    print('result:', r)
                    print('---------------------------')
                    # print(r[0])
                    # 用sample_pretreatment_disperse_number2处理一下: '105-107' ==> '1 0 5 - 1 0 7'
                    x_raw = [sample_pretreatment_disperse_number2(x).strip() for x in r[0]]
                    input_list = [x.split() for x in x_raw]
                    print(input_list)
                    y_test = r[1]
                    print(x_raw)
                    print(y_test)
                    end0 = time.time()

                    # build input_x padding
                    input_samples = map_word2index(input_list, vocab)
                    # print(input_samples)
                    input_padding_samples = makePaddedList2(max_length, input_samples, 0)
                    # build pos padding
                    input_pad = makePosFeatures(input_list)
                    # print(input_pad)
                    pos_raw = map_word2index(input_pad, pos_vocab)
                    input_pos_padding = makePaddedList2(max_length, pos_raw, 0)

                    feed_dict = {
                        input_x: input_padding_samples,
                        input_pos: input_pos_padding,
                        dropout_keep_prob: 1.0,     # set 0.5 at train step
                    }
                    loss = sess.run(loss, feed_dict=feed_dict)
                    # print("loss:", loss)
                    softmax_loss = sess.run(tf.nn.softmax(loss))
                    print("softmax loss:", softmax_loss)

                    cnn_predictions = sess.run(cnn_predictions, feed_dict=feed_dict)
                    print("predictions:", cnn_predictions)
                    loss_max = tf.reduce_max(softmax_loss, reduction_indices=1)
                    print('loss_max:', sess.run(loss_max))
                    score = tf.reduce_sum(loss_max)
                    print('score:', sess.run(score))

                    g_predictions, g_loss_max = greddy_predictions(softmax_loss, np.arange(len(softmax_loss[0])))
                    print('g_prediction:', g_predictions)
                    print('g_loss_max:', g_loss_max)
                    g_score = sess.run(tf.reduce_sum(g_loss_max))
                    print('g_score:', g_score)

                    end1 = time.time()
                    print('time consuming:', str(end1 - end0))

                    temp_list.append([(r[0], r[1], g_predictions), g_score])

                    # Initialize loss and cnn_predictions again in this for loop
                    loss = graph.get_operation_by_name("output/scores").outputs[0]
                    # softmax_loss = graph.get_operation_by_name("output/soft_score").outputs[0]
                    cnn_predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                print('max score result:')
                # print(temp_list)
                result = max_tensor_score(temp_list, sess)
                # pre = [str(x) for x in result[2]]
                print(result)
                print(result[0])    # blocks
                print(result[1])    # labels
                print(result[2])    # predictions

                print(' || '.join(result[0]) + '\n')
                print('[' + ', '.join(result[1]) + ']' + '\n')
                print('[' + ', '.join(result[2]) + ']' + '\n')
                print('\n')

            else:
                print(' || '.join(re_blocks) + '\n')
                print('[' + ', '.join(re_anchors) + ']' + '\n')
                dict_label = lambda x: USED_CAR_DICT.get(x)
                dict_labels = [dict_label(an) for an in re_anchors]
                print(re_blocks)
                print(re_anchors)
                print(dict_labels)

            print("###############################################")


