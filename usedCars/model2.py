from SHH_testdata.generate_dataset import *
import tensorflow as tf
from publication.tools import *
from blocking.block import *
from usedCars.tools import *
from usedCars.model2_tool import *

ANCHOR_THRESHOLD_VALUE = 1

KB = load_kb_us()

# print("build vocab:")
print('reload vocab:')
vocab = load_dict('uc_complete_dict.pickle')
pos_vocab = load_dict('pos.pickle')
print('load vocab over!')

# checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/usedCars/runs/1494942170/checkpoints'
checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/usedCars/runs/1495079056/checkpoints'
max_length = 27

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
            line4 = '2015 Audi,$48999,8V Cabriolet 2dr S tronic 7sp 1.4T (CoD) [MY17],4200,Mythos Black,7 speed Automatic,2 doors 4 seats Convertible,4 cylinder Petrol - Premium ULP Turbo Intercooled Turbo IntercooledL,5.1 (L/100km)'
            blocks, anchors = doBlock5(line4, KB, USED_CAR_DICT, threshold=1)
            print(blocks)
            print(anchors)
            re_blocks, re_anchors = re_block(blocks, anchors)
            print(re_blocks)
            print(re_anchors)            #

            x_raw = [sample_pretreatment_disperse_number2(x).strip() for x in re_blocks]
            input_list = [x.split() for x in x_raw]
            # print(input_list)

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
            # print("softmax loss:", softmax_loss)

            greedy_label = greedy_labeling(re_anchors, softmax_loss, 0.6)
            greedy_blocks, greedy_labels = re_construct_block(re_blocks, greedy_label)
            print(greedy_blocks)
            print(greedy_labels)

            loss = graph.get_operation_by_name("output/scores").outputs[0]
            # cnn_predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            print('All Combination:')
            if len_Unknown2(greedy_labels, USED_CAR_DICT) and len(greedy_labels) >= len(USED_CAR_DICT):
                temp_list = []
                for r in do_blocking2(greedy_blocks, greedy_labels, len(USED_CAR_DICT), USED_CAR_DICT):
                    # print('result:', r)
                    print('---------------------------')
                    # print(r[0])
                    # 用sample_pretreatment_disperse_number2处理一下: '105-107' ==> '1 0 5 - 1 0 7'
                    x_raw = [sample_pretreatment_disperse_number2(x).strip() for x in r[0]]
                    input_list = [x.split() for x in x_raw]
                    print(input_list)
                    # y_test = r[1]
                    # print(x_raw)
                    # print(y_test)

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

                    g_predictions, g_loss_max = greddy_predictions(softmax_loss, np.arange(len(softmax_loss[0])))
                    print('g_prediction:', g_predictions)
                    print('g_loss_max:', g_loss_max)
                    g_score = sess.run(tf.reduce_sum(g_loss_max))
                    print('g_score:', g_score)

                    temp_list.append([(r[0], r[1], g_predictions), g_score])

                    loss = graph.get_operation_by_name("output/scores").outputs[0]
                    cnn_predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                print('max score result:')
                result = max_tensor_score(temp_list, sess)

                print(' || '.join(result[0]) + '\n')
                print('[' + ', '.join(result[1]) + ']' + '\n')
                print('[' + ', '.join(result[2]) + ']' + '\n')
                print('\n')


                # save the result to .json file
                # save2json(record_id, result_json_output, result[0], result[1], result[2])
            else:
                # dict_prediction = lambda x: USED_CAR_DICT.get(x)
                # dict_predictions = [str(dict_prediction(an)) for an in re_anchors]
                # save the result to .json file
                # save2json(record_id, result_json_output, re_blocks, re_anchors, dict_predictions)

                print(' || '.join(re_blocks) + '\n')
                print('[' + ', '.join(re_anchors) + ']' + '\n')
                dict_label = lambda x: USED_CAR_DICT.get(x)
                dict_labels = [dict_label(an) for an in re_anchors]
                print(re_blocks)
                print(re_anchors)
                print(dict_labels)

            print("###############################################")



