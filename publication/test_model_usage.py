import tensorflow as tf
import sys
sys.path.append('..')
import getopt
# from blocking.reconstruction import *
from publication.tools import *
from blocking.block import *
from utilss.utils import *

opts, args = getopt.getopt(sys.argv[1:], "hi:o:")
ANCHOR_THRESHOLD_VALUE = 0

for op, value in opts:
    if op == '-t':
        ANCHOR_THRESHOLD_VALUE = int(value)

if ANCHOR_THRESHOLD_VALUE:
    exit('Please enter threshold value!')

print("threshold value: %d", ANCHOR_THRESHOLD_VALUE)

# ANCHOR_THRESHOLD_VALUE = 0.95
result_output = 'result_'+str(ANCHOR_THRESHOLD_VALUE)+'.json'

# load Knowledge base
author_fp = '../dataset_workshop/lower_linked_authors_no_punctuation.txt'
title_fp = '../dataset_workshop/lower_temp_titles_kb.txt'
journal_fp = '../dataset_workshop/lower_all_journal.txt'
year_fp = '../dataset_workshop/year_kb.txt'
volume_fp = '../dataset_workshop/artificial_volumes.txt'
pages_fp = '../dataset_workshop/temp_page_kb.txt'
KB = loadKB2(title_fp=title_fp, author_fp=author_fp, journal_fp=journal_fp, year_fp=year_fp, volume_fp=volume_fp, pages_fp=pages_fp)
print('Building KB over!')

'''
# reload vocab
vocab = load_dict('publication_complete_dict.pickle')
pos_vocab = load_dict('pos.pickle')
print('Load vocab over!')


# line = 'Invariants, Composition, and Substitution,Acta Inf,Ekkart Kindler,32(4),299-312,1995'
# line = 'Rainer Kemp,A Note on the Density of Inherently Ambiguous Context-free Languages,Acta Inf,14,295-298,1980'


# main result: (['Dominique Fournier', 'Crémilleux', 'A quality pruning. Knowl.-Based Syst', '2002', '15 37-43'], ['Journal', '0_Backup_0_Backup_Unknown', 'Title', 'Volume', 'Pages'])
# main result: (['Dominique Fournier Crémilleux', 'A quality pruning. Knowl.-Based Syst', '2002', '15 37-43'], ['Journal', 'Title', 'Volume', 'Pages'])

# Parameters
# ==================================================
# checkpoint_dir = '/home/humeng/paper_work1/publication/runs/1490626741/checkpoints/'
checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/publication/runs/1490626741/checkpoints'

max_length = 90

print(checkpoint_dir)
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
print(checkpoint_file)
print("test")

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
            # line = 'Another Polynomial Homomorphism,Robert T. Moenck,Acta Inf,1976,6(4),153-169'
            line = 'Estimations and Optimal Designs for Two-Dimensional Haar-Wavelet Regression Models,IJWMIP,Yongge Tian,7(3),281-297,2009'
            # line = 'Hans-Dietrich O. F. Gronau,Combinatorica,On Sperner families in which no k sets have an empty intersection III,2(1),1982,25-36'
            print(line.strip())
            blocks, anchors = doBlock4(line.strip(), KB, threshold=ANCHOR_THRESHOLD_VALUE)
            print(blocks)
            print(anchors)
            re_blocks, re_anchors = re_block(blocks, anchors)
            print(re_blocks)
            print(re_anchors)
            # print('--------------')
            if len_Unknown(re_anchors) and len(re_anchors) >= len(LABEL_DICT):
                temp_list = []
                for r in do_blocking2(re_blocks, re_anchors, len(LABEL_DICT), LABEL_DICT):
                    # print('result:', r)
                    print('---------------------------')
                    # print(r[0])
                    # 用sample_pretreatment_disperse_number2处理一下: '105-107' ==> '1 0 5 - 1 0 7'
                    x_raw = [sample_pretreatment_disperse_number2(x).strip() for x in r[0]]
                    input_list = [x.lower().split() for x in x_raw]
                    y_test = r[1]
                    print(x_raw)
                    print(y_test)

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
                dict_label = lambda x: LABEL_DICT.get(x)
                dict_labels = [dict_label(an) for an in re_anchors]
                print(re_blocks)
                print(re_anchors)
                print(dict_labels)

            print("###############################################")

'''