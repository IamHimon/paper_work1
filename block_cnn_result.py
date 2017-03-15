import tensorflow as tf
from utils import *
from load_word2vec import *
from v3.v3_utils import *
from second_hand_house.hello import *
from blocking.block import *


author_fp = 'dataset_workshop/temp_authors_kb.txt'
title_fp = 'dataset_workshop/temp_titles_kb.txt'
journal_fp = 'v3/all_journal_1614_.txt'
year_fp = 'dataset_workshop/year_kb.txt'
volume_fp = 'dataset_workshop/volume_kb.txt'
pages_fp = 'dataset_workshop/temp_page_kb.txt'
KB = loadKB2(author_fp=author_fp, title_fp=title_fp,journal_fp=journal_fp,year_fp=year_fp,volume_fp=volume_fp,pages_fp=pages_fp)

fo = open('dataset_workshop/temp_dataset2.txt', 'r')
lines = fo.readlines()
random.shuffle(lines)

# Parameters
# ==================================================
# checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/runs/1487144642/checkpoints'    # word2vec/ not-tf
checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/runs/1487824860/checkpoints'      # one-hot/ not-tf
# checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/runs/1489392243/checkpoints'      # dataset shrink 50%
# load word2vec array
print("loading word2vec:")
path = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
vocab, embedding = load_from_binary(path)
vocab_size, embedding_dim = embedding.shape


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
        # word_placeholder = graph.get_operation_by_name("embedding/word_placeholder").outputs[0]
        # Tensors we want to evaluate
        loss = graph.get_operation_by_name("output/scores").outputs[0]
        # softmax_loss = graph.get_operation_by_name("output/soft_score").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        # block
        for line in lines:
            print(line.strip())
            all_blocks = doBlock2(line, KB)
            labels, blocks = dict2list(all_blocks)
            selectSort(line, label=labels, block=blocks)
            print(blocks)
            print(labels)
            x_raw = blocks

            x_numeric = []
            x_text = []
            numeric_index = []
            textual_index = []
            for x in x_raw:
                token = n_or_t(x)
                if token == 't':
                    x_text.append(x.strip())
                    textual_index.append(x_raw.index(x))
                if token == 'n':
                    x_numeric.append(x.strip())
                    numeric_index.append(x_raw.index(x))
            # print(x_text)
            # print(x_numeric)
            input_list = [x.split() for x in x_text]
            # print(input_list)
            input_pad = makePaddedList2(27, input_list, 0)
            # print(input_pad)

            input_samples = sample2index_matrix(input_pad, vocab, 27)
            # print(input_samples)

            feed_dict = {
                input_x: input_samples,
                dropout_keep_prob: 1.0,     # set 0.5 at train step
            }

            # loss = sess.run(loss, feed_dict=feed_dict)
            # print("loss:", loss)
            # softmax_loss = tf.nn.softmax(loss)
            # print("softmax loss:", sess.run(softmax_loss))
            predictions = sess.run(predictions, feed_dict=feed_dict)
            # print("predictions:", predictions)
            # print("Generating predictions for numeric block: ")
            num_predictions = label_numeric(x_numeric)
            # print("numeric predictions:", num_predictions)
            predictions = merge_predictions(predictions, textual_index, num_predictions, numeric_index)
            print("merged prediction:", predictions)
            print('================================')

            # input_x = graph.get_operation_by_name("input_x").outputs[0]
            # dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # loss = graph.get_operation_by_name("output/scores").outputs[0]
            # softmax_loss = graph.get_operation_by_name("output/soft_score").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]




