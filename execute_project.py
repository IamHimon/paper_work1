import tensorflow as tf
from utils import *
from load_word2vec import *
import random
from blocking.block import *

time_str = datetime.datetime.now().isoformat()
write = open('result/experiment_result_'+str(time_str)+'.txt', 'w+')

# Parameters
# ==================================================
checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/runs/1489590694/checkpoints'    # linked author
# checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/runs/1487144642/checkpoints'  # word2vec/ not-tf

# load Knowledge base
author_fp = 'dataset_workshop/lower_temp_authors_kb.txt'
author_fp2 = 'dataset_workshop/lower_linked_authors_no_punctuation.txt'
title_fp = 'dataset_workshop/lower_temp_titles_kb.txt'
journal_fp = 'dataset_workshop/lower_all_journal.txt'
year_fp = 'dataset_workshop/year_kb.txt'
volume_fp = 'dataset_workshop/volume_kb.txt'
pages_fp = 'dataset_workshop/temp_page_kb.txt'
KB = loadKB2(title_fp=title_fp, author_fp=author_fp2, journal_fp=journal_fp, year_fp=year_fp,volume_fp=volume_fp, pages_fp=pages_fp)

# load word2vec array
print("loading word2vec:")
path = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
vocab, embedding = load_from_binary(path)
vocab_size, embedding_dim = embedding.shape
max_length = 51


fo = open('dataset_workshop/temp_dataset3.txt', 'r')
lines = fo.readlines()
random.shuffle(lines)


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
        cnn_predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        for line in lines:
            print(line.strip())
            blocks, anchors = doBlock4(line.strip(), KB, threshold=0.9)
            # print(blocks)
            # print(anchors)
            re_blocks, re_anchors = re_block(blocks, anchors)
            # print(re_blocks)
            # print(re_anchors)
            # print('--------------')
            for b in normal_reblock_and_relabel(re_blocks, re_anchors):
                x_raw = b[0]
                y_test = b[1]
                print(b[0])
                print(b[1])
            print('=================================================')


'''
        #  loading data for evaluation
        samples, labels = read_test_data('dataset_workshop/temp_dataset3.txt')
        for i in range(len(samples)):
            print("==================================================================")
            x_raw = samples[i].strip().split(',')
            y_test = labels[i]
            if len(x_raw) != len(y_test):
                continue
            print(x_raw)
            print(y_test)
            # write.write("y_test:" + str(y_test) + '\n')

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

            # print("Generating predictions for textual block:")
            input_list = [x.split() for x in x_text]
            input_pad = makePaddedList2(max_length, input_list, 0)
            input_samples = sample2index_matrix(input_pad, vocab, max_length)
            feed_dict = {
                input_x: input_samples,
                dropout_keep_prob: 1.0,     # set 0.5 at train step
                # word_placeholder: embedding
            }
            loss = sess.run(loss, feed_dict=feed_dict)
            # print("loss:", loss)
            softmax_loss = tf.nn.softmax(loss)
            print("softmax loss:", sess.run(softmax_loss))
            cnn_predictions = sess.run(cnn_predictions, feed_dict=feed_dict)
            print("predictions:", cnn_predictions)

            text_predictions = revise_predictions(cnn_predictions, loss)
            print("predictions after revise:", text_predictions)

            # print("Generating predictions for numeric block: ")
            num_predictions = label_numeric(x_numeric)
            print("numeric predictions:", num_predictions)

            predictions = merge_predictions(text_predictions, textual_index, num_predictions, numeric_index)
            print("merged prediction:", predictions)
            # write.write("predictions:" + str(predictions) + '\n')

            # Print accuracy if y_test is defined
            if y_test is not None:
                # print(len(predictions))
                # print(len(y_test))
                correct_predictions = float(same_elem_count(predictions, y_test))
                # print("Total number of test examples: {}".format(len(y_test)))
                Accuracy = correct_predictions/float(len(y_test))
                print("Accuracy:", Accuracy)
                # write.write("Accuracy:" + str(Accuracy) + '\n')
            # Initialize loss and cnn_predictions again in this for loop
            loss = graph.get_operation_by_name("output/scores").outputs[0]
            # softmax_loss = graph.get_operation_by_name("output/soft_score").outputs[0]
            cnn_predictions = graph.get_operation_by_name("output/predictions").outputs[0]
'''