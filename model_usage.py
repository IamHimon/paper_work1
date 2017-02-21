import tensorflow as tf
from utils import *
from load_word2vec import *
from v3.v3_utils import *


# prepare test data
# # ==================================================
# x_raw = ["The Interaction between Schema Matching and Record Matching in Data Integration",
#          "Binbin Gu",
#          "Zhixu Li",
#          "Web-ADARE: A Web-Aided Data Repairing System",
#          "AML:Efficient Approximate Membership Localization within a Web-Based Join Framework",
#          "Journal of Intelligent and Fuzzy Systems",
#          "IJTM",
#          "JCSE",
#          "IEEE Wireless Commun",
#          "TOMCCAP",
#          "Progress in AI",
#          "Meng Hu",
#          "Optimizing Cost of  Continuous Overlapping Queries over Data Streams by Filter Adaption"]
# y_test = [0, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 1, 0]

# p1 = "Binbin Gu, Zhixu Li, Xiangliang Zhang, An Liu, Guanfeng Liu, Kai Zheng, Lei Zhao, Xiaofang Zhou," \
#      " The Interaction between Schema Matching and Record Matching in Data Integration, " \
#      "International journal of Distributed and Parallel Databases"
p1 = "Zhixu Li, Mohamed A. Sharaf, Laurianne Sitbon, Shazia Wasim Sadiq, Marta Indulska, Xiaofang Zhou," \
     " A Web-based Approach to Data Imputation," \
     " World Wide Web Journal (WWWJ) "
p3 = "Optimizing Seam Carving on Multi-GPU Systems for Real-time Image Resizing,Ikjoon Kim,Jidong Zhai," \
     "Yan Li,Wenguang Chen,The 20th IEEE International Conference on Parallel and Distributed Systems"
x_raw = p3.split(',')
# print(x_raw)
# y_test = [1, 1, 1, 1, 1, 1, 1, 1, 0, 2]
# y_test = [1, 1, 1, 1, 1, 1, 0, 2]
y_test = [0, 1, 1, 1, 1, 2]

# x_raw, y_test = load_title4test("v3/titles4test.txt")
# x_raw, y_test = load_author4test("v3/authors4test.txt")
# x_raw, y_test = load_journal4test("v3/all_journal_1614_.txt")



# Parameters
# ==================================================
checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/runs/1487144642/checkpoints'
# checkpoint_dir = '/home/himon/PycharmProjects/paper_work1/runs/1487321280/checkpoints'
# load word2vec array
print("loading word2vec:")
path = "/home/himon/Jobs/nlps/word2vec/resized_dblp_vectors.bin"
vocab, embedding = load_from_binary(path)
vocab_size, embedding_dim = embedding.shape

# max_sample_length = max(len(x.split()) for x in x_raw)
# print('max sample length:', max_sample_length)
# embedding_dim = 27
# evaluate
# ====================================================
input_list = [x.split() for x in x_raw]
# print(input_list)
input_pad = makePaddedList2(100, input_list)
print(input_pad)

input_samples = sample2index_matrix(input_pad, vocab, 100)
print(input_samples)


# 构建词频特征
print("build term frequency dictionary:")
tf_dic = build_tf_dic()
# print(tf_dic)
tf_dic_size = len(tf_dic) + 1

t_tf = make_title_tf_feature(input_pad)
a_tf = make_author_tf_feature(input_pad)
j_tf = make_journal_tf_feature(input_pad)

nor_t_tf, nor_a_tf, nor_j_tf = normalize_tf(t_tf, a_tf, j_tf)
print(nor_t_tf)
print(nor_a_tf)
print(nor_j_tf)
t_tf_sample = mapWordToId(nor_t_tf, tf_dic)
a_tf_sample = mapWordToId(nor_a_tf, tf_dic)
j_tf_sample = mapWordToId(nor_j_tf, tf_dic)
print(t_tf_sample)
print(a_tf_sample)
print(j_tf_sample)


# Evaluation
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
        print()

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        t_tf = graph.get_operation_by_name("t_tf").outputs[0]
        a_tf = graph.get_operation_by_name("a_tf").outputs[0]
        j_tf = graph.get_operation_by_name("j_tf").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        word_placeholder = graph.get_operation_by_name("embedding/word_placeholder").outputs[0]
        # Tensors we want to evaluate
        loss = graph.get_operation_by_name("output/scores").outputs[0]
        # softmax_loss = graph.get_operation_by_name("output/soft_score").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        # tf_vec = graph.get_all_collection_keys()
        # TF_vec = sess.run(tf_vec)
        # print(TF_vec)

        feed_dict = {
            t_tf: t_tf_sample,
            a_tf: a_tf_sample,
            j_tf: j_tf_sample,
            input_x: input_samples,
            dropout_keep_prob: 1.0,     # set 0.5 at train step
            word_placeholder: embedding
        }

        loss = sess.run(loss, feed_dict=feed_dict)
        print("loss:", loss)
        softmax_loss = tf.nn.softmax(loss)
        print("softmax loss:", sess.run(softmax_loss))
        predictions = sess.run(predictions, feed_dict=feed_dict)
        print("predictions:", predictions)

        predictions = revise_predictions(predictions, loss)
        print("predictions after revise:", predictions)


# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    Accuracy = correct_predictions/float(len(y_test))
    print("Accuracy:", Accuracy)
    # save result
    # result_path = 'result/titles_result_tf.txt'
    # save_experiment_result(result_path, x_raw, y_test, predictions, Accuracy)
