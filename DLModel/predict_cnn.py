# -*- coding:utf-8 -*-
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import os
import numpy as np
import time
import tensorflow as tf
import data_helpers as dh
import json

# Parameters
# ==================================================

id_to_cat = json.load(open("./json/category_id.json", 'r', encoding='utf-8'))['id_to_cat']

# Data Parameters
# tf.flags.DEFINE_string("training_data_file", "./data/0/train_node_set.txt", "Data source for the training data.")
# tf.flags.DEFINE_string("validation_data_file", "./data/val_node_set.txt", "Data source for the validation data")
# tf.flags.DEFINE_string("test_data_file", "./data/test_node_set.txt", "Data source for the test data")
tf.flags.DEFINE_string("predict_data_file", "./data/0/test_node_set.txt", "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints/", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("vocab_data_file", "./", "Vocabulary file")

# Model Hyperparameters
tf.flags.DEFINE_integer("pad_seq_len", 70, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 123, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 1, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")

# Test Parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
para_key_values = FLAGS.flag_values_dict()

logger = dh.logger_fn('tflog', 'logs/predict_node'+time.strftime("%m-%d-%Y-%H-%M-%S")+'.log')
logger.info("input parameter:")
# parameter_info = " ".join(["\nparameter: {0:<30} value: {1:<50}".format(key, val) for key, val in para_key_values.items()])
parameter_info = " ".join(["\nparameter: %s, value: %s" % (key, val) for key, val in para_key_values.items()])
logger.info(parameter_info)


# print("load train and val data sets.....")
# logger.info('Test data processing...')
# x_train, y_train = dh.process_file(FLAGS.training_data_file)
# x_val, y_val = dh.process_file(FLAGS.validation_data_file)
# x_test, y_test = dh.process_file(FLAGS.test_data_file)

# 得到所有数据中最长文本长度
pad_seq_len = FLAGS.pad_seq_len

# 将数据pad为统一长度，同时对label进行0，1编码
# x_predict = dh.process_data_for_predict(FLAGS.predict_data_file, pad_seq_len)
x_predict, y_ = dh.read_file(FLAGS.predict_data_file)
# x_predict, y_ = dh.pad_seq_label(x_predict, y_, pad_seq_len, FLAGS.num_classes)
y_true = list(map(int, y_))
def predict():
    """Predict Use TextCNN model."""

    # Load cnn model
    logger.info("Loading model...\n")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            feed_dict = {
                input_x: x_predict,
                dropout_keep_prob: 1.0,
            }
            batch_scores = sess.run(scores, feed_dict)
            y_pre = np.argmax(batch_scores, axis=-1)
            print(f1_score(y_true=y_true, y_pred=y_pre, average='macro'))
            print(precision_score(y_true=y_true, y_pred=y_pre, average='macro'))
            print(recall_score(y_true=y_true, y_pred=y_pre, average='macro'))


if __name__ == '__main__':
    predict()
