# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score, precision_score, recall_score
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import time
import tensorflow as tf
import data_helpers as dh
from textCNN import TextCNN


# Data Parameters
tf.flags.DEFINE_string("training_data_file", "./data/0/train_node_set.txt", "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", "./data/0/test_node_set.txt", "Data source for the validation data.")
tf.flags.DEFINE_string("test_data_file", "./data/test_node_set.txt", "Data source for the test data.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.02, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("pad_seq_len", 4, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("vocab_size", 5, "vocabulary size (default: 5000)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 512, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 4, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 2, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 1, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 5000)")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_integer("decay_steps", 400, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_integer("checkpoint_every", 150, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 10)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
para_key_values = FLAGS.flag_values_dict()

# logger = dh.logger_fn('tflog', './logs/train_node_.log')
logger = dh.logger_fn('tflog', './logs/train_CNN0.log')
logger.info("input parameter:")
parameter_info = " ".join(["\nparameter: %s, value: %s" % (key, val) for key, val in para_key_values.items()])
logger.info(parameter_info)

print("load train and val data sets.....")
x_train, y_train = dh.process_file(FLAGS.training_data_file)
x_val, y_val = dh.process_file(FLAGS.validation_data_file)
_, y_ = dh.read_file(FLAGS.validation_data_file)
# x_predict, y_ = dh.pad_seq_label(x_predict, y_, pad_seq_len, FLAGS.num_classes)
y_true = list(map(int, y_))
# x_test, y_test = dh.process_file(FLAGS.test_data_file)

# 得到所有数据中最长文本长度
# pad_seq_len = dh.get_pad_seq_len(x_train, x_val, x_test)
pad_seq_len = FLAGS.pad_seq_len
# 将数据pad为统一长度，同时对label进行0，1编码
# x_train, y_train = dh.pad_seq_label(x_train, y_train, pad_seq_len, FLAGS.num_classes)
# x_val, y_val = dh.pad_seq_label(x_val, y_val, pad_seq_len, FLAGS.num_classes)
# x_test, y_test = dh.pad_seq_label(x_test, y_test, pad_seq_len, FLAGS.num_classes)

def train():
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.8
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("init model .....")
            global_loss = 20
            cnn = TextCNN(
                sequence_length=pad_seq_len,
                num_classes=FLAGS.num_classes,
                vocab_size=FLAGS.vocab_size,
                fc_hidden_size=FLAGS.fc_hidden_size,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                node_pretrained_embedding=None,
                text_pretrained_embedding=None,
                model='CNN',
            )

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=cnn.global_step, decay_steps=FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(cnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=cnn.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
            logger.info("Writing to {0}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            prec_summary = tf.summary.scalar("precision-micro", cnn.precision)
            rec_summary = tf.summary.scalar("recall-micro", cnn.recall)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, prec_summary, rec_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary, prec_summary, rec_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            current_step = sess.run(cnn.global_step)

            def train_step(x_batch, y_batch):
                """A single training step"""
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss = sess.run([train_op, cnn.global_step, train_summary_op,
                                                     cnn.loss], feed_dict)

                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

                return loss

            def validation_step(x_validation, y_validation, writer=None):
                """Evaluates model on a validation set"""

                feed_dict = {
                    cnn.input_x: x_validation,
                    cnn.input_y: y_validation,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, scores, cur_loss = sess.run([cnn.global_step, validation_summary_op, cnn.scores,
                                                              cnn.loss], feed_dict)
                y_pre = np.argmax(scores, axis=-1)
                a = (f1_score(y_true=y_true, y_pred=y_pre, average='macro'))
                b = (precision_score(y_true=y_true, y_pred=y_pre, average='macro'))
                c = (recall_score(y_true=y_true, y_pred=y_pre, average='macro'))
                print(a, b, c)

            # Generate batches
            batches_train = dh.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            num_batches_per_epoch = int((len(x_train) - 1) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            current_loss = 0
            for batch_train in batches_train:
                x_batch_train, y_batch_train = zip(*batch_train)
                current_loss = train_step(x_batch_train, y_batch_train)
                current_step = tf.train.global_step(sess, cnn.global_step)

                print('current_step:', current_step, ' current_loss:', current_loss)
                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    validation_step(x_val, y_val, writer=validation_summary_writer)

                if current_step % FLAGS.checkpoint_every == 0 and current_loss < global_loss:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))
                    global_loss = current_loss

                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("Epoch {0} has finished!".format(current_epoch))
                    print("Epoch {0} has finished!".format(current_epoch))

            logger.info("\nlast batch")
            logger.info("\nEvaluation:")
            validation_step(x_val, y_val, writer=validation_summary_writer)

            if current_loss < global_loss:
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {0}\n".format(path))

        logger.info("Done.")

def cos(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    sum = tf.reduce_sum(tf.multiply(x, y))
    x_ = tf.sqrt(tf.reduce_sum(tf.multiply(x, x)))
    y_ = tf.sqrt(tf.reduce_sum(tf.multiply(y, y)))
    return sum / (x_ * y_)

def E_L(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    y_ = tf.sqrt(tf.reduce_sum(tf.multiply((x - y), (x - y))))
    return y_


if __name__ == '__main__':
    # 载入embedding
    # _, node_embd = dh.loadWord2Vec('./json/node.emb')
    # node_embedding = np.asarray(node_embd)
    # _, text_embd = dh.loadWord2Vec('./json/text.emb')
    # text_embedding = np.asarray(text_embd)
    train()
    # print(text_embedding[4], text_embedding[96], text_embedding[3])

