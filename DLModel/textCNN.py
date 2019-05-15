# -*- coding: utf-8 -*-

import tensorflow as tf

class TextCNN(object):
    """A CNN for text classification."""

    def __init__(
            self, sequence_length, num_classes, vocab_size, fc_hidden_size, embedding_size,
            embedding_type, filter_sizes, num_filters, l2_reg_lambda=0.0, node_pretrained_embedding=None,
            text_pretrained_embedding=None, model='RNN',):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope("node_embedding"):
            if node_pretrained_embedding is None:
                self.node_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0,
                                                               dtype=tf.float32), trainable=True, name="node_embedding")
            else:
                if embedding_type == 0:
                    self.node_embedding = tf.constant(node_pretrained_embedding, dtype=tf.float32, name="node_embedding")
                if embedding_type == 1:
                    self.node_embedding = tf.Variable(node_pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="node_embedding")
            self.embedded_sentence = tf.nn.embedding_lookup(self.node_embedding, self.input_x)
        if model == 'RNN':
            with tf.name_scope("Bi-LSTM"):
                # 定义前向LSTM结构
                lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=128, state_is_tuple=True),
                    output_keep_prob=self.dropout_keep_prob)
                # 定义反向LSTM结构
                lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                    tf.nn.rnn_cell.LSTMCell(num_units=128, state_is_tuple=True),
                    output_keep_prob=self.dropout_keep_prob)
                outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                               self.embedded_sentence,
                                                                               dtype=tf.float32,
                                                                               scope="bi-lstm")
                self.embedded_sentence = tf.reshape(tf.concat(outputs_, 2), shape=[-1, sequence_length * embedding_size * 2])
                self.fc_out = tf.layers.dense(self.embedded_sentence, fc_hidden_size, name='fc_out')
        elif model == 'CNN':
            self.embedded_sentence_expanded = tf.expand_dims(self.embedded_sentence, -1)
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size in filter_sizes:
                with tf.name_scope("conv-filter{0}".format(filter_size)):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, dtype=tf.float32), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters], dtype=tf.float32), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_sentence_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    conv = tf.nn.bias_add(conv, b)
                    conv_out = tf.nn.relu(conv, name="relu")

                with tf.name_scope("pool-filter{0}".format(filter_size)):
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        conv_out,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")

                pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.pool = tf.concat(pooled_outputs, 3)
            self.pool_flat = tf.reshape(self.pool, [-1, num_filters_total])

            # Fully Connected Layer
            with tf.name_scope("fc"):
                w = tf.Variable(tf.truncated_normal(shape=[num_filters_total, fc_hidden_size],
                                                    stddev=0.1, dtype=tf.float32), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
                self.fc = tf.nn.xw_plus_b(self.pool_flat, w, b)

                # Apply nonlinearity
                self.fc_out = tf.nn.relu(self.fc, name="relu")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.fc_out, self.dropout_keep_prob)

        # Final scores
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.scores = tf.nn.softmax(self.logits, name="scores")
            self.predictions = tf.round(self.scores, name="predictions")

        # if model == "RNN":
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
                losses = tf.reduce_mean(losses, name="sigmoid_losses")
                l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                     name="l2_losses") * l2_reg_lambda
                self.loss = tf.add(losses, l2_losses, name="loss")
        # elif model == 'CNN':
        #     with tf.name_scope("loss"):
        #         losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        #         losses = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name="sigmoid_losses")
        #         l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
        #                              name="l2_losses") * l2_reg_lambda
        #         self.loss = tf.add(losses, l2_losses, name="loss")

        # Calculate performance
        with tf.name_scope('performance'):
            self.precision = tf.metrics.precision(self.input_y, self.predictions, name="precision-micro")[1]
            self.recall = tf.metrics.recall(self.input_y, self.predictions, name="recall-micro")[1]

