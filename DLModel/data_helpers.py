# -*- coding: utf-8 -*-

import os
import logging
import sys
import importlib
from collections import Counter
import numpy as np
import json
import re
import random


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w', encoding='utf8')
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def pre_process_node_data(train_data_path='./../Data/KfoldData2/step2data4.npy', output_data_path="data/4",
                          val_ratio=0.17, shuffle=True):
    # 标注平台data下文件夹
    if not os.path.exists(output_data_path):
        os.mkdir(output_data_path)
    train_category_contents = []
    data = np.load(train_data_path, allow_pickle=True).item()
    for index, line in enumerate(data['x_train']):
        l = list(map(str, list(map(int, line))))
        train_category_contents.append(data['y_train'][index] + '\t' + ' '.join(l))
    test_category_contents = []
    for index, line in enumerate(data['x_test']):
        l = list(map(str, list(map(int, line))))
        test_category_contents.append(data['y_test'][index] + '\t' + ' '.join(l))

    if shuffle:
        random.shuffle(train_category_contents)

    len_files = len(train_category_contents)
    train_set = "\n".join(train_category_contents)
    # val_set = "\n".join(train_category_contents[:int(val_ratio * len_files * -1)])
    test_set = "\n".join(test_category_contents)
    all_set = "\n".join(train_category_contents + test_category_contents)

    with open(os.path.join(output_data_path, "train_node_set.txt"), 'w', encoding='utf-8') as f:
        f.write(train_set)
    # with open(os.path.join(output_data_path, "val_node_set.txt"), 'w', encoding='utf-8') as f:
    #     f.write(val_set)
    with open(os.path.join(output_data_path, "test_node_set.txt"), 'w', encoding='utf-8') as f:
        f.write(test_set)
    with open(os.path.join(output_data_path, "all_node_set.txt"), 'w', encoding='utf-8') as f:
        f.write(all_set)

    print("一共处理%s条数据！！！" % len(train_category_contents + test_category_contents))  # 一共处理37924条数据
# pre_process_node_data()

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content.split(' '))
                    # contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_size=3000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    print('words:', len(counter))
    words, _ = list(zip(*count_pairs))

    word_to_id = {}
    id_to_word = {}
    word_id = 0
    word_to_id["<PAD>"] = word_id
    id_to_word[word_id] = "<PAD>"
    for word in words:
        word_id += 1
        word_to_id[word] = word_id
        id_to_word[word_id] = word
    json.dump({"word_to_id": word_to_id, "id_to_word": id_to_word},
              open("./json/words_id.json", 'w', encoding="utf-8"), ensure_ascii=False)


def make_category_id():
    """读取分类目录，固定"""
    categories = ['1', '0']
    print('classes:', len(categories))
    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))
    id_to_cat = dict(zip(range(len(categories)), categories))
    temp = {}
    temp["cat_to_id"] = cat_to_id
    temp["id_to_cat"] = id_to_cat
    json.dump(temp, open("./json/category_id.json", "w", encoding="utf-8"), ensure_ascii=False)
    # return categories, cat_to_id

# make_category_id()
def process_file(filename):
    """将文件转换为id表示"""
    # words_id = json.load(open("./json/words_id.json", 'r', encoding='utf-8'))
    category_id = json.load(open("./json/category_id.json", 'r', encoding="utf-8"))
    #
    # word_to_id = words_id["word_to_id"]
    cat_to_id = category_id["cat_to_id"]
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(labels)):
        if labels[i] == '1':
            label_id.append([0, 1])
        else:
            label_id.append([1, 0])
    # print(data_id, label_id)
    return contents, label_id


def get_pad_seq_len(train_set, val_set, test_set):
    """
    返回样本中最长的文章长度
    :param train_set:
    :param val_set:
    :param test_set:
    :return:
    """
    max_train_len, max_val_len, max_test_len = max([len(item) for item in train_set]), \
                                               max([len(item) for item in val_set]), \
                                               max([len(item) for item in test_set])
    return max([max_train_len, max_val_len, max_test_len])


def pad_seq_label(sequences, labels, pad_seq_len, num_class):
    """
    讲数据编码成统一长度，同时将标签编码成0,1表示
    :param sequences: 统一长度之前文章的数字表示
    :param labels: 编码之前标签的数字表示
    :param pad_seq_len: 需要编码成的文章长度
    :return:
    """
    pad_data = []
    pad_label = []
    for seq in sequences:
        temp = [0] * pad_seq_len
        if len(seq) < pad_seq_len:
            temp[:len(seq)] = seq
        else:
            temp = seq[:pad_seq_len]
        pad_data.append(temp)

    for label in labels:
        temp = [0] * num_class
        for id in label:
            temp[id] = 1
        pad_label.append(temp)
    return pad_data, pad_label


def cal_metric(predicted_labels, labels):
    """
    Calculate the metric(recall, accuracy, F, etc.).

    Args:
        predicted_labels: The predicted_labels
        labels: The true labels
    Returns:
        The value of metric
    """
    label_no_zero = []
    for index, label in enumerate(labels):
        if int(label) == 1:
            label_no_zero.append(index)
    count = 0
    for predicted_label in predicted_labels:
        if int(predicted_label) in label_no_zero:
            count += 1
    rec = count / len(label_no_zero)
    acc = count / len(predicted_labels)
    if (rec + acc) == 0:
        F = 0.0
    else:
        F = (2 * rec * acc) / (rec + acc)
    return rec, acc, F


def get_label_using_scores_by_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict value greater than threshold, then choose the label which has the max predict value.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_values: The predicted values
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        value_list = []
        for index, predict_value in enumerate(score):
            if predict_value > threshold:
                index_list.append(index)
                value_list.append(predict_value)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            value_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_values.append(value_list)
    return predicted_labels, predicted_values


def get_label_using_scores_by_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        value_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            value_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_values.append(value_list)
    return predicted_labels, predicted_values


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def process_data_for_predict(file_name, pad_sequence_len):
    words_id = json.load(open("./json/words_id.json", 'r', encoding='utf-8'))
    word_to_id = words_id["word_to_id"]
    contents = [line.strip('\n').split(' ') for line in open(file_name, 'r', encoding='utf-8').readlines()]
    # print(word_to_id)
    data_ids = []
    for i in range(len(contents)):
        data_ids.append([word_to_id[x] if x in word_to_id else word_to_id["<PAD>"] for x in contents[i]])

    # print(data_ids[10])
    return_id = []
    for data_id in data_ids:
        temp = [0] * pad_sequence_len
        if len(data_id) < pad_sequence_len:
            temp[:len(data_id)] = data_id
        else:
            temp = data_id[:pad_sequence_len]
        return_id.append(temp)
    # print(len(return_id[10]))
    # print(return_id[10])
    return np.array(return_id)

def loadWord2Vec(filename):
    vocab = []
    embd = []
    fr = open(filename, 'r', encoding='utf8')
    line = fr.readline().strip()
    word_dim = int(line.split(' ')[1])
    print('word_num: ', int(line.split(' ')[0]))
    print('word_dim: ', word_dim)
    vocab.append("unk")
    embd.append([0] * word_dim)
    for line in fr:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print("loaded word2vec")
    fr.close()
    return vocab, embd


if __name__ == "__main__":
    # 处理标注数据，生成训练、验证及测试集
    # pre_process_node_data()
    # # 生成分类及其对应id
    # make_category_id()
    # # 构建词汇表
    # build_vocab("./data/train_node_set.txt", 2713 + 1)
    # tongji_node_data('I:/yyy/node2vec-master/graph/train_data1.json',
    #                  'I:/yyy/node2vec-master/graph/test_data1.json')
    pass
"""
一共处理2818条数据！！！
classes: 123
words: 2713
"""