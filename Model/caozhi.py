from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np


# K近邻（K Nearest Neighbor）
def KNN():
    clf = neighbors.KNeighborsClassifier()
    return clf


# 线性鉴别分析（Linear Discriminant Analysis）
def LDA():
    clf = LinearDiscriminantAnalysis()
    return clf


# 支持向量机（Support Vector Machine）
def SVM():
    clf = svm.SVC(gamma='auto')
    return clf


# 逻辑回归（Logistic Regression）
def LR():
    clf = LogisticRegression()
    return clf


# 随机森林决策树（Random Forest）
def RF():
    clf = RandomForestClassifier(n_estimators=10)
    return clf


# 多项式朴素贝叶斯分类器
def native_bayes_classifier():
    clf = MultinomialNB()  # alpha=0.01
    return clf


# 决策树
def decision_tree_classifier():
    clf = tree.DecisionTreeClassifier()
    return clf


# GBDT
def gradient_boosting_classifier():
    clf = GradientBoostingClassifier(n_estimators=200)
    return clf




def all_clr(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train, y_train)
    y_pre = classifier.predict(x_test)
    # b = classification_report(y_pre, y_test, digits=4)
    # print(b)
    return y_pre


def all_model():
    result_p = []
    result_r = []
    result_f = []
    for i in range(5):
        data = np.load('./../Data/KfoldData/step2data' + str(i) + '.npy', allow_pickle=True).item()
        x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
        # print('KNN')
        # c = KNN()
        # print('LDA')
        # c = LDA()
        # print('SVM')
        # c = SVM()
        # print('LR')
        # c = LR()
        # print('RF')
        # c = RF()
        # print('native_bayes_classifier')
        # c = native_bayes_classifier()
        # print('gradient_boosting_classifier')
        # c = gradient_boosting_classifier()
        print('decision_tree_classifier')
        c = decision_tree_classifier()
        y = all_clr(c, x_train, x_test, y_train, y_test)
        result_f.append(f1_score(y_true=data['y_test'], y_pred=y, average='macro'))
        result_p.append(precision_score(y_true=data['y_test'], y_pred=y, average='macro'))
        result_r.append(recall_score(y_true=data['y_test'], y_pred=y, average='macro'))
    print('P: %.4f %.4f %.4f' % (np.min(result_p), np.max(result_p), np.mean(result_p)))
    print('R: %.4f %.4f %.4f' % (np.min(result_r), np.max(result_r), np.mean(result_r)))
    print('F: %.4f %.4f %.4f' % (np.min(result_f), np.max(result_f), np.mean(result_f)))
# all_model()

def eryixing():
    x_ = {}
    m = 0
    with open('./../Data/step2_test.txt', 'r', encoding='utf8')as file:
        for line in file.readlines():
            m += 1
            data = line.strip().split(' ')
            a = data[:-2]
            b = ' '.join(a)
            if b not in x_:
                x_[b] = set([data[-1]])
            else:
                x_[b].add(data[-1])
    print(len(x_))
    print(m)
    n = 0
    for k, v in x_.items():
        if len(v) > 1:
            n += 1
            print(v, k)
    print(n)
eryixing()
"""
348
37924
137

34338
37924
114
"""