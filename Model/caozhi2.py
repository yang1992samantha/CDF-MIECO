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
import pickle
import numpy as np


# K近邻（K Nearest Neighbor）
def KNN():
    with open('./../Data/Result/Model/KNN.pickle', 'rb') as fr:
        clf = pickle.load(fr)
    return clf


# 线性鉴别分析（Linear Discriminant Analysis）
def LDA():
    with open('./../Data/Result/Model/LDA.pickle', 'rb') as fr:
        clf = pickle.load(fr)
    return clf


# 支持向量机（Support Vector Machine）
def SVM():
    with open('./../Data/Result/Model/SVM.pickle', 'rb') as fr:
        clf = pickle.load(fr)
    return clf


# 逻辑回归（Logistic Regression）
def LR():
    with open('./../Data/Result/Model/LR.pickle', 'rb') as fr:
        clf = pickle.load(fr)
    return clf


# 随机森林决策树（Random Forest）
def RF():
    with open('./../Data/Result/Model/RF.pickle', 'rb') as fr:
        clf = pickle.load(fr)
    return clf


# 多项式朴素贝叶斯分类器
def native_bayes_classifier():
    with open('./../Data/Result/Model/MultinomialNB.pickle', 'rb') as fr:
        clf = pickle.load(fr)
    return clf


# 决策树
def decision_tree_classifier():
    with open('./../Data/Result/Model/DecisionTree.pickle', 'rb') as fr:
        clf = pickle.load(fr)
    return clf


# GBDT
def gradient_boosting_classifier():
    with open('./../Data/Result/Model/GBDT.pickle', 'rb') as fr:
        clf = pickle.load(fr)
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
        data = np.load('./../Data/KfoldData2/step2data' + str(i) + '.npy', allow_pickle=True).item()
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

def get_common_disease_info():
    # disease_list = [('J44.901', 811), ('J15.901', 756), ('J44.103', 665), ('J18.901', 593), ('J45.903', 531)]
    disease_list = ['J44.151', 'J15.901', 'J44.901', 'J45.903', 'J18.900']
    # disease_list = {}
    # d_dic = eval(open('./../Resource/ICD_dic.txt', 'r').read())
    # for i in d_dic:
    #     for j in d_dic[i]:
    #         if j not in disease_list:
    #             disease_list[j] = 1
    #         else:
    #             disease_list[j] += 1
    # disease_list = sorted(disease_list.items(), key=lambda x: x[1], reverse=True)[:10]
    print(disease_list)
    result = {}
    shiyonglv = eval(open('./../Resource/probability_dic_record.txt', 'r', encoding='utf8').read())
    for d in disease_list:
        # if d in shiyonglv:
            result[d] = {'shiyonglv': sorted(shiyonglv[d].items(), key=lambda x: x[1], reverse=True)[:10]}

    yichanglv = eval(open('./../Resource/probability_dic.txt', 'r', encoding='utf8').read())
    for d in disease_list:
        print(d)
        if d[:3] in yichanglv:
            for j in result[d]['shiyonglv']:
                try:
                    print(j, yichanglv[d[:3]][j[0]])
                except:
                    pass

        # result[d[0]] = {'shiyonglv': sorted(shiyonglv[d[0][:3]].items(), key=lambda x: x[1], reverse=True)[:10]}
    # for k,v in result.items():
    #     print(k, v)
get_common_disease_info()