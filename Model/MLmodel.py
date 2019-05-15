import numpy as np
import pickle
from time import time
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier


# report函数，将调参的详细结果存储到本地F盘（路径可自行修改，其中n_top是指定输出前多少个最优参数组合以及该组合的模型得分）
def report(clf, clf_name, results, n_top=20):
    rank = 0
    with open('./../Data/Result0/Grid_search/' + clf_name + '.txt', 'w') as f:
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                rank += 1
                f.write("{0} Model with rank: {1}".format(rank, i) + '\n')
                f.write("Mean validation score: {0:.4f} (std: {1:.4f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]) + '\n')
                f.write("Parameters: {0}".format(results['params'][candidate]) + '\n')
                f.write("\n")
    with open('./../Data/Result0/Model/' + clf_name + '.pickle', 'wb') as fr:
        pickle.dump(clf, fr)


# K近邻（K Nearest Neighbor）
def KNN(x_train, y_train):
    clf = neighbors.KNeighborsClassifier()
    # print(clf.get_params().keys())
    param_grid = {"weights": ['uniform', 'distance'],
                  "n_neighbors": [5, 10, 20, 50],
                  "leaf_size": [20, 30, 50],
                  "algorithm": ['auto'],
                  "p": [1, 2],
                  "n_jobs": [-1],
                  }
    grid_search = GridSearchCV(clf, scoring='f1', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'KNN', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')

    return


# 线性鉴别分析（Linear Discriminant Analysis）
def LDA(x_train, y_train):
    clf = LinearDiscriminantAnalysis()
    param_grid = {"solver": ['lsqr', 'eigen'],
                  "shrinkage": [None, 'auto'],
                  }
    grid_search = GridSearchCV(clf, scoring='f1', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'LDA', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')

    return


# 支持向量机（Support Vector Machine）
def SVM(x_train, y_train):
    clf = svm.SVC()
    # print(clf.get_params().keys())
    param_grid = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                  "gamma": ['auto'],
                  "C": [0.75, 1.0, 1.25],
                  "degree": [1, 2, 3, 4],
                  "coef0": [0.0, 0.1],
                  }
    grid_search = GridSearchCV(clf, scoring='f1', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'SVM', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')

    return


# 逻辑回归（Logistic Regression）
def LR(x_train, y_train):
    clf = LogisticRegression()
    # print(clf.get_params().keys())
    param_grid = {"penalty": ['l1', 'l2'],
                  "dual": [False],
                  "C": [0.75, 1.0, 1.25],
                  "fit_intercept": [True, False],
                  'solver': ['liblinear'],
                  }
    grid_search = GridSearchCV(clf, scoring='f1', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'LR', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')
    return


# 多项式朴素贝叶斯分类器
def native_bayes_classifier(x_train, y_train):
    clf = MultinomialNB()
    # print(clf.get_params().keys())
    param_grid = {"alpha": [0.75, 1.0, 1.25],
                  "fit_prior": [True, False],
                  "class_prior": [None],
                  }
    grid_search = GridSearchCV(clf, scoring='f1', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'MultinomialNB', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')
    return


# 决策树
def decision_tree_classifier(x_train, y_train):
    clf = tree.DecisionTreeClassifier()
    # print(clf.get_params().keys())
    param_grid = {"criterion": ['gini', 'entropy'],
                  "max_depth": [5, 10, 20, 30],
                  "min_impurity_decrease": [0., 0.1, 0.15],
                  "min_samples_split": [100, 200, 300],
                  "max_leaf_nodes": [500, 750, 1000],
                  }
    grid_search = GridSearchCV(clf, scoring='f1_macro', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'DecisionTree', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')
    return


# 随机森林决策树（Random Forest）
def RF(x_train, y_train):
    clf = RandomForestClassifier()
    # print(clf.get_params().keys())
    param_grid = {"criterion": ['gini', 'entropy'],
                  "n_estimators": [5, 10, 20, 30],
                  "max_features": ['auto', 'log2', None],
                  "max_depth": [5, 10, 20, 30],
                  "min_samples_split": [50, 75, 100],
                  "bootstrap": [True, False],
                  }
    grid_search = GridSearchCV(clf, scoring='f1', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'RF', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')


# GBDT
def gradient_boosting_classifier(x_train, y_train):
    clf = GradientBoostingClassifier(n_estimators=200)
    # print(clf.get_params().keys())
    param_grid = {"n_estimators": [75, 100, 125],
                  "learning_rate": [0.25, 0.5, 0.75, 1],
                  "subsample": [0.5, 0.7, 0.8, 1],
                  "loss": ['deviance', 'exponential'],
                  }
    grid_search = GridSearchCV(clf, scoring='f1_macro', param_grid=param_grid, cv=5, n_jobs=-1)
    start = time()
    grid_search.fit(x_train, y_train)  # 传入训练集矩阵和训练样本类标
    print("GridSearchCV took %.3f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.best_estimator_, 'GBDT', grid_search.cv_results_)
    print('best_score:', grid_search.best_score_, '\n')


def all_model():
    f = open('./../Data/step2_test.txt', 'r', encoding='utf8')
    samples = []  # len  47852
    target = []
    for i in f.readlines():
        line = i.strip().split(' ')
        l = list(map(float, line))
        # knn 使用率 异常率 价格 使用*异常率 target
        samples.append([l[0], l[1], l[2], l[3]])  # ([l[0], l[1], l[4], l[3]])
        target.append(int(line[-1]))
    # x_train, x_test, y_train, y_test = train_test_split(samples, target, test_size=0.3, random_state=1)

    print('KNN')
    c = KNN(samples, target)
    #
    # print('LDA')
    # c = LDA(samples, target)
    #
    # print('SVM')
    # c = SVM(samples, target)
    #
    # print('LR')
    # c = LR(samples, target)
    #
    # print('native_bayes_classifier')
    # c = native_bayes_classifier(samples, target)
    #
    # print('decision_tree_classifier')
    # c = decision_tree_classifier(samples, target)
    #
    # print('RF')
    # c = RF(samples, target)

    # print('gradient_boosting_classifier')
    # c = gradient_boosting_classifier(samples, target)

def dtree_train(samples, target, test):
    classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth=10, max_leaf_nodes=500,
                                             min_impurity_decrease=0.0, min_samples_split=100)
    classifier.fit(samples, target)  # 训练数据来学习，不需要返回值
    y_pre = classifier.predict(test)  # 测试数据，分类返回标记
    return y_pre


def forest():  # 森林 train:30338    test:7586
    result_p = []
    result_r = []
    result_f = []
    for i in range(5):
        data = np.load('./../Data/KfoldData/step2data' + str(i) + '.npy', allow_pickle=True).item()
        y1 = list(map(int, dtree_train(data['x_train'][:, 0:1], data['y_train'], data['x_test'][:, 0:1])))
        y2 = list(map(int, dtree_train(data['x_train'][:, 1:2], data['y_train'], data['x_test'][:, 1:2])))
        y3 = list(map(int, dtree_train(data['x_train'][:, 2:3], data['y_train'], data['x_test'][:, 2:3])))
        y4 = list(map(int, dtree_train(data['x_train'][:, 3:4], data['y_train'], data['x_test'][:, 3:4])))
        yy = []
        for i in range(len(y1)):
            # c = (y1[i] + y2[i] + y3[i] + y4[i]) / 4
            c = (y3[i]) / 1
            if c >= 0.5:
                yy.append('1')
            else:
                yy.append('0')
        # a = classification_report(data['y_test'], yy, digits=4)
        # print(a)
        result_f.append(f1_score(data['y_test'], y_pred=yy, average='macro'))
        result_p.append(precision_score(data['y_test'], y_pred=yy, average='macro'))
        result_r.append(recall_score(data['y_test'], y_pred=yy, average='macro'))
        # print(f1_score(data['y_test'], y_pred=yy, average='macro'))
        # print(precision_score(data['y_test'], y_pred=yy, average='macro'))
        # print(recall_score(data['y_test'], y_pred=yy, average='macro'))
        # break

    print('P: %.4f %.4f %.4f' % (np.min(result_p), np.max(result_p), np.mean(result_p)))
    print('R: %.4f %.4f %.4f' % (np.min(result_r), np.max(result_r), np.mean(result_r)))
    print('F: %.4f %.4f %.4f' % (np.min(result_f), np.max(result_f), np.mean(result_f)))

def tezhengyouxiaoxing():
    result_p = []
    result_r = []
    result_f = []
    for i in range(5):
        data = np.load('./../Data/KfoldData2/step2data' + str(i) + '.npy', allow_pickle=True).item()
        x_train, x_test = [], []
        for index, line in enumerate(data['x_train']):
            x_train.append([line[0], line[1], line[2], line[3]])
        for index, line in enumerate(data['x_test']):
            x_test.append([line[0], line[1], line[2], line[3]])
        y = dtree_train(x_train, data['y_train'], x_test)
        # a = classification_report(data['y_test'], yy, digits=4)
        # print(a)
        result_f.append(f1_score(y_true=data['y_test'], y_pred=y, average='macro'))
        result_p.append(precision_score(y_true=data['y_test'], y_pred=y, average='macro'))
        result_r.append(recall_score(y_true=data['y_test'], y_pred=y, average='macro'))
        # print(f1_score(data['y_test'], y_pred=yy, average='macro'))
        # print(precision_score(data['y_test'], y_pred=yy, average='macro'))
        # print(recall_score(data['y_test'], y_pred=yy, average='macro'))
        # break
    print('P: %.4f %.4f %.4f' % (np.min(result_p), np.max(result_p), np.mean(result_p)))
    print('R: %.4f %.4f %.4f' % (np.min(result_r), np.max(result_r), np.mean(result_r)))
    print('F: %.4f %.4f %.4f' % (np.min(result_f), np.max(result_f), np.mean(result_f)))


if __name__ == '__main__':
    all_model()
    # forest()
    # tezhengyouxiaoxing()
