import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def dengkuan():
    f = open('.\\PlayWeb\\finalfile.txt', 'r', encoding='utf8')
    f2 = open('.\\PlayWeb\\dengkuan.txt', 'w', encoding='utf8')
    samples = []

    for i in f.readlines():
        line = i.strip().split(',')
        samples.append(list(map(float, line)))
    samples = np.array(samples)
    k = 4
    a = pd.cut(samples[:, 0], k, labels=range(k))
    b = pd.cut(samples[:, 1], k, labels=range(k))
    c = pd.cut(samples[:, 2], k, labels=range(k))
    d = pd.cut(samples[:, 3], k, labels=range(k))
    for i in range(len(a)):
        f2.write(
            str(a[i]) + ', ' + str(b[i]) + ', ' + str(c[i]) + ', ' + str(d[i]) + ', ' + str(int(samples[i][4])) + '\n')


def dengpin():
    f = open('.\\PlayWeb\\finalfile.txt', 'r', encoding='utf8')
    f2 = open('.\\PlayWeb\\dengpin.txt', 'w', encoding='utf8')
    samples = []

    for i in f.readlines():
        line = i.strip().split(',')
        samples.append(list(map(float, line)))
    samples = np.array(samples)
    k = 4

    def makedata(data0):
        # print(data0, type(list(data0)))
        data1 = sorted(list(data0))
        data2 = [data1[8294], data1[8294 * 2], data1[8294 * 3], data1[8294 * 4]]
        print(data2)
        data3 = []
        for i in list(data0):
            if i <= data2[0]:
                data3.append(0)
            elif i >= data2[0] and i <= data2[1]:
                data3.append(1)
            elif i >= data2[1] and i <= data2[2]:
                data3.append(2)
            elif i >= data2[2] and i <= data2[3]:
                data3.append(3)
            else:
                data3.append(4)
        return data3

    a = makedata(samples[:, 0])
    # print(a[:10])
    b = makedata(samples[:, 1])
    c = makedata(samples[:, 2])
    d = makedata(samples[:, 3])
    for i in range(len(a)):
        f2.write(
            str(a[i]) + ', ' + str(b[i]) + ', ' + str(c[i]) + ', ' + str(d[i]) + ', ' + str(int(samples[i][4])) + '\n')
    f2.close()
    """
    [0.0009884470613739248, 0.019082063319545792, 0.23340770990235418, 0.9230381990479329]
    [0.0, 0.0, 0.023501762632197415, 0.38235294117647056]
    [0.0, 0.020361990950226245, 0.3737704918032787, 0.8823529411764706]
    [0.003207481271873238, 0.012117151471521124, 0.02851094463887324, 0.08553283391661974]
    """


def k_means():
    f = open('./../Data/step1_test.txt', 'r', encoding='utf8')  # ????????????????????????
    f2 = open('./../Data/step2_test.txt', 'w', encoding='utf8')  # ????????????????????????
    samples = []

    for i in f.readlines():
        line = i.strip().split(' ')
        samples.append(list(map(float, line)))
    samples = np.array(samples)

    def makedata(data0, ai):
        data0 = data0.reshape(-1, 1)
        # kmeans = KMeans(n_clusters=5, max_iter=1000, init='k-means++', n_init=20, algorithm='auto')
        kmeans = KMeans(init='k-means++', algorithm='auto')
        data0 = kmeans.fit_predict(data0)
        # 保存成python支持的文件格式pickle, 在当前目录下可以看到svm.pickle
        with open('./../Data/KMeans/KMeans'+str(ai)+'.pickle', 'wb') as fw:
            pickle.dump(kmeans, fw)
        lables = []
        for i in kmeans.cluster_centers_:
            lables.append(i[0])
        lables1 = sorted(lables)
        data1 = []
        for i in data0:
            data1.append(lables1.index(lables[i]))
        return data1

    # knn 使用率 异常率 价格 使用*异常率 target
    a = makedata(samples[:, 0], 1)
    b = makedata(samples[:, 1], 2)
    c = makedata(samples[:, 2], 3)
    d = makedata(samples[:, 3], 4)
    e = makedata(samples[:, 4], 5)
    # print(d[:20])
    for i in range(len(a)):
        f2.write(str(a[i]) + ' ' + str(b[i]) + ' ' + str(c[i]) + ' ' + str(d[i]) + ' ' + str(e[i]) + ' ' +
                 str(int(samples[i][5])) + '\n')
    f2.close()
# k_means()

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


f = open('./../Data/step1_test.txt', 'r', encoding='utf8')  # ????????????????????????
samples = []

for i in f.readlines():
    line = i.strip().split(' ')
    samples.append(list(map(float, line)))
samples = np.array(samples)
# create new plot and data
plt.plot()
X = samples[:, 1].reshape(-1, 1)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k of feature 2')
plt.show()