import pickle
import numpy as np

def get_att(Diseases, Symptoms):
    file = open('./../Resource/learning_weight_list.txt', 'r', encoding='utf8')
    file_data = {}
    for line in file.readlines():
        line = line.strip('\n').split(' ')
        file_data[line[0]] = line[1]
    att_data = {}
    for disease in Diseases:
        temp_ = 0
        for symptom in Symptoms:
            temp = symptom+'-->'+disease
            if temp in file_data:
                temp_ += float(file_data[temp])
        att_data[disease] = temp_
    sum_ = sum(att_data.values())
    if sum_:
        for i in att_data:
            att_data[i] = att_data[i] / sum_
    return att_data


def getrank(temp, dic):
    c = 0
    for i in dic.values():
        if i >= temp:
            c += 1
    return c

def judge(diagnosis, judge_list):
    return_list = []
    return_detail_list = []
    file1 = open('./../Resource/probability_dic.txt', 'r', encoding='utf-8')
    probability_dic = eval(file1.read())
    file2 = open('./../Resource/probability_dic_record.txt', 'r', encoding='utf-8')
    probability_dic_record = eval(file2.read())
    file3 = open('./../Resource/check_price.txt', 'r', encoding='utf-8')
    test_price = eval(file3.read())

    if (diagnosis[:3] in probability_dic.keys() and diagnosis in probability_dic_record.keys()):
        judge_dic_2 = probability_dic[diagnosis[:3]]  # 异常
        judge_dic_1 = probability_dic_record[diagnosis]  # 使用
        for item in judge_list:
            if item in judge_dic_1.keys():
                probability_1 = judge_dic_1[item]
                r = getrank(judge_dic_1[item], judge_dic_1)
                probability_1_detail = ('疾病共使用过' + str(len(judge_dic_1)) + '种检查，该检查使用次数排名为：'
                                        + str(r) + '  ', 1 - r/len(judge_dic_1))
            else:
                probability_1 = 0
                probability_1_detail = ('疾病未使用过该项检查  ', -1)
            if item in judge_dic_2.keys():
                probability_2 = judge_dic_2[item]
                r = getrank(judge_dic_2[item], judge_dic_2)
                probability_2_detail = ('疾病使用的检查中共有' + str(len(judge_dic_2)) + '种有异常结果，该检查出现异常结果排名为：' + str(
                    getrank(judge_dic_2[item], judge_dic_2)) + '  ', 1 - r/len(judge_dic_2))
            else:
                probability_2 = 0
                probability_2_detail = ('疾病使用该检查未出现过异常检查结果  ', -1)
            if item in test_price.keys():
                probability_3 = test_price[item]
                probability_3_detail = getrank(test_price[item], test_price)
            else:
                probability_3 = -1
                probability_3_detail = 0
            return_item = [probability_1, probability_2, probability_3, probability_1*probability_2]
            return_detail = [probability_1_detail, probability_2_detail, probability_3_detail]
            return_list.append(return_item)
            return_detail_list.append(return_detail)
            # print(return_item)
    file1.close()
    file2.close()
    file3.close()
    rank_list = sorted([x[3] for x in return_list])
    sum_ = len(return_list)
    for i in range(sum_):
        return_detail_list[i].append(rank_list.index(return_list[i][3])/sum_)
    return return_list, return_detail_list

def judge2(diagnosis_list, judge_list, Symptoms):
    return_list = []
    return_detail_list = []
    file1 = open('./../Resource/probability_dic.txt', 'r', encoding='utf-8')
    probability_dic = eval(file1.read())
    file2 = open('./../Resource/probability_dic_record.txt', 'r', encoding='utf-8')
    probability_dic_record = eval(file2.read())
    file3 = open('./../Resource/check_price.txt', 'r', encoding='utf-8')
    test_price = eval(file3.read())
    att_dic = get_att(diagnosis_list, Symptoms)
    for item in judge_list:
        probability_1 = []
        probability_2 = []
        probability_3 = []
        test_num = 0
        yichang_num = 0
        for diagnosis in diagnosis_list:
            if (diagnosis[:3] in probability_dic.keys() and diagnosis in probability_dic_record.keys()):
                judge_dic_2 = probability_dic[diagnosis[:3]]  # 异常
                judge_dic_1 = probability_dic_record[diagnosis]  # 使用
                if item in judge_dic_1.keys():
                    probability_1.append(judge_dic_1[item])
                else:
                    probability_1.append(0)
                if item in judge_dic_2.keys():
                    probability_2.append(judge_dic_2[item])
                else:
                    probability_2.append(0)
                if item in test_price.keys():
                    probability_3.append(test_price[item])
                else:
                    probability_3.append(0)
                test_num += len(probability_dic_record.keys())
                yichang_num += len(probability_dic.keys())
        a2 = 0
        for i in range(len(probability_1)):
            a2 += probability_1[i] * att_dic[diagnosis_list[i]]
        a3 = 0
        for i in range(len(probability_2)):
            a3 += probability_2[i] * att_dic[diagnosis_list[i]]
        a4 = 0
        for i in range(len(probability_3)):
            a4 += probability_3[i] * att_dic[diagnosis_list[i]]
        return_list.append([a2, a3, a4, a2*a3])

        probability_1_detail = ''
        probability_2_detail = ''
        probability_3_detail = ''
        return_detail = [probability_1_detail, probability_2_detail, probability_3_detail]
        return_detail_list.append(return_detail)
    file1.close()
    file2.close()
    file3.close()
    rank_list = sorted([x[3] for x in return_list])
    sum_ = len(return_list)
    for i in range(sum_):
        return_detail_list[i].append(rank_list.index(return_list[i][3])/sum_)
    return return_list, return_detail_list


def get_lineage(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]
    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
        lineage.append((parent, split, threshold[parent], features[parent]))
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    result = {}
    for child in idx:
        result[child] = []
        for node in recurse(left, right, child):
            # print(node)
            result[child].append(node)
    return result
#
def chuli(samples):
    samples = np.array(samples)
    with open('./../data/KMeans1.pickle', 'rb') as fr:
        KMeans1 = pickle.load(fr)
        lables1 = []
        for i in KMeans1.cluster_centers_:
            lables1.append(i[0])
        lables1_ = sorted(lables1)
    with open('./../data/KMeans2.pickle', 'rb') as fr:
        KMeans2 = pickle.load(fr)
        lables2 = []
        for i in KMeans2.cluster_centers_:
            lables2.append(i[0])
        lables2_ = sorted(lables2)
    with open('./../data/KMeans3.pickle', 'rb') as fr:
        KMeans3 = pickle.load(fr)
        lables3 = []
        for i in KMeans3.cluster_centers_:
            lables3.append(i[0])
        lables3_ = sorted(lables3)
    with open('./../data/KMeans4.pickle', 'rb') as fr:
        KMeans4 = pickle.load(fr)
        lables4 = []
        for i in KMeans4.cluster_centers_:
            lables4.append(i[0])
        lables4_ = sorted(lables4)
    with open('./../data/KMeans5.pickle', 'rb') as fr:
        KMeans5 = pickle.load(fr)
        lables5 = []
        for i in KMeans5.cluster_centers_:
            lables5.append(i[0])
        lables5_ = sorted(lables5)
    # print(samples)
    lable1 = KMeans1.predict(samples[0].reshape(-1, 1))
    lable2 = KMeans2.predict(samples[1].reshape(-1, 1))
    lable3 = KMeans3.predict(samples[2].reshape(-1, 1))
    lable4 = KMeans4.predict(samples[3].reshape(-1, 1))
    lable5 = KMeans5.predict(samples[4].reshape(-1, 1))
    data0 = [lables1_.index(lables1[lable1[0]]), lables2_.index(lables2[lable2[0]]),
             lables3_.index(lables3[lable3[0]]), lables4_.index(lables4[lable4[0]]),
             lables5_.index(lables5[lable5[0]])]
    data0 = np.array(data0)
    # print(data0)
    return data0

def SVM(data0):
    data1 = chuli(data0)[:-1]
    result = {}
    with open('./../../Resource/SVC.pickle', 'rb') as fr:
        zhixindu = []
        classifier = pickle.load(fr)
        x = classifier.predict([data1])
        z = classifier.predict_proba([data1])
        for i in range(len(z)):
            if x[i]:
                zhixindu.append(z[i][1])
            else:
                zhixindu.append(0)
    return result, zhixindu

def DecisionTreeClassifier(data0):
    data1 = chuli(data0)
    result = {}
    with open('./../../Resource/DecisionTreeClassifier.pickle', 'rb') as fr:
        zhixindu = []
        classifier = pickle.load(fr)
        x = classifier.predict([data1])
        z = classifier.predict_proba([data1])
        for i in range(len(z)):
            if x[i]:
                zhixindu.append(z[i][1])
            else:
                zhixindu.append(0)
        # lujings = classifier.decision_path([data1])
        # tree_ = get_lineage(classifier, ['1', '2', '3', '4'])
        # yezi = lujings[0].indices[-1]
        # if yezi in tree_ and x[0] == 0:
        #     for j in range(len(tree_[yezi]) - 1):
        #         if tree_[yezi][j][3] == '4' and tree_[yezi][j][1] == 'r':
        #             if tree_[yezi][j][3] not in result:
        #                 result[tree_[yezi][j][3]] = tree_[yezi][j][2]
        #             elif result[tree_[yezi][j][3]] < tree_[yezi][j][2]:
        #                 result[tree_[yezi][j][3]] = tree_[yezi][j][2]
        #         elif tree_[yezi][j][1] == 'l':
        #             if tree_[yezi][j][3] not in result:
        #                 result[tree_[yezi][j][3]] = tree_[yezi][j][2]
        #             elif result[tree_[yezi][j][3]] > tree_[yezi][j][2]:
        #                 result[tree_[yezi][j][3]] = tree_[yezi][j][2]
        #         # print(tree_[yezi][j][1], tree_[yezi][j][2], tree_[yezi][j][3])
        #     # print(result)
    return result, zhixindu

def DecisionTreeClassifier0(data0):
    fuzhu_lisanhua_list = [[0.0009884470613739248, 0.019082063319545792, 0.23340770990235418, 0.9230381990479329],
     [0.0, 0.0, 0.023501762632197415, 0.38235294117647056],
     [0.0, 0.020361990950226245, 0.3737704918032787, 0.8823529411764706],
     [0.003207481271873238, 0.012117151471521124, 0.02851094463887324, 0.08553283391661974]]
    data1 = []  # 转化后的数组
    for i in range(4):
        flag = 0
        for j in fuzhu_lisanhua_list[i]:
            if data0[i] <= j:
                data1.append(fuzhu_lisanhua_list[i].index(j))
                flag = 1
                break
        if flag == 0:
            data1.append(4)
    # print(data0, data1)
    result = {}
    with open('./../data/DecisionTreeClassifier.pickle', 'rb') as fr:
        zhixindu = []
        classifier = pickle.load(fr)
        x = classifier.predict([data1])
        z = classifier.predict_proba([data1])
        for i in range(len(z)):
            zhixindu.append(z[i][x[i]])
        lujings = classifier.decision_path([data1])
        tree_ = get_lineage(classifier, ['1', '2', '3', '4'])
        yezi = lujings[0].indices[-1]
        if yezi in tree_ and x[0] == 0:
            for j in range(len(tree_[yezi]) - 1):
                if tree_[yezi][j][3] == '4' and tree_[yezi][j][1] == 'r':
                    if tree_[yezi][j][3] not in result:
                        result[tree_[yezi][j][3]] = tree_[yezi][j][2]
                    elif result[tree_[yezi][j][3]] < tree_[yezi][j][2]:
                        result[tree_[yezi][j][3]] = tree_[yezi][j][2]
                elif tree_[yezi][j][1] == 'l':
                    if tree_[yezi][j][3] not in result:
                        result[tree_[yezi][j][3]] = tree_[yezi][j][2]
                    elif result[tree_[yezi][j][3]] > tree_[yezi][j][2]:
                        result[tree_[yezi][j][3]] = tree_[yezi][j][2]
                # print(tree_[yezi][j][1], tree_[yezi][j][2], tree_[yezi][j][3])
            # print(result)
    return result, zhixindu
