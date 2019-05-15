import numpy as np
def get_class():
    classs = set()
    with open('./data/train_data_set.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            class_ = line.split('\t')[0]
            classs.add(class_)
    print(len(classs))

def get_len_data():
    classs = {}
    with open('./data/train_data_set.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            class_ = len(line.split('\t')[1])
            if class_ not in classs:
                classs[class_] = 1
            else:
                classs[class_] += 1
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('sentence length distribution graph')
    # 设置X轴标签
    plt.xlabel('sentence length')  # 设置X/Y轴标签是在对应的figure后进行操作才对应到该figure
    # 设置Y轴标签
    plt.ylabel('frequency')
    # 画散点图
    ax1.scatter(classs.keys(), classs.values(), c='r', marker='.')  # 可以看出画散点图是在对figure进行操作
    # 设置图标
    # plt.legend('show picture x1 ')
    # 显示所画的图
    plt.show()
    print(len(classs))
# get_len_data()

def F_score():
    # new
    result = {}

    lable_list = []
    p = 0
    with open('./data/predict_lables_set.txt', 'r', encoding='utf8') as sfile:
        for line in eval(sfile.read())['lables']:
            lable_list.append(line.split(','))
    for index, lables in enumerate(result['all_threshold']):
        for lable in lables:
            if lable in lable_list[index]:
                p += 1
                break
    print('P_all_threshold:', p / len(lable_list), p, len(lable_list))
    p = 0
    for index, lables in enumerate(result['all_topK']):
        for lable in lables:
            if lable in lable_list[index]:
                p += 1
                break
    print('P_all_topK:     ', p / len(lable_list), p, len(lable_list))
# F_score()
"""
node
P_all_threshold: 0.5574043261231281 335 601
P_all_topK:      0.5490848585690515 330 601
text+node
P_all_threshold: 0.5740432612312812 345 601
P_all_topK:      0.5673876871880200 341 601
text
P_all_threshold: 0.5657237936772047 340 601
P_all_topK:      0.5557404326123128 334 601
"""

def F_score1():
    result = {
    }
    lable_list = []
    p = 0
    lable_list0 = []
    with open('./data/test_node_set.txt', 'r', encoding='utf8') as sfile:
        for line in sfile.readlines():
            lable_list0.append(line.split('\t')[0].split(','))
    with open('./data/predict_lables_set.txt', 'r', encoding='utf8') as sfile:
        for line in eval(sfile.read())['lables']:
            lable_list.append(line.split(','))
    for index, lables in enumerate(lable_list0):
        if lables == lable_list[index]:
            p += 1
    print(lable_list0[:10])
    print(lable_list[:10])
    # for index, lables in enumerate(result['all_topK']):
    #     for lable in lables:
    #         if lable in lable_list[index]:
    #             # print(lable, lable_list[index])
    #             p += 1
    #             break
    print('P:', p / len(lable_list), p, len(lable_list))
# F_score1()  # P: 0.5074875207986689 305 601
"""
RNN
0.9192943538840137 0.9199644690599234 0.9193250725019773
0.911779882912959 0.9123973994876935 0.9118112312153968
0.9183234306171784 0.9195596261711965 0.9183808016877637
0.8898542267764811 0.8987240106661516 0.8904272151898733
0.9164913564040172 0.9174035939644589 0.9165348101265822

CNN
0.9163675679286003 0.9171088841406937 0.9164029535864979
0.9118923252711781 0.9124348562122615 0.9119198312236287
0.9199278853366416 0.9207027415317407 0.9199630801687764
0.9124413645640328 0.913017054449166 0.9124703401001846
0.922077242204228 0.9224421925669678 0.922093329818086
CNN
P: 0.9124 0.9224 0.9171
R: 0.9119 0.9221 0.9166
F: 0.9124 0.9221 0.9165
RNN 
P: 0.8987 0.9200 0.9136
R: 0.8904 0.9193 0.9113
F: 0.8987 0.9193 0.9111

"""
c = '0.9163675679286003 0.9171088841406937 0.9164029535864979' \
    '+0.9118923252711781 0.9124348562122615 0.9119198312236287' \
    '+0.9199278853366416 0.9207027415317407 0.9199630801687764' \
    '+0.9124413645640328 0.913017054449166 0.9124703401001846' \
    '+0.922077242204228 0.9224421925669678 0.922093329818086'
f, p, r = [], [], []
for i in c.split('+'):
    d = list(map(float, i.split(' ')))
    f.append(d[0])
    p.append(d[1])
    r.append(d[2])

print('P: %.4f %.4f %.4f' % (np.min(p), np.max(p), np.mean(p)))
print('R: %.4f %.4f %.4f' % (np.min(r), np.max(r), np.mean(r)))
print('F: %.4f %.4f %.4f' % (np.min(p), np.max(f), np.mean(f)))

rnn = '0.9192943538840137 0.9199644690599234 0.9193250725019773' \
    '+0.911779882912959 0.9123973994876935 0.9118112312153968' \
    '+0.9183234306171784 0.9195596261711965 0.9183808016877637' \
    '+0.8898542267764811 0.8987240106661516 0.8904272151898733' \
    '+0.9164913564040172 0.9174035939644589 0.9165348101265822'
f, p, r = [], [], []
for i in rnn.split('+'):
    d = list(map(float, i.split(' ')))
    f.append(d[0])
    p.append(d[1])
    r.append(d[2])

print('P: %.4f %.4f %.4f' % (np.min(p), np.max(p), np.mean(p)))
print('R: %.4f %.4f %.4f' % (np.min(r), np.max(r), np.mean(r)))
print('F: %.4f %.4f %.4f' % (np.min(p), np.max(f), np.mean(f)))