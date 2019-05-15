import datetime
import os
import linecache
import jieba
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ner_v1.evaluate import dpredict
from Tools.examinationdata import test_name_align
from Tools.yiwangyichangjianchajieguo import DecisionTreeClassifier, judge2, SVM
from py4j.java_gateway import JavaGateway

gateway = JavaGateway()
model_action = gateway.entry_point


def data_preproces(filepath):
    """
    收集文本电子病历信息，存储a1 a2 a3 a4 a5 1/0 形式
    :param filepath:
    :return:
    """
    test_flie_dir = 'G:/xyyy/未改动'
    with open('ICD_dic.txt', 'r', encoding='utf8') as f:
        ICD_data = json.loads(f.read())
    result_file = open(filepath, 'w', encoding='utf8')
    for filename in os.listdir(test_flie_dir):
        pathname = os.path.join(test_flie_dir, filename)
        file = open(pathname, 'r', encoding='utf8')
        recordcontent = file.read()
        file.close()
        pID = filename[:-4]
        doctor_zhenduan_ICD_list = ICD_data[pID]
        if doctor_zhenduan_ICD_list == []:
            continue
        result_list = dpredict(recordcontent)
        after_align_result, stdtest_dic = test_name_align(result_list['complaintsymptom_list'],
                                                          result_list['test_list'], './../resource/1019.txt')
        txt2arff()
        alignedtest2xml(after_align_result)
        split_test_result = split_test(result_list['test_list'])
        stdtest_list = []
        for splittest in split_test_result:
            try:
                stdtest_list.append(stdtest_dic[splittest])
            except:
                pass
        result_list['std_test_list'] = stdtest_list
        mlKNNresult = model_action.prediction("./../resource/1019.arff", "./../resource/label.xml",
                                              "./../resource/MLKNNmodel.dat")
        mlKNNresulttemp = mlKNNresult.split(';')
        testsname, conf_ = rank2name(after_align_result, eval(mlKNNresulttemp[0]), eval(mlKNNresulttemp[1]))
        abnormal_examination_rate_2, _ = judge2(doctor_zhenduan_ICD_list, testsname,
                                                result_list['complaintsymptom_list'])
        for i in result_list['std_test_list']:
            index = testsname.index(i)
            a = [conf_[index][0]] + abnormal_examination_rate_2[index] + [1]
            a = list(map(str, a))
            result_file.write(','.join(a) + '\n')
        suiji_list = [int(random.random() * 232) for i in range(len(result_list['std_test_list']))]
        for index in suiji_list:
            a = [conf_[index][0]] + abnormal_examination_rate_2[index] + [0]
            a = list(map(str, a))
            # a = 置信度 使用率 异常率 价格 效果 分类1/0
            result_file.write(','.join(a) + '\n')
        # break
    result_file.close()


def split_test(filedata):
    """
    拆分组合检查
    :param filedata:
    :return:
    """
    s = "检查"
    l = []
    first_test = ''
    last_test = ''
    seq_list = []
    s1 = ()
    totaltest = []
    t = []
    for line in filedata:
        if line.find(s) == (len(line) - 2) or (line.find("部") != -1):
            # print(line.find(s),",",line,",",len(line))
            line = line.strip(s)
            line = line.replace('部', '')
            t.append(line)
        else:
            t.append(line)
    for line1 in t:
        # print("line1:",line1)
        if ("及" in line1):
            index = line1.find("及")
            seq_list = list(jieba.cut(line1[(index + 1):]))
            if len(seq_list) == 1:
                first_test = line1[:index]
            else:
                if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or ("副" in seq_list[0]) or (
                        "骶" in seq_list[0]):
                    first_test = line1[:index] + "".join(seq_list[2:])
                else:
                    first_test = line1[:index] + "".join(seq_list[1:])
            last_test = "".join(seq_list)
            l.append(first_test)
            l.append(last_test)
        elif ("+" in line1):
            line1 = "".join(line1)
            spl = line1.split("+")
            # print('spl:',spl)
            last_spl = len(spl) - 1
            seq_list = list(jieba.cut(spl[last_spl]))
            if len(seq_list) == 1:
                for i in range(last_spl):
                    # print("seq_list1:","".join(spl[i]))
                    l.append("".join(spl[i]))
            elif len(seq_list) == 2:
                for i in range(last_spl):
                    str2 = "".join(spl[i]) + "".join(seq_list[1])
                    # print("seq_list2:", str2)
                    l.append(str2)
            else:
                str3 = ""
                for i in range(last_spl):
                    if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or ("副" in seq_list[0]) or (
                            "骶" in seq_list[0]):
                        str3 = "".join(spl[i]) + "".join(seq_list[2:])
                    else:
                        str3 = "".join(spl[i]) + "".join(seq_list[1:])
                    # print("seq_list3:", str3)
                    l.append(str3)
            last_test = "".join(seq_list)
            l.append(last_test)
            # print("+l:",l)
        elif ("和" in line1):
            line1 = "".join(line1)
            index = line1.find("和")
            seq_list = list(jieba.cut(line1[(index + 1):]))
            if len(seq_list) == 1:
                first_test = line1[:index]
            else:
                if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or ("副" in seq_list[0]) or (
                        "骶" in seq_list[0]):
                    first_test = line1[:index] + "".join(seq_list[2:])
                else:
                    first_test = line1[:index] + "".join(seq_list[1:])
            last_test = "".join(seq_list)
            l.append(first_test)
            l.append(last_test)
        elif ("加" in line1):
            line1 = "".join(line1)
            index = line1.find("加")
            seq_list = list(jieba.cut(line1[(index + 1):]))
            if len(seq_list) == 1:
                first_test = line1[:index]
            else:
                if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or ("副" in seq_list[0]) or (
                        "骶" in seq_list[0]):
                    first_test = line1[:index] + "".join(seq_list[2:])
                else:
                    first_test = line1[:index] + "".join(seq_list[1:])
            last_test = "".join(seq_list)
            l.append(first_test)
            l.append(last_test)
        else:
            totaltest.append(line1)
    totaltestset = set(totaltest + l)
    return list(totaltestset)


def txt2arff():
    """
    1019.txt 生成 arff 文件
    :param txt_filename:
    :param arff_filename:
    :return:
    """
    data = []
    names = []
    txt_filename = './../resource/1019.txt'
    arff_filename = './../resource/1019.arff'
    relationname = 'test'
    firstLine = True
    file1 = open(txt_filename, 'r', encoding='utf-8')
    file2 = open(arff_filename, 'w+', encoding='utf-8')
    for line in file1.readlines():
        if not firstLine:
            try:
                line = line.replace("\n", "")
                words = line.split(",")
                data.append(words)
            except ValueError:
                print("cant parse file!!")
        else:
            firstLine = False
            line = line.replace("\n", "")
            words = line.split(",")
            names = words
    relationname += "\n"

    relationString = '@RELATION ' + relationname
    file2.write('' + relationString + '')

    for i in range(len(names)):
        str2 = "@ATTRIBUTE " + names[i] + " " + '{0,1}' + "\n"
        file2.write('' + str2 + '')
    file2.write('''@DATA\n''')

    for line in data:
        try:
            file2.write(",".join(line) + "\n")
        except UnicodeEncodeError:
            print("cant write Data to file!!")

    file1.close()
    file2.close()


def alignedtest2xml(initial_label):
    out_labelfilename = './../resource/label.xml'
    writeFile = open(out_labelfilename, 'w+', encoding='utf-8')
    writeFile.write('' + '<?xml version=\"1.0\" encoding=\"utf-8\"?>' + '' + '\n')
    writeFile.write('' + '<labels xmlns=\"http://mulan.sourceforge.net/labels\">' + '' + '\n')
    for line in initial_label:
        str = "<label name=\"" + line + "\"></label> " + "\n"
        writeFile.write('' + str + '')
    writeFile.write('' + '</labels>' + '')
    writeFile.close()


def get_std_test_name(resource_name):
    test__list = linecache.getline(resource_name, 1).strip('\n').split(',')[2171:]
    line_ = linecache.getline(resource_name, 2).strip('\n')
    flag_list = line_.split(',')
    test_name_list = []
    for j in range(2171, len(flag_list)):
        if flag_list[j] == '1':
            test_name_list.append(test__list[j - 2171])
    return test_name_list


def rank2name(label_list, items, conf_items):
    """
    :param label_list:
    :param items:
    :param conf_items:
    :return: checks (test)
    """
    dict = {}
    checks = []
    conf_ = []
    for i in range(len(items)):
        dict[items[i]] = i
    for j in range(len(items)):
        num = j + 1
        index = dict[num]
        check = label_list[index]
        conf_.append([conf_items[index], index])
        checks.append(check)
    return checks, conf_


def process_recordFile(filepath):
    test_flie_dir = 'G:/xyyy/未改动'
    with open('./../Resource/ICD_dic.txt', 'r', encoding='utf8') as f:
        ICD_data = json.loads(f.read())
    c = 0
    result_file = open(filepath, 'w', encoding='utf8')
    all_data = []
    for filename in os.listdir(test_flie_dir):
        result_data = {}
        pathname = os.path.join(test_flie_dir, filename)
        file = open(pathname, 'r', encoding='utf8')
        recordcontent = file.read()
        file.close()
        pID = filename[:-4]
        doctor_zhenduan_ICD_list = ICD_data[pID]
        if doctor_zhenduan_ICD_list == []:
            continue
        c += 1
        now_time = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now_time, '%Y-%m-%d-%H-%M-%S-')
        result_data['file_info'] = now_time + filename
        #  完整NER  disease_list, treatment_list, complaintsymptom_list, test_list, testresult_list
        result_list = dpredict(recordcontent)
        #  test  处理过程   对齐
        after_align_result, stdtest_dic = test_name_align(result_list['complaintsymptom_list'],
                                                          result_list['test_list'], './../Resource/1019.txt')
        txt2arff()
        alignedtest2xml(after_align_result)
        #  拆分组合检查  去除 “部” 等修饰词
        split_test_result = split_test(result_list['test_list'])
        stdtest_list = set([])
        for splittest in split_test_result:
            try:
                stdtest_list.add(stdtest_dic[splittest])
            except:
                pass
        result_list['std_test_list'] = list(stdtest_list)

        #  推断检查  （路径为相对jar包）+ 置信度
        mlKNNresult = model_action.prediction("./1019.arff", "./label.xml", "./MLKNNmodel.dat")
        mlKNNresulttemp = mlKNNresult.split(';')
        testsname, conf_ = rank2name(after_align_result, eval(mlKNNresulttemp[0]), eval(mlKNNresulttemp[1]))
        # 获取特征值 及 排名
        abnormal_examination_rate_2, return_detail_list_2 = judge2(doctor_zhenduan_ICD_list, testsname,
                                                                   result_list['complaintsymptom_list'])
        n_test_num = 0
        all_std_test_num = len(result_list['std_test_list'])
        print(result_list['std_test_list'])
        std_test_num = 0
        testsname_ = testsname.copy()
        random.shuffle(testsname_)
        for i in testsname_:
            index = testsname.index(i)
            # knn 使用率 异常率 价格 使用*异常率 target
            a2 = list(map(str, [conf_[index][0]] + abnormal_examination_rate_2[index]))
            if i not in result_list['std_test_list'] and n_test_num < all_std_test_num:
                # print(i, index, 0, a2)
                all_data.append(' '.join(a2 + ['0']))
                n_test_num += 1
            elif i in result_list['std_test_list'] and std_test_num < all_std_test_num:
                # print(i, index, 1, a2)
                all_data.append(' '.join(a2 + ['1']))
                std_test_num += 1
            if n_test_num == all_std_test_num and std_test_num == all_std_test_num:
                break
        # break

    random.shuffle(all_data)
    result_file.write('\n'.join(all_data))
    result_file.close()


def k_flod_data():
    f = open('./../Data/step2_test.txt', 'r', encoding='utf8')
    samples = []  # len  47852
    target = []
    for i in f.readlines():
        line = i.strip().split(' ')
        l = list(map(float, line))
        # knn 使用率 异常率 价格 使用*异常率 target
        samples.append([l[0], l[1], l[2], l[3]])  # ([l[0], l[1], l[4], l[3]])
        target.append(line[-1])

    x = np.array(samples)
    y = np.array(target)
    skf = StratifiedKFold(n_splits=5)
    i = 0
    for train_index, test_index in skf.split(x, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        data = {
            'x_train': x[train_index],
            'x_test': x[test_index],
            'y_train': y[train_index],
            'y_test': y[test_index],
        }
        np.save('./../Data/KfoldData2/step2data' + str(i), data, allow_pickle=True)
        i += 1


if __name__ == '__main__':
    # process_recordFile('./../Data/step1_test.txt')
    k_flod_data()
    """
    a2[0] > min_knn and result == {} and zhixindu[0] >= min_zhixindu and a2[1] > min_use

    1331 730 745 0.6458030082484231 0.6411368015414258 300
    2076 2061
    1331 730 745 0.6458030082484231 0.6411368015414258 300
    2076 2061
    1331 730 745 0.6458030082484231 0.6411368015414258 300
    2076 2061
    1331 659 745 0.6688442211055277 0.6411368015414258 300
    2076 1990


    12337 6026 6636 0.6718401132712519 0.6502398144731988 2822   0.6608635097493037
    18973 18363

    11216 4215 7757 0.7268485516168751 0.5911558530543404 2810
    f: 0.6520172073014766
    18973 15431
    11216 18973 0.5911558530543404 0.4088441469456596 2810
    0.6038314704483497
    """
