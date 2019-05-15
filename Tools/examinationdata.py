# -*- coding: UTF-8 -*-
import jieba

def test_name_align(complaintsymptom_list, nertest_list, out_classificationfilename):
    """
    检查名称对齐， 生成mlknn输入数据
    :param complaintsymptom_list:
    :param nertest_list:
    :param out_classificationfilename:
    :return:
    """
    aligntest_filename = './../Resource/testnamealign.txt'
    symptom_filename = './../Resource/symptomList.txt'
    pretest_filename = './../Resource/finaltest1.txt'
    out = open(out_classificationfilename, 'w', encoding='utf-8')
    pretest_file = open(pretest_filename, 'r', encoding='utf-8')
    test__ = pretest_file.readlines()
    symptom_list = []
    file1 = open(aligntest_filename, 'r', encoding='utf-8')
    test_dict = dict()
    true_test = []
    wrong_test = []
    # 对file1中的内容读取 将数据对齐
    for line in file1.readlines():
        line = line.strip('\n')
        temp = line.split(' ')
        # print('line:',line,',test_list:',test_list)
        if len(temp) == 1:
            test_dict[temp[0]] = temp[0]
        else:
            test_dict[temp[0]] = temp[1]
            if '，' in temp[1]:
                true_test.extend(temp[1].split('，'))
            elif '否' not in temp[1]:
                true_test.append(temp[1])
            wrong_test.append(temp[0])
    # temp_set = set(true_test)
    for i in open(symptom_filename, 'r', encoding='utf-8').readlines():
        i = i.strip('\n')
        i = i.replace(' ', '')
        symptom_list.append(i)
    tests_list = []

    for i in test__:
        tests_list.append(i.strip('\n'))
    for w in wrong_test:
        if w not in tests_list:
            # print('w', w)
            pass
        else:
            tests_list.remove(w)
    for each in true_test:
        if each not in tests_list:
            tests_list.append(each)

    table_tille = symptom_list + tests_list
    # print('length of table_title:',len(table_tille),',length of symptom:',len(symptoms_list))
    out.write(','.join(table_tille) + u'\n')

    test_list = set([])
    test_dic = {}
    for houxuan in nertest_list:
        s = '检查'
        if houxuan.find(s) == (len(houxuan) - 2) or (houxuan.find("部") != -1):
            # print(line.find(s),",",line,",",len(line))
            houxuan = houxuan.strip(s)
            houxuan = houxuan.replace('部', '')
        t = data_align(houxuan)
        # print('t', t, houxuan, 'J45.903，支气管哮喘')
        for it in t:  # 先在test__中匹配 如果在test__中匹配到了 将file1中数据拆分
            if it in tests_list:
                test_list.add(it)
                test_dic[it] = it
            else:
                for key in test_dict:
                    if it == key:
                        if '，' in ''.join(test_dict[key]):
                            for i in range(len(''.join(test_dict[key]).split('，'))):
                                test_list.add(''.join(test_dict[key]).split('，')[i])
                                test_dic[it] = ''.join(test_dict[key]).split('，')[i]
                        elif '否' not in ''.join(test_dict[key]):
                            test_list.add(''.join(test_dict[key]))
                            test_dic[it] = ''.join(test_dict[key])
                    # print('it:', it, ',key:', key,'split[i]',''.join(test_dict[key]).split('，')[i])
                if it in tests_list:
                    test_list.add(it)
                    test_dic[it] = it
                    break

    result = []
    for j in symptom_list:
        flag1 = 1
        for k in complaintsymptom_list:
            if j == k:
                flag1 = 0
                break
        if flag1:
            result.append('0')
        else:
            result.append('1')
    for j in tests_list:
        result.append('0')
    tou = len(symptom_list)
    c = 0

    for n in test_list:
        flag = 0
        pipeidaode_test = n
        if n in tests_list:
            flag = 1
            # print(n, n)
        else:
            for test_ in tests_list:
                if test_ in n:
                    # print(test_, n)
                    pipeidaode_test = test_
                    flag = 1
                    break
        if flag:
            result[tou + tests_list.index(pipeidaode_test)] = '1'
            c += 1
    out.write(','.join(result) + u'\n')
    out.close()
    file1.close()
    pretest_file.close()
    # print('test_dic', test_dic)
    # palytestlist = []
    # for i in test_list:
    #     palytestlist.append()
    return tests_list, test_dic

def data_align(houxuan):
    t = []
    if ("及" in houxuan):
        index = houxuan.find("及")
        # l.append(line[:index])
        # l.append(line[index+1:])
        seq_list = list(jieba.cut(houxuan[(index + 1):]))
        if len(seq_list) == 1:
            first_test = houxuan[:index]
        else:
            if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or ("副" in seq_list[0]) or ("骶" in seq_list[0]):
                first_test = houxuan[:index] + "".join(seq_list[2:])
            else:
                first_test = houxuan[:index] + "".join(seq_list[1:])
        last_test = "".join(seq_list)
        t.append(first_test)
        t.append(last_test)
        # print("firt_test:"+first_test+",last_test:"+last_test)
    elif ("+" in houxuan):
        houxuan = "".join(houxuan)
        spl = houxuan.split("+")
        # print('spl:', spl)
        last_spl = len(spl) - 1
        # print("jieba:",spl[last_spl])
        seq_list = list(jieba.cut(spl[last_spl]))
        if len(seq_list) == 1:
            for i in range(last_spl):
                # print("seq_list1:", "".join(spl[i]))
                t.append("".join(spl[i]))
        elif len(seq_list) == 2:
            for i in range(last_spl):
                str2 = "".join(spl[i]) + "".join(seq_list[1])
                # print("seq_list2:", str2)
                t.append(str2)
        elif len(seq_list) > 2:
            str3 = ""
            for i in range(last_spl):
                if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or (
                            "副" in seq_list[0]) or ("骶" in seq_list[0]):
                    str3 = "".join(spl[i]) + "".join(seq_list[2:])
                else:
                    str3 = "".join(spl[i]) + "".join(seq_list[1:])
                # print("seq_list3:", str3)
                t.append(str3)
        last_test = "".join(seq_list)
        t.append(last_test)

    elif ("和" in houxuan):
        houxuan = "".join(houxuan)
        index = houxuan.find("和")
        seq_list = list(jieba.cut(houxuan[(index + 1):]))
        if len(seq_list) == 1:
            first_test = houxuan[:index]
        else:
            if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or (
                        "副" in seq_list[0]) or ("骶" in seq_list[0]):
                first_test = houxuan[:index] + "".join(seq_list[2:])
            else:
                first_test = houxuan[:index] + "".join(seq_list[1:])
        last_test = "".join(seq_list)
        t.append(first_test)
        t.append(last_test)
    elif ("加" in houxuan):
        houxuan = "".join(houxuan)
        index = houxuan.find("加")
        seq_list = list(jieba.cut(houxuan[(index + 1):]))
        # print('houxuan:',houxuan,',seq_list', len(seq_list))
        if len(seq_list) <= 1:
            first_test = houxuan[:index]
        else:
            if ("部" in seq_list[1]) or ("中" in seq_list[0]) or ("全" in seq_list[0]) or (
                        "副" in seq_list[0]) or ("骶" in seq_list[0]):
                first_test = houxuan[:index] + "".join(seq_list[2:])
            else:
                first_test = houxuan[:index] + "".join(seq_list[1:])
        last_test = "".join(seq_list)
        t.append(first_test)
        t.append(last_test)
    else:
        t.append(houxuan)
    return t