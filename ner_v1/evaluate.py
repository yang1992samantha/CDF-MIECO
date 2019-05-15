from ner_v1.model.data_utils import CoNLLDataset
from ner_v1.model.ner_model import NERModel
from ner_v1.model.config import Config
import os
import re

# create instance of config
config = Config()
import jieba
# build model
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def predict_into_ent(model, segfilepath, emrfilepath, entfilepath):
    # file = open(segfilepath,'r',encoding='utf-8')
    # emrfile = open('./test.txt',"w",encoding='utf-8')
    # entfile = open('./test.ent','w',encoding='utf-8')

    segfile_list = os.listdir(segfilepath)
    print(segfile_list)
    for segfilename in segfile_list:
        if (segfilename.__contains__('.seg')):
            position = 0
            emrfilename = segfilename[0:-4] + '.xml'
            entfilename = segfilename[0:-4] + '.ent'
            emrfile_path = os.path.join(emrfilepath, emrfilename)
            entfile_path = os.path.join(entfilepath, entfilename)
            segfile_path = os.path.join(segfilepath, segfilename)
            print(entfile_path)
            file = open(segfile_path, 'r', encoding='utf-8')
            emrfile = open(emrfile_path, 'w', encoding='utf-8')
            entfile = open(entfile_path, 'w', encoding='utf-8')
            sentence_words = []
            for line in file.readlines():
                line = line.strip()
                if (line == '#'):
                    emrfile.write('\n')
                else:
                    emrfile.write(line)

                sentence_words.append(line)
                if (line == '。'):
                    predict_labels = model.predict(sentence_words)
                    if (len(predict_labels) == len(sentence_words)):
                        for i in range(len(predict_labels)):
                            word = sentence_words[i]
                            label = predict_labels[i]
                            if ('B-' in label):
                                T = label[2:]
                                _C = sentence_words[i]

                                begin_position = position
                                endposition = position + len(word)

                                n = i
                                while ('I-' in predict_labels[n + 1] and n + 1 < len(predict_labels)):
                                    endposition = endposition + len(sentence_words[n + 1])
                                    _C = _C + sentence_words[n + 1]
                                    n = n + 1

                                ent_line = "C=" + _C + " P=" + str(begin_position) + ":" + str(endposition) + " T=" + T
                                # print(ent_line)
                                entfile.writelines(ent_line + '\n')

                            position = position + len(word)
                    else:
                        print('error')
                    sentence_words = []

                else:
                    pass
            file.close()
            entfile.close()
            emrfile.close()


def generate_ceshifile(model, validfilename, ceshifilename):
    file = open(validfilename, 'r', encoding='utf-8')
    file2 = open(ceshifilename, 'w', encoding='utf-8')
    setence_words = []
    setence_labels = []
    for line in file.readlines():
        line = line.strip()
        items = line.split()
        if (len(items) > 1):
            setence_words.append(items[0])
            setence_labels.append(items[1])
        else:
            predicts_lables = model.predict(setence_words)
            for word, r_lable, p_lable in zip(setence_words, setence_labels, predicts_lables):
                file2.writelines(word + ' ' + r_lable + ' ' + p_lable + '\n')

            print(setence_labels)
            print(predicts_lables)
            print('-------------------')
            setence_words = []
            setence_labels = []


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)

    # evaluate and interact
    # model.evaluate(test)
    # interactive_shell(model)

    # predict_into_ent(model,'record_huxineike_segment','emr','ent')

    generate_ceshifile(model, './data/valid2.txt', './data/ceshi2.txt')

key=['未','无','否','不']
stop_words = [',', '.', '。', '，', ':', '；', '：' , ';','\n',' ']


def extract_negative(ent_filedata, text):
    new_filedata = []
    for line in ent_filedata:
        rfsplit = re.split('C=|P=|T=|A=', line)
        indexs = rfsplit[2].strip().split(':')

        start_index = int(indexs[0])
        end_index = int(indexs[1])
        temp_index = start_index
        flag_1 = -1
        flag_2 = -1
        while (temp_index >= 0):
            temp_index = temp_index - 1
            if text[temp_index] in stop_words:
                flag_1 = 0
                break
            else:
                if text[temp_index] in key:
                    flag_1 = 1
                    break
        temp_index = end_index - 1
        while (temp_index < len(text) - 1):
            temp_index = temp_index + 1
            if text[temp_index] in stop_words:
                flag_2 = 0
                break
            else:
                if text[temp_index] in key:
                    flag_2 = 1
                    break
        p = 0
        if (flag_1 == 1 or flag_2 == 1):
            p = 1
        else:
            new_filedata.append(line.strip() + ' ' + 'E=present')
            p += 1
    return new_filedata

def dpredict(sentence):
    stop_signs = ['\n', '\r', '\t', '#']
    for i in stop_signs:
        sentence.replace(i, '')
    sentence_words = list(jieba.cut(sentence))
    predict_labels = model.predict(sentence_words)
    position = 0
    disease_list = set([])
    treatment_list = set([])
    complaintsymptom_list = set([])
    test_list = set([])
    testresult_list = set([])
    ner_result = []
    if (len(predict_labels) == len(sentence_words)):
        for i in range(len(predict_labels)):
            word = sentence_words[i]
            label = predict_labels[i]
            if ('B-' in label):
                T = label[2:]
                _C = sentence_words[i].strip('\n')
                begin_position = position
                endposition = position + len(word)
                n = i
                while ('I-' in predict_labels[n + 1] and n + 1 < len(predict_labels)):
                    endposition = endposition + len(sentence_words[n + 1])
                    _C = _C + sentence_words[n + 1]
                    n = n + 1
                if T == "test":
                    if ((sentence_words[n + 1] == '：' or sentence_words[n + 1] == ':' or sentence_words[n + 1] == '示' or sentence_words[ n + 1] == '、') and _C not in test_list):
                        ent_line = "C=" + _C + " P=" + str(begin_position) + ":" + str(endposition) + " T=" + T
                # print(ent_line)
                        ner_result.append(ent_line)
                else:
                    ent_line = "C=" + _C + " P=" + str(begin_position) + ":" + str(endposition) + " T=" + T
                    # print(ent_line)
                    ner_result.append(ent_line)
                # entfile.writelines(ent_line + '\n')
            position = position + len(word)
    del_no_ner_result = extract_negative(ner_result, sentence)
    for i in del_no_ner_result:
        T = i.split(' ')[2][2:]
        _C = i.split(' ')[0][2:].replace('\r', '')
        if T == "treatment":
            treatment_list.add(_C)
        elif T == "complaintsymptom":
            complaintsymptom_list.add(_C)
        elif T == "test":
            test_list.add(_C)
        elif T == "testresult":
            testresult_list.add(_C)
        elif T == "disease":
            disease_list.add(_C)
    nerresult = {
        'disease_list': list(disease_list),
        'treatment_list': list(treatment_list),
        'complaintsymptom_list': list(complaintsymptom_list),
        'test_list': list(test_list),
        'testresult_list': list(testresult_list),
    }
    return nerresult