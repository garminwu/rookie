from torch.autograd import Variable
import torch
import modelElite_cnn
import modelElite
import torch.nn.functional as F
import random
import collections
import numpy as np
import re
import os


class Inst(object):  # 定义Inst类型，将文件中的每句话生成Inst的数据类型
    def __init__(self):
        self.m_word = []  # 将一句话中除label外的每个单词都存入m_word列表
        self.m_label = 0  # 将一句话中的label存入m_label中
        # def show(self):
        #     print(self.m_word)
        #     print(self.m_label)


class vocab(object):  # 生成词典
    def __init__(self):
        self.v_list = []
        self.v_dict = collections.OrderedDict()  # 固定字典

    def MakeVocab(self, ListName):  # 生成词典的过程
        for i in range(len(ListName)):
            if ListName[i] not in self.v_list:
                self.v_list.append(ListName[i])
        self.v_list.append("-unknown-")
        for n in range(len(self.v_list)):
            self.v_dict[self.v_list[n]] = n

        return self.v_dict


class example(object):  #
    def __init__(self):
        self.word_indexes = []
        self.label_index = []


class HyperParameters(object):
    def __init__(self):
        self.lr = 0.001
        self.epochs = 0
        self.batch = 16
        self.hiddenSize = 0
        self.labelSize = 0
        self.embedding_num = 0
        self.embedding_dim = 300
        self.wordEmbeddingSize = 0
        self.unknown = None
        self.kernel_size = [1, 2, 3]
        self.kernel_num = 300
        self.word_Embedding = True
        self.word_Embedding_Path = "./data/converted_word_Subj.txt"
        self.pretrained_weight = None
        self.LSTM_hidden_dim =300
        self.num_layers = 1
        self.class_num = 2
        self.dropout_embed = 0.2
        self.dropout = 0.2
        self.save_dir = "snapshot"
        self.LSTM_model = False

class Reader():
    def readfile(self, path):
        f = open(path, 'r')
        newList = []
        count = 0
        for line in f.readlines():
            count += 1
            new = Inst()
            # x = line.strip().split("||| ")  # big bug big bug big bug stupid bug stupid bug
            label, seq, sentence = line.partition(" ")
            sentence = Reader.clean_str(sentence)
            new.m_word.append(sentence.split(" "))
            new.m_label = label
            newList.append(new)

            if count == -1:
                break

        random.shuffle(newList)
        f.close()
        return newList[:int(len(newList) * 0.7)], \
               newList[int(len(newList) * 0.7):int(len(newList) * 0.8)],\
               newList[int(len(newList) * 0.8):]

    # class Reader():
    #     def readfile(self, path):
    #         f = open(path, 'r')
    #         newList = []
    #         for line in f.readlines():
    #             new = Inst()
    #             x = line.strip().split("||| ")  # big bug big bug big bug stupid bug stupid bug
    #             new.m_word.append(x[0].split(" "))
    #             new.m_label = x[1]
    #             newList.append(new)
    #         f.close()
    #         return newList

    def clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip()


class Classifier(object):
    def __init__(self):
        self.testLabel = vocab()
        self.testWord = vocab()

        self.hyperparameter = HyperParameters()

    def eval(self, data_iter, model, args, test):
        runTimeList = []
        # model.eval()
        total_num = len(data_iter)
        print("dev num:", total_num)

        part = total_num // self.hyperparameter.batch
        if total_num % self.hyperparameter.batch != 0:
            part += 1
        for n in range(1):

            # print('第%d次迭代：' % (n + 1))
            correct = 0
            sum = 0
            for idx in range(part):
                begin = idx * self.hyperparameter.batch
                end = (idx + 1) * self.hyperparameter.batch
                if end > total_num:
                    end = total_num
                batch_list = []
                for idy in range(begin, end):
                    batch_list.append(data_iter[idy])

                random.shuffle(batch_list)
                # print('第%d次迭代：' % (n + 1))
                #correct = 0
                #sum = 0
                # print("fe", batch_list[0])
                # optimizer.zero_grad()

                feature, target = self.ToVariable(batch_list)
                if self.hyperparameter.LSTM_model:
                    if feature.size(0) == self.hyperparameter.batch :
                        self.model.hidden = self.model.init_hidden(self.hyperparameter.num_layers, self.hyperparameter.batch)

                    else :
                        self.model.hidden = self.model.init_hidden(self.hyperparameter.num_layers, feature.size(0))

                # print("fea",feature)
                logit = self.model(feature)
                # print(logit)
                loss = F.cross_entropy(logit, target)  # 目标函数求导
                print("idx", idx, "dev loss:", loss.data[0])
                # loss.backward()
                # optimizer.step()
                for i in range(len(target)):
                    # print(logit[i])
                    # print(logit)
                    if target.data[i] == self.getMaxIndex(logit[i].view(1, self.hyperparameter.labelSize)):
                        correct += 1
                        # print(""correct)
                    # print('loss:',loss.data[0])
                    # print(correct)

                    sum += 1
                #print(sum)
            if test :
                print("test", end=" ")
            else :
                print("dev", end=" ")
            print("correct number:", correct)
            acc = correct / sum
            print('acc:', acc)

        return acc


    def load_my_vecs(self, path, vocab, freqs, k=None):
        word_vecs = {}
        with open(path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                # word = word.lower()
                # if word in vocab and freqs[word] != 1:  # whether to judge if in vocab
                count += 1
                if word in vocab:  # whether to judge if in vocab
                    # if word in vocab:  # whether to judge if in vocab
                    #     if count % 5 == 0 and freqs[word] == 1:
                    #         continue
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknown_words_by_uniform(self, word_vecs, vocab, k=100):
        list_word2vec = []
        oov = 0
        iov = 0
        # uniform = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
        for word in vocab:
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-0.25, 0.25, k).round(6).tolist()
                # word_vecs[word] = np.random.uniform(-0.1, 0.1, k).round(6).tolist()
                # word_vecs[word] = uniform
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        print("oov count", oov)
        print("iov count", iov)

        return list_word2vec

    def train(self, data):

        random.seed(0)
        torch.manual_seed(0)

        reader = Reader()

        initRst_train, initRst_dev, initRst_test = reader.readfile(data)
        # initRst_test = reader.readfile(test)
        # initRst_train = reader.readfile(train)

        (wordList_train, labelList_train) = self.ToList(initRst_train)

        word_dict = self.testWord.MakeVocab(wordList_train)
        label_dict = self.testLabel.MakeVocab(labelList_train)
        label_dict.pop("-unknown-")

        word2vec = self.load_my_vecs(path=self.hyperparameter.word_Embedding_Path,
                                     vocab=word_dict, freqs=None, k=300)
        self.hyperparameter.pretrained_weight = self.add_unknown_words_by_uniform(word_vecs=word2vec,
                                                                                  vocab=word_dict, k=300)
        Example_list_dev = self.SentenceInNum(initRst_dev, word_dict, label_dict)
        Example_list_test = self.SentenceInNum(initRst_test, word_dict, label_dict)
        Example_list_train = self.SentenceInNum(initRst_train, word_dict, label_dict)

        self.hyperparameter.unknown = word_dict["-unknown-"]
        self.hyperparameter.embedding_num = self.hyperparameter.unknown + 1
        self.hyperparameter.labelSize = len(label_dict)

        # NN = modelElite.model(self.hyperparameter)
        if self.hyperparameter.LSTM_model:
            self.model = modelElite.model(self.hyperparameter)
        else :
            self.model = modelElite_cnn.model(self.hyperparameter)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameter.lr)
        total_num = len(Example_list_train)
        print("train num:", total_num)

        part = total_num // self.hyperparameter.batch
        if total_num % self.hyperparameter.batch != 0:
            part += 1

        self.model.train()
        steps = 0
        model_count = 0
        listOfAcc = []
        for n in range(50):
            print('第%d次迭代：' % (n + 1))
            correct = 0
            sum = 0
            for idx in range(part):
                begin = idx * self.hyperparameter.batch
                end = (idx + 1) * self.hyperparameter.batch
                if end > total_num:
                    end = total_num
                batch_list = []
                for idy in range(begin, end):
                    batch_list.append(Example_list_train[idy])

                random.shuffle(batch_list)
                # print('第%d次迭代：' % (n + 1))
                #correct = 0
                #sum = 0
                # print("fe", batch_list[0])
                optimizer.zero_grad()
                self.model.zero_grad()
                feature, target = self.ToVariable(batch_list)
                if self.hyperparameter.LSTM_model:
                    if feature.size(0) == self.hyperparameter.batch :
                        self.model.hidden = self.model.init_hidden(self.hyperparameter.num_layers, self.hyperparameter.batch)

                    else :
                        self.model.hidden = self.model.init_hidden(self.hyperparameter.num_layers, feature.size(0))

                # print("fea",feature)
                logit = self.model(feature)
                # print("wwwwwww", logit.size())
                loss = F.cross_entropy(logit, target)  # 目标函数求导
                # print(loss)
                print("idx", idx, "loss:", loss.data[0])
                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                steps += 1
                for i in range(len(target)):
                    # print(logit[i])
                    # print(logit)
                    if target.data[i] == self.getMaxIndex(logit[i].view(1, self.hyperparameter.labelSize)):
                        correct += 1
                        # print(""correct)
                    # print('loss:',loss.data[0])
                    # print(correct)

                    sum += 1
                #print(sum)
            print("train", end=" ")
            print("correct number:", correct)
            print('acc:', correct / sum)

            devAcc=self.eval(Example_list_dev, self.model, self.hyperparameter, test=False)

            if not os.path.isdir(self.hyperparameter.save_dir): os.makedirs(self.hyperparameter.save_dir)
            save_prefix = os.path.join(self.hyperparameter.save_dir, 'snapshot')
            save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            torch.save(self.model, save_path)
            print("\n", save_path, end=" ")
            test_model = torch.load(save_path)
            model_count += 1

            # def test_eval(data_iter, model, save_path, args, model_count):

            testAcc = self.eval(Example_list_test, test_model, self.hyperparameter, test=True)
            listOfAcc.append(testAcc)
        max = listOfAcc[0]
        location =0
        for i in listOfAcc:
            if i > max :
                max = i
        for n in range(len(listOfAcc)):
            if listOfAcc[n] == max :
                print("max acc located in ", n+1)
                # break

        print("test max acc: ", max)


    def ToList(self, InitList):
        wordList = []
        labelList = []
        for i in range(len(InitList)):
            for j in InitList[i].m_word[0]:
                wordList.append(j)
            labelList.append(InitList[i].m_label)
        print(len(wordList))
        return wordList, labelList

    def SentenceInNum(self, initRst, word_dict, label_dict):
        example_list = []
        for i in range(len(initRst)):
            dist = example()
            for j in initRst[i].m_word[0]:
                if j in word_dict:
                    id = word_dict[j]
                else:
                    id = word_dict["-unknown-"]
                dist.word_indexes.append(id)
            num = label_dict[initRst[i].m_label]
            dist.label_index.append(num)
            example_list.append(dist)
        return example_list

    def ToVariable(self, Examples):  # 输入为包含batch个example的list

        batch = len(Examples)
        maxLength = 0
        for i in range(len(Examples)):
            if len(Examples[i].word_indexes) > maxLength:
                maxLength = len(Examples[i].word_indexes)

        x = Variable(torch.LongTensor(batch, maxLength))
        y = Variable(torch.LongTensor(batch))
        for i in range(len(Examples)):
            for n in range(len(Examples[i].word_indexes)):
                x.data[i][n] = Examples[i].word_indexes[n]
                for j in range(len(Examples[i].word_indexes), maxLength):
                    x.data[i][j] = self.hyperparameter.unknown
            y.data[i] = Examples[i].label_index[0]
            # print("Y:", y)

        return x, y

    def getMaxIndex(self, score):
        # print("sss", score)
        labelsize = score.size()[1]
        max = score.data[0][0]
        maxIndex = 0
        for idx in range(labelsize):
            tmp = score.data[0][idx]
            if max < tmp:
                max = tmp
                maxIndex = idx
        return maxIndex


test = Classifier()

# test.train("./data/raw.clean.train", "./data/raw.clean.train", "./data/raw.clean.train")
test.train("./data/subj.all")
# test.train("./data/sample.txt", "./data/sample.txt", "./data/sample.txt")
# test.train("./data/rt-polarity.all", "./data/rt-polarity.all", "./data/rt-polarity.all")
