from torch.autograd import Variable
import torch
import modelElite
import torch.nn.functional as F


class Inst(object): #定义Inst类型，将文件中的每句话生成Inst的数据类型
    def __init__(self):
        self.m_word = []        #将一句话中除label外的每个单词都存入m_word列表
        self.m_label = 0       #将一句话中的label存入m_label中
    # def show(self):
    #     print(self.m_word)
    #     print(self.m_label)


class vocab(object):            #生成词典
    def __init__(self):
        self.v_list = []
        self.v_dict = {}

    def MakeVocab(self,ListName):   #生成词典的过程
        for i in range(len(ListName)):
            if ListName[i] not in self.v_list:
                self.v_list.append(ListName[i])
        self.v_list.append("-unknown-")
        for n in range(len(self.v_list)):
            self.v_dict[self.v_list[n]] = n

        return self.v_dict


class example(object):          #
    def __init__(self):
        self.word_indexes = []
        self.label_index = []


class HyperParameters(object):
    def __init__(self):
        self.lr = 0.001
        self.dropout = 0
        self.epochs = 0
        self.hiddenSize = 0
        self.labelSize = 0
        self.embedding_num = 0
        self.embedding_dim = 50
        self.wordEmbeddingSize = 0
        self.unknown = 18281


class Reader():
    def readfile(self, path):
        f = open(path, 'r')
        newList = []
        for line in f.readlines():
            new = Inst()
            x = line.strip().split("||| ")  # big bug big bug big bug stupid bug stupid bug
            new.m_word.append(x[0].split(" "))
            new.m_label=x[1]
            newList.append(new)
        f.close()
        return newList


class Classifier(object):
    def __init__(self):
        self.testLabel = vocab()
        self.testWord = vocab()

        self.hyperparameter = HyperParameters()

    def train(self, test, dev, train):

        reader = Reader()

        initRst_dev = reader.readfile(dev)
        initRst_test = reader.readfile(test)
        initRst_train = reader.readfile(train)

        (wordList_train, labelList_train) = self.ToList(initRst_train)

        word_dict = self.testWord.MakeVocab(wordList_train)

        label_dict = self.testLabel.MakeVocab(labelList_train)
        label_dict.pop("-unknown-")

        Example_list_dev = self.SentenceInNum(initRst_dev, word_dict, label_dict)
        Example_list_test = self.SentenceInNum(initRst_test, word_dict, label_dict)
        # Example_list_train = self.SentenceInNum(initRst_train, word_dict, label_dict)
        Example_list_train = self.SentenceInNum(initRst_dev, word_dict, label_dict)

        self.hyperparameter.unknown = word_dict["-unknown-"]
        self.hyperparameter.embedding_num = self.hyperparameter.unknown + 1
        self.hyperparameter.labelSize = len(label_dict)

        # NN = modelElite.model(self.hyperparameter)
        #
        self.model = modelElite.model(self.hyperparameter)
        # optimizer = torch.optim.Adagrad(NN.parameters(), lr=self.hyperparameter.lr)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameter.lr)
        total_num = len(Example_list_train)
        print("train num:", total_num)
        for n in range(1, 100):
            print('第%d次迭代：' % n)
            correct = 0
            sum = 0
            for i in Example_list_train:
                optimizer.zero_grad()
                feature, target = self.ToVariable(i)
                # print("】ll")
                # print(target)
                # print(feature)
                # logit = NN(feature)
                logit = self.model(feature)
                # print(logit)
                loss = F.cross_entropy(logit, target)  # 目标函数求导
                # print(loss)
                loss.backward()
                optimizer.step()
                if target.data[0] == self.getMaxIndex(logit):
                    correct += 1
                # print('loss:',loss.data[0])
                # print(correct)

                sum += 1
            print(sum)
            print(correct)
            print('acc:', correct / sum)
        # print("词典")
        # print("\n")
        # print(word_dict)
        # print("\n"*5)
        # print(label_dict)
        # print("\n"*5)

        # for i in Example_list_dev:
        #     print(i.word_indexes)
        #     print(i.label_index)
        #
        # print("*"*50)
        #
        # for j in Example_list_test:
        #     print(j.word_indexes)
        #     print(j.label_index)
        #
        # print("*"*50)
        #
        # for k in Example_list_train:
        #     print(k.word_indexes)
        #     print(k.label_index)
        #
        # print("*"*50)
        #
        # print("\n"*2)
        # print("未知词的存储号码")
        # print(word_dict["-unknown-"])

    def ToList(self, InitList):
        wordList = []
        labelList = []
        for i in range(len(InitList)):
            for j in InitList[i].m_word[0]:
                wordList.append(j)
            labelList.append(InitList[i].m_label)
        return wordList, labelList

    def SentenceInNum(self,initRst,word_dict,label_dict):
        example_list = []
        for i in range(len(initRst)):
            dist = example()
            for j in initRst[i].m_word[0]:
                if j in word_dict:
                    id = word_dict[j]
                else :
                    id = word_dict["-unknown-"]
                dist.word_indexes.append(id)
            num = label_dict[initRst[i].m_label]
            dist.label_index.append(num)
            example_list.append(dist)
        return example_list

    def ToVariable(self,Example):           #输入为Example_List_xxx里面的一个元素

        x = Variable(torch.LongTensor(1,len(Example.word_indexes)))
        y = Variable(torch.LongTensor(1))
        for n in range(len(Example.word_indexes)):
            x.data[0][n] = Example.word_indexes[n]
        y.data[0] =  Example.label_index[0]
        # print("Y:", y)

        return x, y

    def getMaxIndex(self, score):
        labelsize = score.size()[1]
        max = score.data[0][0]
        maxIndex = 0
        for idx in range(labelsize):
            tmp = score.data[0][idx]
            if max < tmp:
                max = tmp
                maxIndex = idx
        return maxIndex



test =  Classifier()
test.train("./data/sample.txt", "./data/sample.txt", "./data/sample.txt")



