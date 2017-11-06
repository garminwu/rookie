import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from torch.autograd import Variable

class model(nn.Module):
    def __init__(self,parameters):                      #parameters为hyperparameters的实例
        super(model, self).__init__()
        self.parameter = parameters
        self.embed_num = self.parameter.embedding_num  #dict为字典类型 其长度比编号大一
        self.embed_dim = self.parameter.embedding_dim
        self.class_num = self.parameter.labelSize      #几分类
        self.hidden_dim = self.parameter.LSTM_hidden_dim
        self.num_layers = self.parameter.num_layers

        print("embedding中词的数量：", self.embed_num)
        self.embedding = nn.Embedding(self.embed_num, self.embed_dim)
        #预训练 （glove）
        if self.parameter.word_Embedding:
            pretrained_weight = np.array(self.parameter.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # fixed the word embedding
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, dropout=self.parameter.dropout, num_layers=self.num_layers)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim, self.class_num)
        # hidden
        self.hidden = self.init_hidden(self.num_layers, self.parameter.batch)
        # dropout
        self.dropout = nn.Dropout(self.parameter.dropout)

        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        # return (Variable(torch.zeros(1, batch_size, self.hidden_dim)),
        #          Variable(torch.zeros(1, batch_size, self.hidden_dim)))
        # return ""
        return (Variable(torch.zeros(1 * num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(1 * num_layers, batch_size, self.hidden_dim)))

    def forward(self, x):
        embed = self.embedding(x)
        embed = self.dropout_embed(embed)
        # print(embed.size())
        x = embed.view(len(x), embed.size(1), -1)
        # print("*"*10,x.size())
        # lstm

        # print(self.hidden[0].size())
        lstm_out, self.hidden = self.lstm(x.permute(1,0,2), self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)
        # linear
        logit = self.hidden2label(lstm_out)
        return logit