import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class model(nn.Module):
    def __init__(self,parameters):                      #parameters为hyperparameters的实例
        super(model, self).__init__()
        self.parameter = parameters
        self.embed_num = self.parameter.embedding_num  #dict为字典类型 其长度比编号大一
        self.embed_dim = self.parameter.embedding_dim

        print("embedding中词的数量：", self.embed_num)
        self.embedding = nn.Embedding(self.embed_num,self.embed_dim)
        #预训练 （glove）
        if self.parameter.word_Embedding:

            pretrained_weight = np.array(self.parameter.pretrained_weight)
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # fixed the word embedding
            self.embedding.weight.requires_grad = True

        #print("dddd {} ".format(self.embedding.weight.data.size()))
        #print(self.embedding.weight)

        self.convs = [nn.Conv2d(in_channels=1, out_channels=self.parameter.kernel_num, kernel_size=(K, self.embed_dim),
                                bias=True) for K in self.parameter.kernel_size]
        #print(self.convs)
        self.dropout = nn.Dropout(self.parameter.dropout)
        self.line = nn.Linear(self.parameter.kernel_num * len(self.parameter.kernel_size), self.parameter.labelSize)
        self.dropout_embed = nn.Dropout(self.parameter.dropout_embed)

    def forward(self, x):
        # print("16:", x.size())
        #print(x)
        # print(x.size())

        x = self.embedding(x)
        x = self.dropout_embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # print("18:", x[0].size())
        # print("19", x.permute(0,2,1).size())
        # maxpooling_rst = F.max_pool1d(x.permute(0, 2, 1), x.size()[1]).squeeze(2) #permute转换对应位置的维度 squeeze
        # maxpooling_rst = F.max_pool1d(x.permute(0, 2, 1), x.size()[1]).squeeze(2) #permute转换对应位置的维度 squeeze
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # print(x[0].size())
        x = torch.cat(x, 1) #1 竖向cat  0 横向cat
        # print(x.size())
        # print(maxpooling_rst.size())
        # Linear_rst = self.line(maxpooling_rst.view(1, self.embed_dim))
        x = self.dropout(x)
        Linear_rst = self.line(x)
        return Linear_rst





