import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self,parameters):                      #parameters为hyperparameters的实例
        super(model, self).__init__()
        self.parameter = parameters
        self.embed_num = self.parameter.embedding_num  #dict为字典类型 其长度比编号大一
        self.embed_dim = self.parameter.embedding_dim
        self.embedding = nn.Embedding(self.embed_num,self.embed_dim)
        self.line = nn.Linear(self.embed_dim, self.parameter.labelSize)

    def forward(self, x):
        # print("16:", x.size())
        x = self.embedding(x)
        # print("18:", x.size())
        # print("19", x.permute(0,2,1).size())
        # maxpooling_rst = F.max_pool1d(x.permute(0, 2, 1), x.size()[1]).squeeze(2) #permute转换对应位置的维度 squeeze
        maxpooling_rst = F.max_pool1d(x.permute(0, 2, 1), x.size()[1]) #permute转换对应位置的维度 squeeze

        # print(maxpooling_rst.size())
        Linear_rst = self.line(maxpooling_rst.view(1, self.embed_dim))
        # Linear_rst = self.line(maxpooling_rst)
        return Linear_rst





