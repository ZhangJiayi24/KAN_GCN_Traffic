import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolutionWithKAN

class GCNWithKAN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNWithKAN, self).__init__()

        # 使用 GraphConvolutionWithKAN 替换原有的 GraphConvolution
        self.gc1 = GraphConvolutionWithKAN(nfeat, nhid)
        self.gc2 = GraphConvolutionWithKAN(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)