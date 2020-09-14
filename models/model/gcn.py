import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import scipy.sparse as sp
import numpy as np
import h5py


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GCN(nn.Module):
    def __init__(self, noutfeat, nhid=1024, dropout=0.5):
        super(GCN, self).__init__()

        # get and normalize adjacency matrix.
        A_raw = torch.load("./data/gcn/adjmat.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        # glove embeddings for all the objs.
        objects = open("./data/gcn/objects.txt").readlines()
        objects = [o.strip() for o in objects]
        n = 83
        self.n = n
        all_glove = torch.zeros(n, 300)
        glove = Glove("./data/gcn/glove_map300d.hdf5")
        for i in range(n):
            all_glove[i, :] = torch.Tensor(glove.glove_embeddings[objects[i]][:])
        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False

        # GCN layer
        self.gc1 = GraphConvolution(self.all_glove.shape[1], nhid)
        self.gc2 = GraphConvolution(nhid, 1)
        self.dropout = dropout

        self.final_mapping = nn.Linear(self.n, noutfeat)


    def forward(self, batch_size):
        word_embed = self.all_glove.detach()
        x = F.relu(self.gc1(word_embed, self.A))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, self.A))
        x = x.view(1, self.n)
        x = x.repeat(batch_size, 1)
        x = self.final_mapping(x)
        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Glove:
    def __init__(self, glove_file):
        self.glove_embeddings = h5py.File(glove_file, "r")