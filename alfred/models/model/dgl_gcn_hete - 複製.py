import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl.function as fn
import dgl.nn.pytorch as dglnn
# import nn.vnn as vnn
# nodes
PATH_OBJECT = '../graph_analysis/data_dgl/object.csv'
PATH_ROOM = '../graph_analysis/data_dgl/room.csv'
PATH_ATTRIBUTE = '../graph_analysis/data_dgl/attribute.csv'
PATH_SUBGOAL = '../graph_analysis/data_dgl/subgoal.csv'
PATH_LOWACTION = '../graph_analysis/data_dgl/lowaction.csv'
# edges
PATH_OBJECT_OBJECT = '../graph_analysis/data_dgl/object-interact-object.csv'
PATH_ROOM_OBJECT = '../graph_analysis/data_dgl/room-interact-object.csv'
PATH_ATTRIBUTE_OBJECT = '../graph_analysis/data_dgl/attribute-behave-object.csv'
PATH_LOWACTION_SUBGOAL = '../graph_analysis/data_dgl/lowaction-behave-subgoal.csv'


# import pdb; pdb.set_trace()
def load_heterograph_data():
    # edges
    src_objectobject, dst_objectobject = read_edge(PATH_OBJECT_OBJECT)
    src_roomobject, dst_roomobject = read_edge(PATH_ROOM_OBJECT)
    src_attributeobject, dst_attributeobject = read_edge(PATH_ATTRIBUTE_OBJECT)
    src_attributeobject, dst_attributeobject = read_edge(PATH_ATTRIBUTE_OBJECT)

    graph_data = {
        ('object', 'interacts', 'object'): (src_objectobject, dst_objectobject),
        ('room', 'interacts', 'object'): (src_roomobject, dst_roomobject),
        ('attribute', 'behave', 'object'): (src_attributeobject, dst_attributeobject),
    }
    # graph
    g = dgl.heterograph(graph_data)
    # nodes
    g.nodes['object'].data['feature'] = read_node_data(PATH_OBJECT)
    g.nodes['attribute'].data['feature'] = read_node_data(PATH_ATTRIBUTE)
    g.nodes['room'].data['feature'] = read_node_data(PATH_ROOM)
    return g


def read_node_data(path):
    def get_feature(csv_nodes_data):
        feature = [csv_nodes_data['feature'].to_list()]
        for i in range(1, 300):
            feature.extend([csv_nodes_data['feature.{}'.format(i)].to_list()])
        feature = torch.tensor(feature).float().transpose(0, 1)
        return feature

    nodes_data = pd.read_csv(path)
    node_feature = get_feature(nodes_data)
    return node_feature


def read_edge(path):
    edges_data = pd.read_csv(path)
    src_node1_node2 = edges_data['Src'].to_numpy()
    dst_node1_node2 = edges_data['Dst'].to_numpy()
    return src_node1_node2, dst_node1_node2


# https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#dgl.nn.pytorch.HeteroGraphConv
# https://docs.dgl.ai/en/latest/guide/nn-heterograph.html
class NetGCN(nn.Module):
    def __init__(self, HETAttention, o_feats_dgcn, device=0):
        '''
        HETAttention : bool
        o_feats_dgcn : int
        '''
        super(NetGCN, self).__init__()
        self.device = torch.device("cuda:%d" % device if torch.cuda.is_available() else "cpu")
        self.HETAttention = HETAttention
        # graph data
        self.g = load_heterograph_data()
        self.g = self.g.to(self.device)
        # para define
        in_feats = self.g.nodes['object'].data['feature'].shape[1]
        if self.HETAttention:
            h_feats = o_feats_dgcn
        else:
            h_feats = 32
            num_object = self.g.num_nodes("object")
            self.final_mapping = nn.Linear(num_object*h_feats, o_feats_dgcn)
        # graph model define
        self.conv1 = dglnn.HeteroGraphConv({
            'interacts': dglnn.GraphConv(in_feats, h_feats, activation=nn.ReLU()),
            'behave': dglnn.GraphConv(in_feats, h_feats, activation=nn.ReLU())},
            aggregate='mean'
        )
        self.conv2 = dglnn.HeteroGraphConv({
            'interacts': dglnn.GraphConv(h_feats, h_feats)},
            aggregate='mean'
        )

    def forward(self, frames):
        in_feat = self.g.ndata['feature']
        h = self.conv1(self.g, in_feat)
        # {'object': tensor([[-0.0047,
        # [object len, h_feats], i.e. [108, 32] or [108, 1024]
        h = self.conv2(self.g, h)
        if self.HETAttention:
            # [108, 1024]
            x = h["object"]
            # [2, 108, 1024]
            x = x.unsqueeze(0).repeat(frames.shape[0], 1, 1)
            # import pdb; pdb.set_trace()
        else:
            # [1, 3456]
            x = h["object"].view(1, -1)
            # [1, 512]
            x = self.final_mapping(x)
            # [2, 61, 512]
            x = x.unsqueeze(0).repeat(frames.shape[0], frames.shape[1], 1)
        return x
