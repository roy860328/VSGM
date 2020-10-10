import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl.function as fn
import dgl.nn.pytorch as dglnn
# import nn.vnn as vnn


# import pdb; pdb.set_trace()
def load_heterograph():
    # edges
    edges_data = pd.read_csv('../graph_analysis/data_dgl/object-interact-object.csv')
    src_objectobject = edges_data['Src'].to_numpy()
    dst_objectobject = edges_data['Dst'].to_numpy()
    edges_data = pd.read_csv('../graph_analysis/data_dgl/room-interact-object.csv')
    src_roomobject = edges_data['Src'].to_numpy()
    dst_roomobject = edges_data['Dst'].to_numpy()
    edges_data = pd.read_csv('../graph_analysis/data_dgl/attribute-behave-object.csv')
    src_attributeobject = edges_data['Src'].to_numpy()
    dst_attributeobject = edges_data['Dst'].to_numpy()
    graph_data = {
        ('object', 'interacts', 'object'): (src_objectobject, dst_objectobject),
        ('room', 'interacts', 'object'): (src_roomobject, dst_roomobject),
        ('attribute', 'behave', 'object'): (src_attributeobject, dst_attributeobject),
    }
    # graph
    g = dgl.heterograph(graph_data)
    # nodes
    csv_nodes_object = pd.read_csv('../graph_analysis/data_dgl/object.csv')
    csv_nodes_attribute = pd.read_csv('../graph_analysis/data_dgl/attribute.csv')
    csv_nodes_room = pd.read_csv('../graph_analysis/data_dgl/room.csv')
    g.nodes['object'].data['feature'] = _get_feature(csv_nodes_object)
    g.nodes['attribute'].data['feature'] = _get_feature(csv_nodes_attribute)
    g.nodes['room'].data['feature'] = _get_feature(csv_nodes_room)
    return g


def _get_feature(csv_nodes_data):
    feature = [csv_nodes_data['feature'].to_list()]
    for i in range(1, 300):
        feature.extend([csv_nodes_data['feature.{}'.format(i)].to_list()])
    feature = torch.tensor(feature).float().transpose(0, 1)
    return feature


# https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#dgl.nn.pytorch.HeteroGraphConv
# https://docs.dgl.ai/en/latest/guide/nn-heterograph.html
class NetGCN(nn.Module):
    def __init__(self, o_feats_dgcn, device=0):
        super(NetGCN, self).__init__()
        device = torch.device("cuda:%d" % device if torch.cuda.is_available() else "cpu")
        self.g = load_heterograph()
        self.g = self.g.to(device)
        in_feats = self.g.nodes['object'].data['feature'].shape[1]
        h_feats = 32
        num_object = self.g.num_nodes("object")
        self.conv1 = dglnn.HeteroGraphConv({
            'interacts': dglnn.GraphConv(in_feats, h_feats, activation=nn.ReLU()),
            'behave': dglnn.GraphConv(in_feats, h_feats, activation=nn.ReLU())},
            aggregate='mean'
        )
        self.conv2 = dglnn.HeteroGraphConv({
            'interacts': dglnn.GraphConv(h_feats, h_feats)},
            aggregate='mean'
        )
        self.final_mapping = nn.Linear(num_object*h_feats, o_feats_dgcn)

    def forward(self, frames):
        in_feat = self.g.ndata['feature']
        h = self.conv1(self.g, in_feat)
        h = self.conv2(self.g, h)
        # [1, 3456]
        x = h["object"].view(1, -1)
        # [1, 512]
        x = self.final_mapping(x)
        # [2, 61, 512]
        x = x.unsqueeze(0).repeat(frames.shape[0], frames.shape[1], 1)
        # import pdb; pdb.set_trace()
        return x
