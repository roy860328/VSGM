import torch
from torch import nn
from torch.nn import functional as F


# https://docs.dgl.ai/en/latest/tutorials/models/3_generative_model/5_dgmg.html#dgmg-the-main-flow
class WeightedSum(nn.Module):
    def __init__(self, cfg):
        super(WeightedSum, self).__init__()

        # Setting from the paper
        self.node_feature_size = cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE
        self.embed_feature_SIZE = cfg.SCENE_GRAPH.EMBED_FEATURE_SIZE

        # Embed graphs
        ### every nodes weight
        self.node_gating = nn.Sequential(
            nn.Linear(self.node_feature_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(self.node_feature_size,
                                       self.embed_feature_SIZE)

    def forward(self, features):
        merge_features = (self.node_gating(features) * self.node_to_graph(features)).sum(0, keepdim=True)
        return merge_features


class SelfAttn(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.node_feature_size = cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE
        self.embed_feature_SIZE = cfg.SCENE_GRAPH.EMBED_FEATURE_SIZE

        self.scorer = nn.Linear(cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE, 1)

    def forward(self, features):
        scores = F.softmax(self.scorer(features), dim=1)
        merge_features = scores.bmm(features).sum(0, keepdim=True)
        return merge_features


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        # [2, 145, 1] -> [2, 145, 1024] -> * inp -> sum all 145 word to 1024 feature
        # -> [2, 1024]
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        '''
        inp : [2, 145, 1024]
        h : [2, 1024]
        '''
        # import pdb; pdb.set_trace()
        # [2, 145, 1]
        raw_score = inp.bmm(h.unsqueeze(2))
        # [2, 145, 1]
        score = F.softmax(raw_score, dim=1)
        return score