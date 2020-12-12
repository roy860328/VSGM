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