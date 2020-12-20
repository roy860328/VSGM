import torch
from torch import nn
from torch.nn import functional as F


# https://docs.dgl.ai/en/latest/tutorials/models/3_generative_model/5_dgmg.html#dgmg-the-main-flow
class WeightedSum(nn.Module):
    def __init__(self, INPUT_FEATURE_SIZE, EMBED_FEATURE_SIZE):
        super(WeightedSum, self).__init__()

        # Setting from the paper
        self.node_feature_size = INPUT_FEATURE_SIZE
        self.embed_feature_SIZE = EMBED_FEATURE_SIZE

        # Embed graphs
        ### every nodes weight
        self.node_gating = nn.Sequential(
            nn.Linear(self.node_feature_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(self.node_feature_size,
                                       self.embed_feature_SIZE)

    def forward(self, features):
        '''
        features: torch.Size([10, 40])
        '''
        # import pdb; pdb.set_trace()
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


class DotAttnChoseImportentNode(nn.Module):
    def __init__(self, bert_hidden_size, NODE_FEATURE_SIZE, NUM_CHOSE_NODE, GPU):
        super().__init__()
        self.NUM_CHOSE_NODE = NUM_CHOSE_NODE
        self.bert_hidden_size = bert_hidden_size
        self.OUTPUT_SHAPE = NODE_FEATURE_SIZE * NUM_CHOSE_NODE
        self.GPU = GPU
        # 64 -> 40
        self.hidden_state_to_node = nn.Linear(bert_hidden_size, NODE_FEATURE_SIZE)

    def forward(self, nodes, hidden_state):
        '''
        nodes: torch.Size([3, 40])
        hidden_state: torch.Size([1, 64])
        '''
        # import pdb; pdb.set_trace()
        if hidden_state is None:
            print("WARNING hidden_state is None")
            hidden_state = torch.zeros((1, self.bert_hidden_size))
            if self.GPU:
                hidden_state = hidden_state.to('cuda')
        # torch.Size([1, 40])
        hidden_state = self.hidden_state_to_node(hidden_state)
        # torch.Size([3])
        score = self.softmax(nodes, hidden_state)

        chose_nodes = None
        sort_index = torch.argsort(score, dim=0)
        for index in sort_index.to('cpu').numpy():
            if index < self.NUM_CHOSE_NODE:
                node = nodes[index].unsqueeze(0)
                if chose_nodes is None:
                    chose_nodes = node
                else:
                    chose_nodes = torch.cat((chose_nodes, node), dim=1)
        # can chose nodes smaller than OUTPUT_SHAPE, cat zeros vectors
        if self.OUTPUT_SHAPE != chose_nodes.shape[-1]:
            tensor_zeros = torch.zeros((chose_nodes.shape[0], self.OUTPUT_SHAPE - chose_nodes.shape[-1]))
            if self.GPU:
                tensor_zeros = tensor_zeros.to('cuda')
            chose_nodes = torch.cat((chose_nodes, tensor_zeros), dim=1)
        return chose_nodes

    def softmax(self, nodes, hidden_state):
        '''
        nodes: torch.Size([3, 40])
        hidden_state : torch.Size([1, 40])
        '''
        # import pdb; pdb.set_trace()
        # nodes * hidden_state -> torch.Size([3, 1]) -> torch.Size([3])
        score = torch.matmul(nodes, hidden_state.T).reshape(-1)
        return score
