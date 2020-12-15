import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, ChebConv
import graph_embed

class Net(torch.nn.Module):
    """docstring for Net"""
    def __init__(self, cfg):
        super(Net, self).__init__()
        input_size = cfg.SCENE_GRAPH.NODE_FEATURE_SIZE
        middle_size = cfg.SCENE_GRAPH.NODE_MIDDEL_FEATURE_SIZE
        output_size = cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE
        # True => one of the variables needed for gradient computation has been modified by an inplace operation
        normalize = cfg.SCENE_GRAPH.NORMALIZATION
        graph_embed_model = getattr(graph_embed, cfg.SCENE_GRAPH.EMBED_TYPE)
        self.cfg = cfg
        self.conv1 = GCNConv(input_size, middle_size, cached=True,
                             normalize=normalize,
                             # add_self_loops=False
                             )
        self.conv2 = GCNConv(middle_size, output_size, cached=True,
                             normalize=normalize,
                             # add_self_loops=False
                             )
        # self.conv1 = ChebConv(input_size, middle_size, K=2)
        # self.conv2 = ChebConv(middle_size, output_size, K=2)
        self.final_mapping = graph_embed_model(cfg)

    def forward(self, data):
        '''
        data.x
        tensor([[-0.0474,  0.0324,  0.1443,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0440, -0.0058,  0.0014,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0057,  0.0471,  0.0377,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0724, -0.0065, -0.0210,  ...,  0.0000,  0.0000,  0.0000],
                [-0.0474,  0.0324,  0.1443,  ...,  1.0000,  0.0000,  0.0000]],
               grad_fn=<CatBackward>)
        data.edge_index
        tensor([[3, 0],
                [3, 1],
                [3, 2],
                [3, 4]])
        data.obj_cls_to_ind
        {64: [0, 4], 70: [1], 47: [2], 81: [3]}
        data.obj_id_to_ind
        {'Pillow|-02.89|+00.62|+00.82': 0, 'RemoteControl|-03.03|+00.56|+02.01': 1, 'Laptop|-02.81|+00.56|+01.81': 2, 'Sofa|-02.96|+00.08|+01.39': 3, 'Pillow|-02.89|+00.62|+01.19': 4}
        '''
        # import pdb; pdb.set_trace()
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, edge_index, edge_weight = data.x.clone().detach(), data.edge_index, data.edge_attr
        if edge_index is not None:
            edge_index = edge_index.clone().detach()
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            # x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = self.conv2(x, edge_index, edge_weight)
        x = self.final_mapping(x)
        return x
