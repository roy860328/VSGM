import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import graph_embed


# https://github.com/rusty1s/pytorch_geometric/issues/1083
class Net(torch.nn.Module):
    """docstring for Net"""
    def __init__(self, cfg, config=None):
        super(Net, self).__init__()
        input_size = cfg.SCENE_GRAPH.NODE_FEATURE_SIZE
        middle_size = cfg.SCENE_GRAPH.NODE_MIDDEL_FEATURE_SIZE
        ATTRIBUTE_FEATURE_SIZE = cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE
        output_size = cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE
        # True => one of the variables needed for gradient computation has been modified by an inplace operation
        normalize = cfg.SCENE_GRAPH.NORMALIZATION
        self.cfg = cfg
        self.conv1 = GCNConv(input_size, middle_size, cached=True,
                             normalize=normalize,
                             )
        self.conv2 = GCNConv(middle_size + ATTRIBUTE_FEATURE_SIZE, output_size, cached=True,
                             normalize=normalize,
                             )
        # self.attri_linear = nn.Linear(ATTRIBUTE_FEATURE_SIZE, ATTRIBUTE_FEATURE_SIZE
        #                               )
        graph_embed_model = getattr(graph_embed, cfg.SCENE_GRAPH.EMBED_TYPE)
        NODE_FEATURE_SIZE = cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE + cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE
        EMBED_FEATURE_SIZE = cfg.SCENE_GRAPH.EMBED_FEATURE_SIZE
        self.final_mapping = graph_embed_model(
            INPUT_FEATURE_SIZE=NODE_FEATURE_SIZE,
            EMBED_FEATURE_SIZE=EMBED_FEATURE_SIZE
        )
        if cfg.SCENE_GRAPH.CHOSE_IMPORTENT_NODE:
            # nn.linear bert_hidden_size -> NODE_FEATURE_SIZE
            bert_hidden_size = config['general']['model']['block_hidden_dim']
            NUM_CHOSE_NODE = cfg.SCENE_GRAPH.NUM_CHOSE_NODE
            self.chose_node_module = graph_embed.DotAttnChoseImportentNode(
                bert_hidden_size,
                NODE_FEATURE_SIZE,
                NUM_CHOSE_NODE,
                cfg.SCENE_GRAPH.GPU
            )

    def forward(self, data, hidden_state=None):
        '''
        data.x
        tensor([[-0.0474,  0.0324,  0.1443,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0440, -0.0058,  0.0014,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0057,  0.0471,  0.0377,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0724, -0.0065, -0.0210,  ...,  0.0000,  0.0000,  0.0000],
                [-0.0474,  0.0324,  0.1443,  ...,  1.0000,  0.0000,  0.0000]],
               grad_fn=<CatBackward>)
        data.edge_obj_to_obj
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
        x, attributes, edge_obj_to_obj, edge_weight = \
            data.x.clone().detach(), \
            data.attributes.clone().detach(), \
            data.edge_obj_to_obj, \
            data.edge_attr
        if edge_obj_to_obj is not None:
            edge_obj_to_obj = edge_obj_to_obj.clone().detach()
            # torch.Size([2, 16])
            x = F.relu(self.conv1(x, edge_obj_to_obj, edge_weight))
            x = F.dropout(x, training=self.training)
            x = torch.cat([x, attributes], dim=1)
            # torch.Size([2, 40])
            x = self.conv2(x, edge_obj_to_obj, edge_weight)
            # torch.Size([2, 16]) + torch.Size([2, 24]) => torch.Size([2, 40])
            x = torch.cat([x, attributes], dim=1)
            if self.cfg.SCENE_GRAPH.CHOSE_IMPORTENT_NODE:
                # torch.Size([1, 400])
                chose_nodes = self.chose_node_module(x, hidden_state)

            # torch.Size([1, 128])
            x = self.final_mapping(x)
            # torch.Size([1, 528]) = self.cfg.SCENE_GRAPH.RESULT_FEATURE
            x = torch.cat([x, chose_nodes], dim=1)

        else:
            x = torch.zeros((1, self.cfg.SCENE_GRAPH.RESULT_FEATURE))
            if self.cfg.SCENE_GRAPH.GPU:
                x = x.to('cuda')
        return x