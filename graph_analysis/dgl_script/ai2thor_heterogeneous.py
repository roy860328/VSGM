import dgl
import pandas as pd
import torch
import torch.nn.functional as F


# import pdb; pdb.set_trace()
def load_object_interact_object():
    nodes_data = pd.read_csv('../data_dgl/object.csv')
    edges_data = pd.read_csv('../data_dgl/object-interact-object.csv')
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    Ids = nodes_data['Id'].to_list()
    Ids = torch.tensor(Ids).long()
    # We can also convert it to one-hot encoding.
    feature = [nodes_data['feature'].to_list()]
    for i in range(1, 300):
        feature.extend([nodes_data['feature.{}'.format(i)].to_list()])
    feature = torch.tensor(feature).long().transpose(0, 1)
    import pdb; pdb.set_trace()
    g.ndata.update({'Ids': Ids, 'feature': feature})
    return g
g = load_object_interact_object()
print(g)