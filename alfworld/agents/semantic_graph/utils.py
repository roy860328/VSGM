'''
# https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html
# https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=Y9MOs8iSwKFD
G = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)
# https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing#scrollTo=9r_VmGMukf5R
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
'''
import os
import glob
import numpy as np
from torch_geometric.utils import to_networkx
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def visualize_node(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        # nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
        #                  node_color=color, cmap="Set2")
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    return plt


def visualize_node_feature(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    return plt


def save_graph_data(graph_data, path):
    num_obj_cls = len(graph_data.obj_cls_to_features.keys()) + 1
    colors = cm.rainbow(np.linspace(0, 1, num_obj_cls))
    node_to_color = [colors[node_obj_cls_ind] for node_obj_cls_ind in graph_data.list_node_obj_cls]
    G = to_networkx(graph_data, to_undirected=False)
    plt = visualize_node(G, color=node_to_color)
    save_plt_img(plt, path, "node_")
    plt = visualize_node_feature(graph_data.x, color=node_to_color)
    save_plt_img(plt, path, "node_feature_")


def save_plt_img(plt, path, name):
    idx = len(glob.glob(path + '/{}*.png'.format(name)))
    name = os.path.join(path, '%s%09d.png' % (name, idx))
    plt.savefig(name)


def load_graph_data(path):
    pass