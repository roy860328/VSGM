import os
import sys
import torch
from torch_geometric.data import Data
from collections import defaultdict
import pandas as pd

class GraphData(Data):
    def __init__(self, obj_cls_name_to_features, GPU, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        super(GraphData, self).__init__(x=None, edge_index=None, edge_attr=None, y=None,
                                        pos=None, normal=None, face=None, **kwargs)
        # {'Fridge|-00.33|+00.00|-00.77': 0, 'Pot|-01.13|+00.94|-03.50': 0, 'CounterTop|-00.30|+00.95|-02.79': 1, 'Toaster|-00.15|+00.90|-02.76': 2, 'Fork|-00.30|+00.79|-01.86': 3, 'Bowl|-00.42|+00.08|-02.16': 4, 'Spatula|-00.30|+00.92|-02.54': 5, 'Cabinet|-00.58|+00.39|-01.80': 6}
        self.obj_id_to_ind = {}
        # {54: [0], 15: [1], 11: [2], 34: [3], 77: [4], 24: [5], 26: [6], 33: [7], 78: [8], 22: [9], 83: [10]}
        self.obj_cls_to_ind = defaultdict(list)
        self.obj_cls_to_features = obj_cls_name_to_features
        self.GPU = GPU

    def add_node(self, obj_cls, feature, obj_id=None):
        '''
        obj_cls : 24 (int), object class index
        obj_id : StoveKnob|-03.60|+01.11|+01.87
        feature : tensor size 23, tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.], requires_grad=True)

        return the index of the node
        '''
        # shape : 300 + 23
        feature = self._cat_wore_embed_and_attribute(obj_cls, feature).unsqueeze(0)
        # init self.x
        if self.x is None:
            ind = 0
            self.x = feature
        else:
            ind = len(self.x)
            self.x = torch.cat([self.x, feature], dim=0)
        # Oracle
        if obj_id is not None:
            self.obj_id_to_ind[obj_id] = ind
        # Not Oracle. Create new obj_id
        else:
            # creat_obj_id
            raise NotImplementedError
        self.obj_cls_to_ind[obj_cls].append(ind)
        return ind

    def update_node(self, obj_cls, feature, obj_id):
        '''
        obj_cls : 24, object class index
        obj_id : StoveKnob|-03.60|+01.11|+01.87

        return the index of the node
        '''
        # shape : 300 + 23
        feature = self._cat_wore_embed_and_attribute(obj_cls, feature)
        ind = self.obj_id_to_ind[obj_id]
        self.x[ind] = feature
        return ind

    def update_relation(self, node_src, node_dst, relation):
        '''
        node_src : 320
        node_dst : 321
        relation : 1
        '''
        tensor_node = torch.tensor([[node_src], [node_dst]], dtype=torch.long).contiguous()
        if self.GPU:
            tensor_node = tensor_node.cuda()
        if self.edge_index is None:
            self.edge_index = tensor_node
        else:
            '''
            tensor([[ 5,  0],
                    [ 9,  0],
                    [ 2,  1],
                    [ 8,  1],
                    [ 8,  2],
                    [ 8,  6],
                    [ 8,  7],
                    [ 5,  9],
                    [ 6, 10],
                    [ 8, 10]])
            '''
            self.edge_index = torch.cat([self.edge_index, tensor_node], dim=1)

    def search_node(self, obj_id=None):
        # Oracle
        if obj_id is not None:
            if obj_id in self.obj_id_to_ind:
                return True
            else:
                return False
        # compare alg
        else:
            raise NotImplementedError

    def _cat_wore_embed_and_attribute(self, obj_cls, attribute):
        word_embed = self.obj_cls_to_features[obj_cls].clone().detach()
        attribute = attribute.clone().detach()
        feature = torch.cat([word_embed, attribute])
        if self.GPU:
            feature = feature.cuda()
        return feature


class SceneGraph(object):
    """docstring for SceneGraph"""

    def __init__(self, cfg):
        super(SceneGraph, self).__init__()
        self.cfg = cfg
        self.isORACLE = cfg.SCENE_GRAPH.ORACLE
        self.GPU = cfg.SCENE_GRAPH.GPU
        self.obj_cls_name_to_features = self._get_obj_cls_name_to_features()
        self.init_graph_data()

    def init_graph_data(self):
        self.global_graph = GraphData(self.obj_cls_name_to_features, self.GPU)

    def _get_obj_cls_name_to_features(self):
        def get_feature(csv_nodes_data):
            feature = [csv_nodes_data['feature'].to_list()]
            for i in range(1, 300):
                feature.extend([csv_nodes_data['feature.{}'.format(i)].to_list()])
            feature = torch.tensor(feature).float().transpose(0, 1)
            obj_cls_name_to_features = {}
            for i, obj_cls_name in enumerate(csv_nodes_data['Id'].to_list()):
                # for avoid class 0 __background__
                obj_cls_name += 1
                obj_cls_name_to_features[obj_cls_name] = feature[i]
            return obj_cls_name_to_features
        path_object_embedding = self.cfg.SCENE_GRAPH.OBJ_NAME_EMBEDDING
        path_object_embedding = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), path_object_embedding)
        csv_obj_features = pd.read_csv(path_object_embedding)
        obj_cls_name_to_features = get_feature(csv_obj_features)
        return obj_cls_name_to_features

    def get_graph_data(self):
        return self.global_graph

    def compare_existing_node(self, tar_id=None):
        isExist = False
        raise NotImplementedError

    def add_oracle_local_graph_to_global_graph(self, img, target):
        current_frame_obj_cls_to_node_index = []
        tar_ids = target["objectIds"]
        obj_clses = target["labels"].numpy().astype(int)
        obj_attributes = target["attributes"]
        # other way to process (index 4 is out of bounds for dimension 0 with size 4)
        obj_relations = target["relation_labels"].numpy().astype(int)
        '''
        tar_ids : ['Fridge|-01.50|+00.00|-00.70', 'Bread|+00.40|+00.96|+00.11|BreadSliced_6', 'Mug|-01.28|+01.01|-01.64', 'Microwave|-01.23|+00.90|-01.68', 'Cabinet|-00.84|+00.47|-01.67', 'CounterTop|+00.23|+00.95|-02.00', 'Drawer|-00.82|+00.75|-01.69']
        obj_cls : tensor(24.), object class index
        '''
        for i, tar_id in enumerate(tar_ids):
            obj_cls, obj_attribute = \
                obj_clses[i], obj_attributes[i]
            isFindNode = self.global_graph.search_node(obj_id=tar_id)
            if isFindNode:
                ind = self.global_graph.update_node(obj_cls, obj_attribute, tar_id)
            else:
                ind = self.global_graph.add_node(obj_cls, obj_attribute, tar_id)
            # [317, 318, 148, 149, 150, 151, 152, 319, 320, 321, 322, 154, 323, 155, 157, 158, 159, 160, 161, 324]
            current_frame_obj_cls_to_node_index.append(ind)
        for obj_relation_triplet in obj_relations:
            src_obj_ind, dst_obj_ind, relation = obj_relation_triplet
            src_node_ind = current_frame_obj_cls_to_node_index[src_obj_ind]
            dst_node_ind = current_frame_obj_cls_to_node_index[dst_obj_ind]
            self.global_graph.update_relation(src_node_ind, dst_node_ind, relation)

    def add_local_graph_to_global_graph(self, img, obj_features, obj_boxes, obj_relations, obj_attributes):
        raise NotImplementedError()
        obj_features = target["obj_features"]
        for i, obj_feature, obj_box, obj_relation, obj_attribute in enumerate(obj_features, obj_boxes, obj_relations, obj_attributes):
            isExist, obj_id = self.compare_existing_node()
            self.global_graph.num_nodes("object")


if __name__ == '__main__':
    sys.path.insert(0, os.environ['ALFRED_ROOT'])
    sys.path.insert(0, os.path.join(os.environ['ALFRED_ROOT'], 'agents'))
    import modules.generic as generic
    from sgg import alfred_data_format

    cfg = generic.load_config()
    cfg = cfg['semantic_cfg']
    scenegraph = SceneGraph(cfg)
    # trans_meta_data = alfred_data_format.TransMetaData(cfg)
    alfred_dataset = alfred_data_format.AlfredDataset(cfg)
    for i in range(len(alfred_dataset)):
        img, target, idx = alfred_dataset[i]
        dict_target = {
            "labels": target.extra_fields["objectIds"],
            "obj_relations": target.extra_fields["obj_relations"],
            "relation_labels": target.extra_fields["relation_labels"],
            "attributes": target.extra_fields["attributes"],
            "objectIds": target.extra_fields["objectIds"],
        }
        scenegraph.add_oracle_local_graph_to_global_graph(img, dict_target)
