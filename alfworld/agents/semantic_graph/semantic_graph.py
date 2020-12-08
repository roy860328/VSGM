import torch
from torch_geometric.data import Data
from collections import defaultdict
import pandas as pd


class GraphData(Data):
    def __init__(self, obj_cls_name_to_features, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        super(GraphData, self).__init__(x=None, edge_index=None, edge_attr=None, y=None,
                                        pos=None, normal=None, face=None, **kwargs)
        # {'Fridge|-00.33|+00.00|-00.77': 0, 'Pot|-01.13|+00.94|-03.50': 0, 'CounterTop|-00.30|+00.95|-02.79': 1, 'Toaster|-00.15|+00.90|-02.76': 2, 'Fork|-00.30|+00.79|-01.86': 3, 'Bowl|-00.42|+00.08|-02.16': 4, 'Spatula|-00.30|+00.92|-02.54': 5, 'Cabinet|-00.58|+00.39|-01.80': 6}
        self.obj_id_to_ind = {}
        # {tensor(37.): [0], tensor(67.): [0], tensor(24.): [1], tensor(93.): [2], tensor(36.): [3], tensor(11.): [4], tensor(82.): [5], tensor(17.): [6]})
        self.obj_cls_to_ind = defaultdict(list)
        self.obj_cls_to_features = obj_cls_name_to_features

    def add_node(self, obj_cls, feature, obj_id=None):
        '''
        obj_cls : tensor(24.), object class index
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
        obj_cls : tensor(24.), object class index
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
        node_src : 
        node_dst : 
        relation : 1
        '''
        tensor_node = torch.tensor([node_src, node_dst], dtype=torch.long).unsqueeze(0)
        if self.edge_index is None:
            self.edge_index = tensor_node
        else:
            self.edge_index = torch.cat([self.edge_index, tensor_node], dim=0)


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
        word_embed = self.obj_cls_to_features[obj_cls].clone().detach().requires_grad_(True)
        attribute = attribute.clone().detach().requires_grad_(True)
        feature = torch.cat([word_embed, attribute])
        return feature

class SceneGraph(object):
    """docstring for SceneGraph"""

    def __init__(self, cfg):
        super(SceneGraph, self).__init__()
        self.cfg = cfg
        self.isORACLE = cfg.SCENE_GRAPH.ORACLE
        self.init_graph_data()

    def init_graph_data(self):
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
        path_object_embedding = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_object_embedding)
        csv_obj_features = pd.read_csv(path_object_embedding)
        obj_cls_name_to_features = get_feature(csv_obj_features)
        self.scene_graph = GraphData(obj_cls_name_to_features)

    def compare_existing_node(self, tar_id=None):
        isExist = False
        raise NotImplementedError

    def add_oracle_local_graph_to_global_graph(self, img, target):
        current_frame_obj_cls_to_node_index = []
        tar_ids = target.extra_fields["objectIds"]
        obj_clses = target.extra_fields["labels"].numpy().astype(int)
        obj_attributes = target.extra_fields["attributes"]
        obj_boxes = target.bbox
        # other way to process (index 4 is out of bounds for dimension 0 with size 4)
        obj_relations = target.extra_fields["relation_labels"].numpy().astype(int)
        '''
        tar_ids : ['Fridge|-01.50|+00.00|-00.70', 'Bread|+00.40|+00.96|+00.11|BreadSliced_6', 'Mug|-01.28|+01.01|-01.64', 'Microwave|-01.23|+00.90|-01.68', 'Cabinet|-00.84|+00.47|-01.67', 'CounterTop|+00.23|+00.95|-02.00', 'Drawer|-00.82|+00.75|-01.69']
        obj_cls : tensor(24.), object class index
        '''
        for i, tar_id in enumerate(tar_ids):
            obj_cls, obj_box, obj_attribute = \
                obj_clses[i], obj_boxes[i], obj_attributes[i]
            isFindNode = self.scene_graph.search_node(obj_id=tar_id)
            if isFindNode:
                print("update_node")
                ind = self.scene_graph.update_node(obj_cls, obj_attribute, tar_id)
            else:
                ind = self.scene_graph.add_node(obj_cls, obj_attribute, tar_id)
            # [317, 318, 148, 149, 150, 151, 152, 319, 320, 321, 322, 154, 323, 155, 157, 158, 159, 160, 161, 324]
            current_frame_obj_cls_to_node_index.append(ind)
        for obj_relation_triplet in obj_relations:
            src_obj_ind, dst_obj_ind, relation = obj_relation_triplet
            src_node_ind = current_frame_obj_cls_to_node_index[src_obj_ind]
            dst_node_ind = current_frame_obj_cls_to_node_index[dst_obj_ind]
            self.scene_graph.update_relation(src_node_ind, dst_node_ind, relation)
        # import pdb;pdb.set_trace()


    def add_local_graph_to_global_graph(self, img, obj_features, obj_boxes, obj_relations, obj_attributes):
        obj_features = target["obj_features"]
        for i, obj_feature, obj_box, obj_relation, obj_attribute in enumerate(obj_features, obj_boxes, obj_relations, obj_attributes):
            isExist, obj_id = self.compare_existing_node()
            self.scene_graph.num_nodes("object")


if __name__ == '__main__':
    import os
    import sys
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
        scenegraph.add_oracle_local_graph_to_global_graph(img, target)
