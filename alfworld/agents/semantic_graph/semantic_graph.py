import os
import sys
import torch
from torch_geometric.data import Data
from collections import defaultdict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.join(os.environ['ALFRED_ROOT']))
from agents.semantic_graph import utils
import importlib


__all__ = ['GraphData', 'SceneGraph', 'HeteGraphData']


class GraphData(Data):
    def __init__(self, obj_cls_name_to_features, GPU, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        super(GraphData, self).__init__(x=None, edge_index=None, edge_attr=None, y=None,
                                        pos=None, normal=None, face=None, **kwargs)
        # {'ButterKnife|-00.06|+01.11|+00.25': 0, 'Lettuce|-00.06|+01.19|-00.76': 1, 'Statue|+00.22|+01.11|-00.51': 2, 'CounterTop|-00.08|+01.15|00.00': 3, 'Knife|+00.22|+01.14|+00.25': 4, 'LightSwitch|+02.33|+01.31|-00.16': 5, 'DishSponge|-00.06|+01.11|-00.25': 6, 'Vase|-00.34|+01.11|+00.25': 7, 'Book|+00.16|+01.10|+00.62': 8}
        self.obj_id_to_ind = {}
        # {0: 'ButterKnife|-00.06|+01.11|+00.25', 1: 'Lettuce|-00.06|+01.19|-00.76', 2: 'Statue|+00.22|+01.11|-00.51', 3: 'CounterTop|-00.08|+01.15|00.00', 4: 'Knife|+00.22|+01.14|+00.25', 5: 'LightSwitch|+02.33|+01.31|-00.16', 6: 'DishSponge|-00.06|+01.11|-00.25', 7: 'Vase|-00.34|+01.11|+00.25', 8: 'Book|+00.16|+01.10|+00.62'}
        self.ind_to_obj_id = {}
        # {15: [0], 50: [1], 85: [2], 24: [3], 45: [4], 52: [5], 30: [6], 102: [7], 10: [8]}
        self.obj_cls_to_ind = defaultdict(list)
        # list[0] = x[0] label
        # [15, 50, 85, 24, 45, 52, 30, 102, 10]
        self.list_node_obj_cls = []
        # edge
        self.edge_obj_to_obj = None
        # dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105])
        # self.obj_cls_to_features[1].shape 
        # torch.Size([300])
        self.obj_cls_to_features = obj_cls_name_to_features
        self.GPU = GPU

    def __inc__(self, key, value):
        increasing_funcs = ["edge_obj_to_obj"]
        if key in increasing_funcs:
            return len(self["x"])
        else:
            return 0

    def add_node(self, obj_cls, feature, obj_id=None):
        '''
        obj_cls : 24 (int), object class index
        obj_id : StoveKnob|-03.60|+01.11|+01.87
        feature : tensor size 23, tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.], requires_grad=True)

        return the index of the node
        '''
        # shape : 300 + 23
        feature = self._cat_feature_and_wore_embed(obj_cls, feature).unsqueeze(0)
        # init self.x
        if self.x is None:
            ind = 0
            self.x = feature
        else:
            ind = len(self.x)
            self.x = torch.cat([self.x, feature], dim=0)
        # Oracle
        if obj_id is None:
            # Not Oracle. Create new obj_id
            obj_id = str(obj_cls) + "_" + str(len(self.obj_cls_to_ind[obj_cls]))
        self.obj_id_to_ind[obj_id] = ind
        self.ind_to_obj_id[ind] = obj_id
        self.obj_cls_to_ind[obj_cls].append(ind)
        self.list_node_obj_cls.append(obj_cls)
        return ind

    def update_node(self, obj_cls, feature, obj_id):
        '''
        obj_cls : 24, object class index
        obj_id : StoveKnob|-03.60|+01.11|+01.87

        return the index of the node
        '''
        # shape : 300 + 23
        isFeatureChange = False
        feature = self._cat_feature_and_wore_embed(obj_cls, feature)
        ind = self.obj_id_to_ind[obj_id]
        if not torch.equal(self.x[ind], feature):
            isFeatureChange = True
        self.x[ind] = feature
        return ind, isFeatureChange

    def update_relation(self, node_src, node_dst, relation):
        '''
        node_src : 320
        node_dst : 321
        relation : 1
        '''
        tensor_node = torch.tensor([[node_src], [node_dst]], dtype=torch.long).contiguous()
        if self.GPU:
            tensor_node = tensor_node.cuda()
        if self.edge_obj_to_obj is None:
            self.edge_obj_to_obj = tensor_node
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
            self.edge_obj_to_obj = torch.cat([self.edge_obj_to_obj, tensor_node], dim=1)

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

    def get_wore_embed(self, obj_cls):
        word_embed = self.obj_cls_to_features[obj_cls].clone().detach()
        if self.GPU:
            word_embed = word_embed.cuda()
        return word_embed

    def _cat_feature_and_wore_embed(self, obj_cls, feature):
        feature = feature.clone().detach()
        if self.GPU:
            feature = feature.cuda()
        word_embed = self.get_wore_embed(obj_cls)
        feature = torch.cat([feature, word_embed])
        return feature


class HeteGraphData(GraphData):
    def __init__(self, obj_cls_name_to_features, GPU, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, **kwargs):
        super(HeteGraphData, self).__init__(obj_cls_name_to_features, GPU,
                                            x=None, edge_index=None, edge_attr=None, y=None,
                                            pos=None, normal=None, face=None, **kwargs)
        self.attributes = None

    def __inc__(self, key, value):
        increasing_funcs = ["edge_obj_to_obj"]
        if key in increasing_funcs:
            return len(self["x"])
        else:
            return 0

    def _append_unique_obj_index_to_attribute(self, feature, obj_cls):
        unique_obj_index = len(self.obj_cls_to_ind[obj_cls])
        unique_obj_index_tensor = torch.tensor([unique_obj_index], dtype=torch.float)
        feature = torch.cat([feature, unique_obj_index_tensor]).unsqueeze(0)
        if self.GPU:
            feature = feature.cuda()
        return feature, unique_obj_index

    def add_node(self, obj_cls, feature, obj_id=None):
        word_embed = self.get_wore_embed(obj_cls).unsqueeze(0)
        feature, unique_obj_index = self._append_unique_obj_index_to_attribute(feature, obj_cls)
        # init self.x
        if self.x is None:
            ind = 0
            self.x = word_embed
            self.attributes = feature
        else:
            ind = len(self.x)
            self.x = torch.cat([self.x, word_embed], dim=0)
            self.attributes = torch.cat([self.attributes, feature], dim=0)
        # Oracle
        if obj_id is None:
            # Not Oracle. Create new obj_id
            obj_id = str(obj_cls) + "_" + str(unique_obj_index)
        self.obj_id_to_ind[obj_id] = ind
        self.ind_to_obj_id[ind] = obj_id
        self.obj_cls_to_ind[obj_cls].append(ind)
        self.list_node_obj_cls.append(obj_cls)
        return ind

    def update_node(self, obj_cls, feature, obj_id):
        '''
        '''
        isFeatureChange = False
        ind = self.obj_id_to_ind[obj_id]
        if self.GPU:
            feature = feature.cuda()
        if not torch.equal(self.attributes[ind][:-1], feature):
            isFeatureChange = True
        self.attributes[ind][:-1] = feature
        return ind, isFeatureChange

    def update_relation(self, node_src, node_dst, relation):
        '''
        node_src : 320
        node_dst : 321
        relation : 1
        '''
        tensor_node = torch.tensor([[node_src], [node_dst]], dtype=torch.long).contiguous()
        if self.GPU:
            tensor_node = tensor_node.cuda()
        if self.edge_obj_to_obj is None:
            self.edge_obj_to_obj = tensor_node
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
            self.edge_obj_to_obj = torch.cat([self.edge_obj_to_obj, tensor_node], dim=1)

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


class SceneGraph(object):
    """docstring for SceneGraph"""

    def __init__(self, cfg):
        super(SceneGraph, self).__init__()
        #
        self.cfg = cfg
        self.isORACLE = cfg.SCENE_GRAPH.ORACLE
        self.GPU = cfg.SCENE_GRAPH.GPU
        self.GRAPH_RESULT_PATH = cfg.SCENE_GRAPH.GRAPH_RESULT_PATH
        # self.EMBED_CURRENT_STATE = self.cfg_semantic.SCENE_GRAPH.EMBED_CURRENT_STATE
        # self.EMBED_HISTORY_CHANGED_NODES = self.cfg_semantic.SCENE_GRAPH.EMBED_HISTORY_CHANGED_NODES
        # vision
        self.VISION_FEATURE_SIZE = cfg.SCENE_GRAPH.VISION_FEATURE_SIZE
        self.SAME_VISION_FEATURE_THRESHOLD = cfg.SCENE_GRAPH.SAME_VISION_FEATURE_THRESHOLD
        # graph
        self.obj_cls_name_to_features = self._get_obj_cls_name_to_features()
        self.init_graph_data()

    def init_graph_data(self):
        graphdata = getattr(
            importlib.import_module(
                'agents.semantic_graph.semantic_graph'),
            self.cfg.SCENE_GRAPH.GraphData
            )
        self.global_graph = graphdata(self.obj_cls_name_to_features, self.GPU)
        self.current_state_graph = graphdata(self.obj_cls_name_to_features, self.GPU)
        self.history_changed_nodes_graph = graphdata(self.obj_cls_name_to_features, self.GPU)

    def init_current_state_data(self):
        graphdata = getattr(
            importlib.import_module(
                'agents.semantic_graph.semantic_graph'),
            self.cfg.SCENE_GRAPH.GraphData
            )
        self.current_state_graph = graphdata(self.obj_cls_name_to_features, self.GPU)

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

    def get_history_changed_nodes_graph_data(self):
        return self.history_changed_nodes_graph

    def get_current_state_graph_data(self):
        return self.current_state_graph

    def analyze_graph(self, dict_ANALYZE_GRAPH):
        '''
        dict_ANALYZE_GRAPH = {
            "score": score.clone().detach().to('cpu'),
            "sort_nodes_index": sort_nodes_index.clone().detach().to('cpu'),
        }
        '''
        dict_objectIds_to_score = {}
        try:
            for nodes_index in dict_ANALYZE_GRAPH["sort_nodes_index"]:
                objectIds = self.global_graph.ind_to_obj_id[nodes_index]
                dict_objectIds_to_score[objectIds] = dict_ANALYZE_GRAPH["score"][nodes_index]
        except Exception as e:
            pass
        return dict_objectIds_to_score

    def save_graph_data(self):
        if not os.path.exists(self.GRAPH_RESULT_PATH):
            os.mkdir(self.GRAPH_RESULT_PATH)
        utils.save_graph_data(self.global_graph, self.GRAPH_RESULT_PATH)

    def compare_existing_node(self, obj_cls, feature, compare_feature_len=0):
        obj_id = None
        nodes = self.global_graph.x
        obj_cls_to_ind = self.global_graph.obj_cls_to_ind
        for suspect_node_ind in obj_cls_to_ind[obj_cls]:
            suspect_node_feature = nodes[suspect_node_ind]
            suspect_node_feature = suspect_node_feature[:compare_feature_len].clone().to('cpu')
            similarity = self.compare_features(feature, suspect_node_feature)[0,0]
            print("similarity: ", similarity)
            if similarity >= self.SAME_VISION_FEATURE_THRESHOLD:
                # node ind to obj_id
                obj_id = self.global_graph.ind_to_obj_id[suspect_node_ind]
                break
        return obj_id

    def compare_features(self, feature1, feature2):
        return cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))

    def add_oracle_local_graph_to_global_graph(self, img, target):
        self.init_current_state_data()
        tar_ids = target["objectIds"]
        obj_clses = target["labels"].numpy().astype(int)
        obj_attributes = target["attributes"]
        # other way to process (index 4 is out of bounds for dimension 0 with size 4)
        obj_relations = target["relation_labels"].numpy().astype(int)
        global_graph_current_frame_obj_cls_to_node_index = []
        current_state_graph_current_frame_obj_cls_to_node_index = []

        '''
        tar_ids : ['Fridge|-01.50|+00.00|-00.70', 'Bread|+00.40|+00.96|+00.11|BreadSliced_6', 'Mug|-01.28|+01.01|-01.64', 'Microwave|-01.23|+00.90|-01.68', 'Cabinet|-00.84|+00.47|-01.67', 'CounterTop|+00.23|+00.95|-02.00', 'Drawer|-00.82|+00.75|-01.69']
        obj_cls : tensor(24.), object class index
        '''
        for i, tar_id in enumerate(tar_ids):
            obj_cls, obj_attribute = \
                obj_clses[i], obj_attributes[i]
            isFindNode = self.global_graph.search_node(obj_id=tar_id)
            # check node
            if isFindNode:
                ind, isFeatureChange = self.global_graph.update_node(obj_cls, obj_attribute, tar_id)
                # EMBED_HISTORY_CHANGED_NODES
                if isFeatureChange:
                    self.history_changed_nodes_graph.add_node(obj_cls, obj_attribute, tar_id)
            else:
                ind = self.global_graph.add_node(obj_cls, obj_attribute, tar_id)
            # EMBED_CURRENT_STATE
            current_state_node_ind = self.current_state_graph.add_node(obj_cls, obj_attribute, tar_id)
            # [317, 318, 148, 149, 150, 151, 152, 319, 320, 321, 322, 154, 323, 155, 157, 158, 159, 160, 161, 324]
            global_graph_current_frame_obj_cls_to_node_index.append(ind)
            current_state_graph_current_frame_obj_cls_to_node_index.append(current_state_node_ind)
        # relation
        for obj_relation_triplet in obj_relations:
            src_obj_ind, dst_obj_ind, relation = obj_relation_triplet
            src_node_ind = global_graph_current_frame_obj_cls_to_node_index[src_obj_ind]
            dst_node_ind = global_graph_current_frame_obj_cls_to_node_index[dst_obj_ind]
            self.global_graph.update_relation(src_node_ind, dst_node_ind, relation)
        # relation
        # EMBED_CURRENT_STATE
        for obj_relation_triplet in obj_relations:
            src_obj_ind, dst_obj_ind, relation = obj_relation_triplet
            src_node_ind = current_state_graph_current_frame_obj_cls_to_node_index[src_obj_ind]
            dst_node_ind = current_state_graph_current_frame_obj_cls_to_node_index[dst_obj_ind]
            self.current_state_graph.update_relation(src_node_ind, dst_node_ind, relation)

    def add_local_graph_to_global_graph(self, img, sgg_results):
        self.init_current_state_data()
        current_frame_obj_cls_to_node_index = []
        obj_clses = sgg_results["labels"].numpy().astype(int)
        obj_features = sgg_results["features"]
        obj_attributes = sgg_results["attribute_logits"]
        obj_relations_idx_pairs = sgg_results["obj_relations_idx_pairs"].numpy().astype(int)
        obj_relations_scores = sgg_results["obj_relations_scores"]
        for i, obj_cls in enumerate(obj_clses):
            obj_feature = obj_features[i].reshape(-1)
            obj_attribute = obj_attributes[i].reshape(-1)
            feature = torch.cat([obj_feature, obj_attribute], dim=0)
            obj_id = self.compare_existing_node(obj_cls, feature, compare_feature_len=self.VISION_FEATURE_SIZE)
            if obj_id is not None:
                ind, isFeatureChange = self.global_graph.update_node(obj_cls, feature, obj_id)
            else:
                ind = self.global_graph.add_node(obj_cls, feature)
            current_frame_obj_cls_to_node_index.append(ind)

        for i, max_relation in enumerate(torch.argmax(obj_relations_scores, dim=1).numpy()):
            if max_relation == 1:
                src_obj_ind, dst_obj_ind = obj_relations_idx_pairs[i]
                src_node_ind = current_frame_obj_cls_to_node_index[src_obj_ind]
                dst_node_ind = current_frame_obj_cls_to_node_index[dst_obj_ind]
                print("src_node_ind {}, dst_node_ind {}, relation {}".format(src_node_ind, dst_node_ind, max_relation))
                self.global_graph.update_relation(src_node_ind, dst_node_ind, max_relation)


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
    for i in range(10):
        img, target, idx = alfred_dataset[i]
        dict_target = {
            "labels": target.extra_fields["labels"],
            "obj_relations": target.extra_fields["pred_labels"],
            "relation_labels": target.extra_fields["relation_labels"],
            "attributes": target.extra_fields["attributes"],
            "objectIds": target.extra_fields["objectIds"],
        }
        scenegraph.add_oracle_local_graph_to_global_graph(img, dict_target)
        scenegraph.save_graph_data()
