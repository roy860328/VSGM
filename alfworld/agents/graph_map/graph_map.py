import torch
from sys import platform
if platform != "win32":
    from torch_geometric.data import Data
else:
    import open3d as o3d
import sys
import os
import io
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json
from icecream import ic
import importlib
from PIL import Image
import imageio
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT']))
from agents.graph_map.utils_graph_map import *#intrinsic_from_fov, load_extrinsic, load_intrinsic, pixel_coord_np, grid, get_cam_coords


class BasicGraphMap(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        '''
        Camera Para
        '''
        self.K = intrinsic_from_fov(cfg.GRAPH_MAP.INTRINSIC_HEIGHT, cfg.GRAPH_MAP.INTRINSIC_WIDTH, cfg.GRAPH_MAP.INTRINSIC_FOV)
        self.PIXEL_COORDS = pixel_coord_np(cfg.GRAPH_MAP.INTRINSIC_WIDTH, cfg.GRAPH_MAP.INTRINSIC_HEIGHT)  # [3, npoints]
        '''
        GraphMap Para
        '''
        self.S = cfg.GRAPH_MAP.GRAPH_MAP_SIZE_S
        self.CLASSES = cfg.GRAPH_MAP.GRAPH_MAP_CLASSES
        self.V = cfg.GRAPH_MAP.GRID_COORDS_XY_RANGE_V
        self.R = cfg.GRAPH_MAP.GRID_MIN_SIZE_R
        self.SHIFT_COORDS_HALF_S_TO_MAP = self.S//2
        self.map = np.zeros([self.S, self.S, self.CLASSES]).astype(int)
        self.buffer_plt = []

    def reset_graph_map(self):
        self.map = np.zeros([self.S, self.S, self.CLASSES]).astype(int)
        '''
        visualize
        '''
        with imageio.get_writer('./graph_map_.gif', mode='I') as writer:
            for buf_file in self.buffer_plt:
                plt_img = np.array(Image.open(buf_file))
                writer.append_data(plt_img)
        self.buffer_plt = []

    def update_graph_map(self, depth_image, agent_meta, sgg_result):
        bboxs = sgg_result["bbox"]
        labels = sgg_result["labels"]
        cam_coords = get_cam_coords(
            depth_image,
            agent_meta,
            bboxs, labels,
            self.K, self.PIXEL_COORDS)
        self.put_label_to_map(cam_coords)
        return cam_coords

    def put_label_to_map(self, cam_coords):
        x, y, z, labels = cam_coords
        x = np.round(x / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        z = np.round(z / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        labels = labels.astype(int)
        self.map[x, z, labels] = labels

    def visualize_graph_map(self, KEEP_DISPLAY=False):
        colors = cm.rainbow(np.linspace(0, 1, self.CLASSES))
        Is, Js, Ks = np.where(self.map != 0)
        label_color = []
        for i, j, k in zip(Is, Js, Ks):
            label_color.append(colors[self.map[i, j, k]])
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [plt.close() if event.key == 'escape' else None])
        plt.scatter(Is, Js, s=70, c=label_color, cmap="Set2")
        plt.plot(self.S//2, self.S//2, "ob")
        plt.gca().set_xticks(np.arange(0, self.S, 1))
        plt.gca().set_yticks(np.arange(0, self.S, 1))
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        self.buffer_plt.append(buf)
        # import pdb; pdb.set_trace()
        if KEEP_DISPLAY:
            plt.show()
        else:
            plt.pause(1.0)


class GraphMap(BasicGraphMap):
    def __init__(self, cfg, priori_features, dim_rgb_feature, device="cuda"):
        '''
        priori_features: dict. priori_obj_cls_name_to_features, rgb_features, attributes
        '''
        super().__init__(cfg)
        '''
        Graph Type
        '''
        self.device = device
        self.priori_features = priori_features
        self.GPU = cfg.SCENE_GRAPH.GPU
        self.dim_rgb_feature = dim_rgb_feature
        self.graphdata_type = getattr(
            importlib.import_module(
                'agents.semantic_graph.semantic_graph'),
            self.cfg.SCENE_GRAPH.GraphData
            )
        self._set_label_to_features()
        '''
        CLASSES
        '''
        self.CLASSES = len(self.label_to_features)
        print("self.CLASSES = cfg.GRAPH_MAP.GRAPH_MAP_CLASSES would not be use")

        self.init_graph_map()

    def _set_label_to_features(self):
        '''
        word & rgb & attributes features
        '''
        features = []
        attributes = []
        # background
        features.append(
            torch.zeros([self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE + self.cfg.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE]))
        attributes.append(
            torch.zeros([self.cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE]))
        # objects
        for k, word_feature in self.priori_features["priori_obj_cls_name_to_features"].items():
            rgb_feature = torch.tensor(self.priori_features["rgb_features"][str(k)]).float()
            feature = torch.cat([word_feature, rgb_feature])
            # [0] is _append_unique_obj_index_to_attribute
            attribute = torch.tensor(self.priori_features["attributes"][str(k)] + [0]).float()
            features.append(feature)
            attributes.append(attribute)
        self.label_to_features = torch.stack(features).to(device=self.device, dtype=torch.float)
        self.label_to_attributes = torch.stack(attributes).to(device=self.device, dtype=torch.float)
        assert len(self.label_to_features) == len(self.priori_features["rgb_features"].keys()), "len diff error"
        assert self.label_to_attributes.shape[-1] == self.cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE, "len diff error"

    def init_graph_map(self):
        self.map = self.graphdata_type(
            self.priori_features["priori_obj_cls_name_to_features"],
            self.GPU,
            self.dim_rgb_feature,
            device=self.device
            )
        '''
        Create graph map node space
        '''
        feature_size = self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE + self.cfg.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE
        attribute_size = self.cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE
        self.map.x = torch.zeros([self.S * self.S * self.CLASSES, feature_size], device=self.device, dtype=torch.float)
        self.map.attributes = torch.zeros([self.S * self.S * self.CLASSES, attribute_size], device=self.device, dtype=torch.float)
        for x in range(self.S):
            for z in range(self.S):
                for label in range(self.CLASSES):
                    target_node_index = x + self.S * z + self.S * self.S * label
                    self.map.x[target_node_index] = self.label_to_features[label]
                    self.map.attributes[target_node_index] = self.label_to_attributes[label]
        self.map.activate_nodes = list(range(self.S * self.S))
        '''
        graph map node relation
        '''
        edges = []
        '''
        most top grid connect together
        would be square grid
        '''
        for x in range(self.S):
            for z in range(self.S):
                if x < self.S-1 and z < self.S-1:
                    top_map = x + self.S * z
                    # right edges
                    edge = torch.tensor(
                        [top_map, top_map+1],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
                    edge = torch.tensor(
                        [top_map+1, top_map],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
                    edge = torch.tensor(
                        [top_map, top_map+self.S * (z+1)],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
                    edge = torch.tensor(
                        [top_map+self.S * (z+1), top_map],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
        '''
        depth node connect to top grid node
        '''
        for x in range(self.S):
            for z in range(self.S):
                '''
                layer
                '''
                top_map = x + self.S * z
                for label in range(1, self.CLASSES):
                    src = top_map + self.S * self.S * label
                    dst = top_map
                    edge = torch.tensor([src, dst], device=self.device, dtype=torch.long).contiguous()
                    edges.append(edge)
        self.map.edge_obj_to_obj = torch.stack(edges).reshape(2, -1)

    def reset_graph_map(self):
        self.map.activate_nodes = list(range(self.S * self.S))
        '''
        visualize
        '''
        with imageio.get_writer('./graph_map.gif', mode='I') as writer:
            for buf_file in self.buffer_plt:
                plt_img = np.array(Image.open(buf_file))
                writer.append_data(plt_img)
        self.buffer_plt = []

    def put_label_to_map(self, cam_coords):
        max_index = self.S-1
        x, y, z, labels = cam_coords
        x = np.round(x / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        x[x > max_index] = max_index
        z = np.round(z / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        z[z > max_index] = max_index
        labels = labels.astype(int)
        coors = x + self.S * z + self.S * self.S * labels
        activate_node = np.unique(coors).tolist()
        self.map.activate_nodes.extend(activate_node)
        self.map.activate_nodes = list(set(self.map.activate_nodes))
        # ic(self.map.activate_nodes)
        # ic(len(self.map.activate_nodes))

    def visualize_graph_map(self, KEEP_DISPLAY=False):
        print("visualize_graph_map not implement")

'''
test
'''
def test_load_img(path, list_img_traj, type_image=".png"):
    def _load_with_path():
        frames_depth = None
        low_idx = -1
        for i, dict_frame in enumerate(list_img_traj):
            # 60 actions need 61 frames
            if low_idx != dict_frame["low_idx"]:
                low_idx = dict_frame["low_idx"]
            else:
                continue
            name_frame = dict_frame["image_name"].split(".")[0]
            frame_path = os.path.join(path, name_frame + type_image)
            if os.path.isfile(frame_path):
                img_depth = Image.open(frame_path).convert("RGB")
            else:
                print("file is not exist: {}".format(frame_path))
            img_depth = torch.tensor(np.array(img_depth))
            img_depth = img_depth.unsqueeze(0)

            if frames_depth is None:
                frames_depth = img_depth
            else:
                frames_depth = torch.cat([frames_depth, img_depth], dim=0)
        frames_depth = torch.cat([frames_depth, img_depth], dim=0)
        return frames_depth
    frames_depth = _load_with_path()
    return frames_depth


def test_load_meta_data(root, list_img_traj):
    def agent_sequences_to_one(META_DATA_FILE="meta_agent.json",
                               SGG_META="agent_meta",
                               EXPLORATION_META="exploration_agent_meta",
                               len_meta_data=-1):
        # load
        # print("_load with path", root)
        all_meta_data = {
            "agent_sgg_meta_data": [],
            "exploration_agent_sgg_meta_data": [],
        }
        low_idx = -1
        for i, dict_frame in enumerate(list_img_traj):
            # 60 actions need 61 frames
            if low_idx != dict_frame["low_idx"]:
                low_idx = dict_frame["low_idx"]
            else:
                continue
            name_frame = dict_frame["image_name"].split(".")[0]
            file_path = os.path.join(root, SGG_META, name_frame + ".json")
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    meta_data = json.load(f)
                all_meta_data["agent_sgg_meta_data"].append(meta_data)
            else:
                print("file is not exist: {}".format(file_path))
        all_meta_data["agent_sgg_meta_data"].append(meta_data)
        n_meta_gap = len(all_meta_data["agent_sgg_meta_data"])-len_meta_data
        for _ in range(n_meta_gap):
            print("{}.gap num {}".format(root, n_meta_gap))
            all_meta_data["agent_sgg_meta_data"].append(meta_data)
        exploration_path = os.path.join(root, EXPLORATION_META, "*.json")
        exploration_file_paths = glob.glob(exploration_path)
        for exploration_file_path in exploration_file_paths:
            with open(exploration_file_path, 'r') as f:
                meta_data = json.load(f)
            all_meta_data["exploration_agent_sgg_meta_data"].append(meta_data)
        return all_meta_data

    agent_meta_data = agent_sequences_to_one(
        META_DATA_FILE="meta_agent.json",
        SGG_META="agent_meta",
        EXPLORATION_META="exploration_agent_meta")
    return agent_meta_data


if __name__ == '__main__':
    import yaml
    import glob
    import json
    sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
    sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
    from agents.sgg import alfred_data_format
    from agents.semantic_graph.semantic_graph import SceneGraph
    from config import cfg
    if sys.platform == "win32":
        root = r"D:\cvml_project\projections\inverse_projection\data\d2\trial_T20190909_100908_040512\\"
        semantic_config_file = r"D:\alfred\alfred\models\config\sgg_without_oracle.yaml"
    else:
        root = r"/home/alfred/data/full_2.1.0/train/pick_and_place_simple-RemoteControl-None-Ottoman-208/trial_T20190909_100908_040512/"
        semantic_config_file = "/home/alfred/models/config/sgg_without_oracle.yaml"

    config = cfg
    config.merge_from_file(semantic_config_file)
    alfred_dataset = alfred_data_format.AlfredDataset(config)
    if sys.platform == "win32":
        grap_map = BasicGraphMap(config)
    else:
        scene_graph = SceneGraph(
            config,
            alfred_dataset.trans_meta_data.SGG_result_ind_to_classes,
            config.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE,
            "cuda",
            )
        grap_map = GraphMap(
            config,
            scene_graph.priori_features,
            config.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE,
            "cuda",
            )

    traj_data_path = root + "traj_data.json"
    with open(traj_data_path, 'r') as f:
        traj_data = json.load(f)
    frames_depth = test_load_img(os.path.join(root, 'depth_images'), traj_data["images"]).view(-1, 300, 300, 3)
    agent_meta_data = test_load_meta_data(root, traj_data["images"])
    cat_cam_coords = np.array([[], [], [], []])
    for i in range(10):
        depth_image = frames_depth[i]
        agent_meta = agent_meta_data['agent_sgg_meta_data'][i]
        img, target, idx, rgb_img = alfred_dataset[i]
        bbox = target.bbox
        bbox[bbox>=300] = 299
        # import pdb; pdb.set_trace()
        target = {
            "bbox": bbox,
            "labels": target.get_field("labels"),
        }
        cam_coords = grap_map.update_graph_map(
            np.array(depth_image),
            agent_meta,
            target)
        grap_map.visualize_graph_map()
        cat_cam_coords = np.concatenate([cat_cam_coords, cam_coords], axis=1)

    grap_map.visualize_graph_map(KEEP_DISPLAY=True)
    grap_map.reset_graph_map()

    # Visualize
    if platform == "win32":
        pcd_cam = o3d.geometry.PointCloud()
        pcd_cam.points = o3d.utility.Vector3dVector(cat_cam_coords.T[:, :3])
        # Flip it, otherwise the pointcloud will be upside down
        pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd_cam])

        # Do top view projection
        # project_topview(cam_coords)
        grid(cat_cam_coords, KEEP_DISPLAY=True)