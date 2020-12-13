import os
import sys
import numpy as np
import torch
import random
import json
from PIL import Image
from torch.utils.data import Dataset
sys.path.insert(0, os.environ['ALFRED_ROOT'])
sys.path.insert(0, os.path.join(os.environ['ALFRED_ROOT'], 'agents'))
sys.path.insert(0, os.environ["GRAPH_RCNN_ROOT"])
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
import sgg.parser_scene as parser_scene


# Transform
class TransMetaData():

    def __init__(self, cfg):
        super(TransMetaData, self).__init__()
        self.cfg = cfg
        self.transforms = parser_scene.get_transform(cfg, train=True)
        # para
        self.object_classes = parser_scene.get_object_classes(cfg.ALFREDTEST.object_types)
        self.object_classes.insert(0, '__background__')
        ''' graph-rcnn need '''
        self.class_to_ind = parser_scene.get_dict_class_to_ind(self.object_classes)
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                                     self.class_to_ind[k])
        # cfg.ind_to_class = self.ind_to_classes
        self.predicate_to_ind = parser_scene.get_dict_predicate_to_ind()
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                        self.predicate_to_ind[k])


    def trans_object_meta_data_to_relation_and_attribute(self, data_obj_relation_attribute, boxes_id=None, boxes_labels=None):
        if boxes_id is None:
            boxes_id, boxes_labels = [], []
            for obj_relation_attribute in data_obj_relation_attribute:
                if obj_relation_attribute["visible"] == True and obj_relation_attribute["objectType"] in self.object_classes:
                    boxes_id.append(obj_relation_attribute["objectId"])
                    class_idx = self.object_classes.index(obj_relation_attribute["objectType"])
                    boxes_labels.append(class_idx)
        obj_relations, obj_relation_triplets, obj_attribute = parser_scene.transfer_object_meta_data_to_relation_and_attribute(
            boxes_id, data_obj_relation_attribute)
        '''
        {
         'pred_labels': array([[0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 1., 0., 0., 1.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [1., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.]]), 
         'relation_labels': array([[6, 0, 1],
                                   [2, 1, 1],
                                   [2, 4, 1],
                                   [2, 7, 1]]), 
         'attributes': array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                               0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                               0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0]]), 
         'objectIds': ['Spatula|+03.07|+00.76|-00.03', 'ButterKnife|+02.97|+00.90|-00.70', 'CounterTop|+03.11|+00.94|+00.02', 'Sink|+03.08|+00.89|+00.09', 'HousePlant|+03.20|+00.89|-00.37', 'Window|+03.70|+01.68|+00.05', 'Sink|+03.08|+00.89|+00.09|SinkBasin', 'Faucet|+03.31|+00.89|+00.09'], 'labels': array(['Spatula', 'ButterKnife', 'CounterTop', 'Sink', 'HousePlant',
                       'Window', 'SinkBasin', 'Faucet'], dtype='<U11')}

        '''
        dict_obj_rel_attr = {
            "pred_labels": obj_relations,
            "relation_labels": torch.from_numpy(obj_relation_triplets).to(dtype=torch.float),
            "attributes": torch.from_numpy(obj_attribute).to(dtype=torch.float),
            "objectIds": boxes_id,
            "labels": torch.tensor(boxes_labels).to(dtype=torch.float),
        }
        return dict_obj_rel_attr

    def trans_meta_data_to_sgg(self, img, mask, color_to_object, data_obj_relation_attribute):
        masks, boxes, boxes_labels, boxes_id = parser_scene.transfer_mask_semantic_to_bbox_label(
            mask, color_to_object, self.object_classes, data_obj_relation_attribute)
        dict_obj_rel_attr = self.trans_object_meta_data_to_relation_and_attribute(
            data_obj_relation_attribute, boxes_id, boxes_labels)
        sgg_data = dict_obj_rel_attr
        sgg_data["boxes"] = boxes
        sgg_data["objectIds"] = boxes_id
        return sgg_data


'''
data from gen/scripts/augment_sgg_trajectories.py
'''


class AlfredDataset(Dataset):
    def __init__(self, cfg):
        self.root = cfg.ALFREDTEST.data_path
        self.cfg = cfg

        ''''''
        # load all image files, sorting them to
        # ensure that they are aligned
        self.get_data_files(self.root, balance_scenes=cfg.ALFREDTEST.balance_scenes)

        self.trans_meta_data = TransMetaData(cfg)
        self.ind_to_classes = self.trans_meta_data.ind_to_classes
        self.object_classes = self.trans_meta_data.object_classes
        self.class_to_ind = self.trans_meta_data.class_to_ind
        self.ind_to_classes = self.trans_meta_data.ind_to_classes
        # cfg.ind_to_class = self.ind_to_classes
        self.predicate_to_ind = self.trans_meta_data.predicate_to_ind
        self.ind_to_predicates = self.trans_meta_data.ind_to_predicates

    def get_data_files(self, root, balance_scenes=False):
        if balance_scenes:
            kitchen_path = os.path.join(root, 'kitchen', 'images')
            living_path = os.path.join(root, 'living', 'images')
            bedroom_path = os.path.join(root, 'bedroom', 'images')
            bathroom_path = os.path.join(root, 'bathroom', 'images')

            kitchen = list(sorted(os.listdir(kitchen_path)))
            living = list(sorted(os.listdir(living_path)))
            bedroom = list(sorted(os.listdir(bedroom_path)))
            bathroom = list(sorted(os.listdir(bathroom_path)))

            min_size = min(len(kitchen), len(living), len(bedroom), len(bathroom))
            kitchen = [os.path.join(kitchen_path, f) for f in random.sample(
                kitchen, int(min_size*self.cfg.ALFREDTEST.kitchen_factor))]
            living = [os.path.join(living_path, f) for f in random.sample(
                living, int(min_size*self.cfg.ALFREDTEST.living_factor))]
            bedroom = [os.path.join(bedroom_path, f) for f in random.sample(
                bedroom, int(min_size*self.cfg.ALFREDTEST.bedroom_factor))]
            bathroom = [os.path.join(bathroom_path, f) for f in random.sample(
                bathroom, int(min_size*self.cfg.ALFREDTEST.bathroom_factor))]

            self.imgs = kitchen + living + bedroom + bathroom
            self.masks = [f.replace("images", "masks") for f in self.imgs]
            self.metas = [f.replace("images", "meta").replace(".png", ".json") for f in self.imgs]
            self.sgg_metas = [f.replace("images", "sgg_meta").replace(".png", ".json")
                              for f in self.imgs]
        else:
            self.imgs = [os.path.join(root, "images", f)
                         for f in list(sorted(os.listdir(os.path.join(root, "images"))))]
            self.masks = [os.path.join(root, "masks", f)
                          for f in list(sorted(os.listdir(os.path.join(root, "masks"))))]
            self.metas = [os.path.join(root, "meta", f)
                          for f in list(sorted(os.listdir(os.path.join(root, "meta"))))]
            self.sgg_metas = [os.path.join(root, "sgg_meta", f)
                              for f in list(sorted(os.listdir(os.path.join(root, "sgg_meta"))))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        meta_path = self.metas[idx]
        sgg_meta_path = self.sgg_metas[idx]

        # print("Opening: %s" % (self.imgs[idx]))

        with open(meta_path, 'r') as f:
            color_to_object = json.load(f)
        with open(sgg_meta_path, 'r') as f:
            data_obj_relation_attribute = json.load(f)

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        #
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)

        sgg_data = self.trans_meta_data.trans_meta_data_to_sgg(
            img, mask, color_to_object, data_obj_relation_attribute)
        boxes = sgg_data["boxes"]
        labels = sgg_data["labels"]
        obj_relations = sgg_data["pred_labels"]
        relation_labels = sgg_data["relation_labels"]
        attributes = sgg_data["attributes"]
        objectIds = sgg_data["objectIds"]

        if len(boxes) == 0:
            return None, None

        width, height = img.size
        target_raw = BoxList(boxes, (width, height), mode="xyxy")
        img, target = self.trans_meta_data.transforms(img, target_raw)
        target.add_field("labels", labels)
        target.add_field("pred_labels", torch.from_numpy(obj_relations).to(dtype=torch.float))
        target.add_field("relation_labels", relation_labels)
        target.add_field("attributes", attributes)
        target.add_field("objectIds", objectIds)
        target = target.clip_to_image(remove_empty=False)

        return img, target, idx

    def get_img_info(self, img_id):
        # w, h = self.im_sizes[img_id, :]
        return {"height": self.cfg.ALFREDTEST.height, "width": self.cfg.ALFREDTEST.weight}

    def map_class_id_to_class_name(self, class_id):
        return self.trans_meta_data.ind_to_classes[class_id]


def main(cfg):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = len(parser_scene.get_object_classes(cfg.object_types))+1
    # use our dataset and defined transformations
    dataset = AlfredDataset(cfg)
    for i in range(100):
        print(dataset[i])

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    indices = list(range(len(dataset)))
    dataset = torch.utils.data.Subset(dataset, indices[:-4000])

    # define training and validation data loaders
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4,
    #     collate_fn=utils.collate_fn)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
    #     collate_fn=utils.collate_fn)


def get_dataset(cfg, transform=None):
    dataset = AlfredDataset(cfg)
    return dataset


if __name__ == "__main__":
    import modules.generic as generic
    cfg = generic.load_config()
    main(cfg)
