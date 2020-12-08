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
import sgg.parser_scene as parser_scene
from lib.scene_parser.rcnn.structures.bounding_box import BoxList


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

    def trans_meta_data_to_sgg(self, img, mask, color_to_object, data_obj_relation_attribute):
        masks, boxes, labels, objectIds, obj_relations, obj_relation_triplets, obj_attribute = \
            parser_scene.transfer_mask_semantic_to_bbox_label(
                mask, color_to_object, self.object_classes, data_obj_relation_attribute)

        if len(boxes) == 0:
            return None, None

        width, height = img.size
        target_raw = BoxList(boxes, (width, height), mode="xyxy")
        img, target = self.transforms(img, target_raw)
        target.add_field("labels", torch.from_numpy(labels).to(dtype=torch.float))
        target.add_field("pred_labels", torch.from_numpy(obj_relations).to(dtype=torch.float))
        target.add_field("relation_labels", torch.from_numpy(
            obj_relation_triplets).to(dtype=torch.float))
        target.add_field("attributes", torch.from_numpy(obj_attribute).to(dtype=torch.float))
        target.add_field("objectIds", objectIds)
        target = target.clip_to_image(remove_empty=False)

        return img, target

# data from gen/scripts/augment_sgg_trajectories.py


class AlfredDataset(Dataset):
    def __init__(self, cfg):
        self.root = cfg.ALFREDTEST.data_path
        self.cfg = cfg

        ''''''
        # load all image files, sorting them to
        # ensure that they are aligned
        self.get_data_files(self.root, balance_scenes=cfg.ALFREDTEST.balance_scenes)

        self.trans_meta_data = TransMetaData(cfg)

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

        img, target = self.trans_meta_data.trans_meta_data_to_sgg(
            img, mask, color_to_object, data_obj_relation_attribute)

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
