import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, os.environ['GRAPH_RCNN_ROOT'])
from lib.scene_parser.parser import SceneParser
from lib.scene_parser.parser import SceneParser
from lib.scene_parser.rcnn.utils.visualize import select_top_predictions, overlay_boxes, overlay_class_names


class SGG(SceneParser):
    def __init__(self, cfg, transforms, ind_to_classes, device):
        super(SGG, self).__init__(cfg)
        self.cfg = cfg
        self.transforms = transforms
        self.ind_to_classes = ind_to_classes
        self.device = device
        self.SAVE_SGG_RESULT = cfg.MODEL.SAVE_SGG_RESULT
        self.SAVE_SGG_RESULT_PATH = cfg.MODEL.SAVE_SGG_RESULT_PATH
        if self.SAVE_SGG_RESULT and not os.path.exists(self.SAVE_SGG_RESULT_PATH):
            os.mkdir(self.SAVE_SGG_RESULT_PATH)

    def predict(self, imgs, img_ids=0):
        '''
            imgs: torch.Size([n, 3, 800, 800])

            output
            detections
            extra_fields : 'labels', 'scores', 'logits', 'features', 'attribute_logits'
            # 'labels': tensor([17, 17, 32]), 'scores': tensor([0.2573, 0.1595, 0.1070]), 'logits': tensor([[ 9.7290e+00, -2.7969e+00,  2.7352e+00, -5.6947e-01, -2.8215e+00,

            detection_pairs
            extra_fields : 'idx_pairs', 'scores'
            # __background__ label = 0 
            # => then parentReceptacles label 1, col 0 bigger than col 1, they donot have realtion
            'scores': tensor([[0.9716, 0.0284],
                              [0.9784, 0.0216],
                              [0.9716, 0.0284],
                              [0.8863, 0.1137],
                              [0.9784, 0.0216],
                              [0.8863, 0.1137]],
            'idx_pairs': tensor([[0, 1],
                                [0, 2],
                                [1, 0],
                                [1, 2],
                                [2, 0],
                                [2, 1]], device='cuda:0'), 
            'scores': tensor([[0.9716, 0.0284],
                            [0.9784, 0.0216],
                            [0.9716, 0.0284],
                            [0.8863, 0.1137],
                            [0.9784, 0.0216],
                            [0.8863, 0.1137]], device='cuda:0')}}
            
            detection_attr
            extra_fields : 'labels', 'scores', 'logits', 'features', 'attribute_logits'
        '''
        with torch.no_grad():
            import pdb; pdb.set_trace()
            if type(imgs) != torch.Tensor:
                imgs = [Image.fromarray(imgs) for img in imgs]
                imgs = self.transforms(imgs)
            imgs = imgs.to(self.device)
            output = self.forward(imgs)
            detections, detection_pairs, detection_attrs = output
            detections = [o.to('cpu') for o in detections]
            detection_pairs = [o.to('cpu') for o in detection_pairs]
            detection_attrs = [o.to('cpu') for o in detection_attrs]
            # detections.bbox
            if self.SAVE_SGG_RESULT:
                self.save_detect_result(imgs, detections, img_ids)
            result = self.parser_sgg_result(detections, detection_pairs, detection_attrs)
        return result

    def parser_sgg_result(self, detections, detection_pairs, detection_attrs):
        '''
        detections[0]
        bbox: tensor([[372.5945, 659.5366, 517.6358, 792.8099],
                        [752.1789, 542.3512, 792.4908, 584.1647],
                        [  0.0000, 775.8300, 481.3463, 797.7077]])
        features: torch.Size([3, 2048, 1, 1])
        '''
        results = []
        for i in range(len(detections)):
            result = {
                "labels": detections[i].get_field("labels"),
                "features": detections[i].get_field("features"),
                "obj_relations_idx_pairs": detection_pairs[i].get_field("idx_pairs"),
                "obj_relations_scores": detection_pairs[i].get_field("scores"),
                "attribute_logits": detection_attrs[i].get_field("attribute_logits"),
            }
            results.append(result)
        return results

    def load(self):
        checkpoint = torch.load(self.cfg.MODEL.WEIGHT_IMG)
        model_para = checkpoint["model"]
        # for name, param in self.named_parameters():
        #     print(name)
        self.load_state_dict(model_para)

    def save_detect_result(self, imgs, detections, img_ids=0):
        # graph-rcnn visualize_detection
        for i, prediction in enumerate(detections):
            top_prediction = select_top_predictions(prediction)
            img = imgs[i].permute(1, 2, 0).contiguous().cpu().numpy() + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            ### RuntimeError: expected device cuda:0 but got device cpu
            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, self.ind_to_classes)
            cv2.imwrite(os.path.join(self.SAVE_SGG_RESULT_PATH, "detection_{}.jpg".format(img_ids)), result)


def load_pretrained_model(cfg, transforms, ind_to_classes, device):
    '''
    cfg = config['sgg_cfg']
    '''
    scene_parser = SGG(cfg, transforms, ind_to_classes, device)
    scene_parser.load()
    return scene_parser


if __name__ == '__main__':
    sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
    sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
    import modules.generic as generic
    import alfred_data_format
    from semantic_graph.semantic_graph import SceneGraph

    cfg = generic.load_config()
    '''
    semantic_cfg
    '''
    cfg_semantic = cfg['semantic_cfg']
    trans_MetaData = alfred_data_format.TransMetaData(cfg_semantic)
    scenegraph = SceneGraph(cfg_semantic)
    alfred_dataset = alfred_data_format.AlfredDataset(cfg_semantic)
    '''
    sgg_cfg
    '''
    cfg_sgg = cfg['sgg_cfg']
    detector = load_pretrained_model(cfg_sgg, trans_MetaData.transforms, alfred_dataset.ind_to_classes, 'cuda')
    detector.eval()
    detector.to(device='cuda')

    for i in range(10):
        img, target, idx = alfred_dataset[i]
        img = img.unsqueeze(0)
        sgg_results = detector.predict(img, idx)
        scenegraph.add_local_graph_to_global_graph(img, sgg_results[0])
