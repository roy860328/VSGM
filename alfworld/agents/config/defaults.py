import os
from yacs.config import CfgNode as CN

_C = CN()

_C.ALFREDTEST = CN()
_C.ALFREDTEST.data_path = "/home/alfworld/detector/data/test/"
_C.ALFREDTEST.object_types = "all"
_C.ALFREDTEST.height = 400
_C.ALFREDTEST.weight = 400
_C.ALFREDTEST.balance_scenes = True
_C.ALFREDTEST.kitchen_factor = 1.
_C.ALFREDTEST.living_factor = 1.
_C.ALFREDTEST.bedroom_factor = 1.
_C.ALFREDTEST.bathroom_factor = 1.

_C.INPUT = CN()
_C.INPUT.MIN_SIZE_TRAIN = (800,)   # Size of the smallest side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1024     # Maximum size of the side of the image during training
_C.INPUT.MIN_SIZE_TEST = 800       # Size of the smallest side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1024      # Maximum size of the side of the image during testing
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]  # Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]  # Values to be used for image normalization
_C.INPUT.TO_BGR255 = True          # Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.BRIGHTNESS = 0.0          # Image ColorJitter
_C.INPUT.CONTRAST = 0.0            # Image ColorJitter
_C.INPUT.SATURATION = 0.0          # Image ColorJitter
_C.INPUT.HUE = 0.0                 # Image ColorJitter
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0


_C.GENERAL = CN()
_C.GENERAL.LOAD_PRETRAINED = False
_C.GENERAL.LOAD_PRETRAINED_PATH = ""


_C.SCENE_GRAPH = CN()
_C.SCENE_GRAPH.GPU = True
_C.SCENE_GRAPH.ORACLE = True
_C.SCENE_GRAPH.OBJ_NAME_EMBEDDING = "word_embed/object_alfworld.csv"
_C.SCENE_GRAPH.SAME_VISION_FEATURE_THRESHOLD = 0.7
_C.SCENE_GRAPH.VISION_FEATURE_SIZE = 2048
_C.SCENE_GRAPH.NODE_FEATURE_SIZE = 323
_C.SCENE_GRAPH.NODE_MIDDEL_FEATURE_SIZE = 32
_C.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE = 16
_C.SCENE_GRAPH.MODEL = "gcn"
_C.SCENE_GRAPH.NORMALIZATION = True
_C.SCENE_GRAPH.EMBED_TYPE = "GraphEmbed"
_C.SCENE_GRAPH.EMBED_FEATURE_SIZE = 128