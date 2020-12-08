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

_C.SCENE_GRAPH = CN()
_C.SCENE_GRAPH.ORACLE = True
_C.SCENE_GRAPH.OBJ_NAME_EMBEDDING = "word_embed/object_alfworld.csv"
_C.SCENE_GRAPH.MODEL = "GCN"
