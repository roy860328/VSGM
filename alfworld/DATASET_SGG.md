# SGG training data
https://github.com/jwyang/graph-rcnn.pytorch
```
pip install -r requirements.txt
apt-get update
apt-get install libglib2.0-0
apt-get install libsm6
cd lib/scene_parser/rcnn
python setup.py build develop
```

You have to change /graph-rcnn.pytorch/configs/attribute.yaml parameters
ROI_BOX_HEAD.NUM_CLASSES : len(gen.constants.OBJECTS_DETECTOR) + 1 (background)
ROI_RELATION_HEAD.NUM_CLASSES : 1 (if object has parentReceptacle) + 1 (background) + 2 (Assertion t >= 0 && t < n_classes failed, https://github.com/facebookresearch/maskrcnn-benchmark/pull/1214)
ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES : ? (visible, isToggled ...)

SGG train transformers would .convert("RGB")/255

'''
cd /home/alfworld
export ALFRED_ROOT=/home/alfworld
export GRAPH_RCNN_ROOT=/home/graph-rcnn.pytorch/

'''
## Test
### Check alfred detector data is ok 
'''
cd /home/alfworld/agents
python sgg/alfred_data_format.py --balance_scenes
'''
### Check oracle scene graph is ok
'''
cd /home/alfworld/agents
python semantic_graph/semantic_graph.py config/semantic_graph.yaml
'''

### preload ALFWORLD images, masks, meta, sgg_meta
/home/alfworld/agents/sgg/alfred_data_format.py
/home/alfworld/agents/sgg/parser_scene.py


## generate images, masks, meta, sgg_meta
```
python gen/scripts/augment_sgg_trajectories.py --data_path data/json_2.1.1/train --save_path detector/data/test
```


### meta
```
{"(95, 252, 121)": "Mug|+02.82|+00.79|-01.22", "(8, 94, 186)": "Mug", "(108, 125, 127)": "CD|-01.29|+00.68|-00.85", ...}
```

### sgg_meta
```
{
  "name": "Mug_2a940808(Clone)_copy_28",
  "position": {
    "x": 2.82123375,
    "y": 0.788715661,
    "z": -1.22375751
  },
  "rotation": {
    "x": 0.00006351283,
    "y": 0.000053627562,
    "z": -0.000171551466
  },
  "cameraHorizon": 0,
  "visible": false,
  "receptacle": true,
  "toggleable": false,
  "isToggled": false,
  "breakable": true,
  "isBroken": false,
  "canFillWithLiquid": true,
  "isFilledWithLiquid": false,
  "dirtyable": true,
  "isDirty": false,
  "canBeUsedUp": false,
  "isUsedUp": false,
  "cookable": false,
  "isCooked": false,
  "ObjectTemperature": "RoomTemp",
  "canChangeTempToHot": false,
  "canChangeTempToCold": false,
  "sliceable": false,
  "isSliced": false,
  "openable": false,
  "isOpen": false,
  "pickupable": true,
  "isPickedUp": false,
  "mass": 1,
  "salientMaterials": [
    "Ceramic"
  ],
  "receptacleObjectIds": [],
  "distance": 3.2283268,
  "objectType": "Mug",
  "objectId": "Mug|+02.82|+00.79|-01.22",
  "parentReceptacle": null,
  "parentReceptacles": [
    "Desk|+02.29|+00.01|-01.15"
  ],
  "currentTime": 0,
  "isMoving": true,
  "objectBounds": {
    "objectBoundsCorners": [
      {
        "x": 2.88107681,
        "y": 0.7887154,
        "z": -1.16639853
      },
      {
        "x": 2.736586,
        "y": 0.788715839,
        "z": -1.16639841
      },
      {
        "x": 2.736586,
        "y": 0.788715959,
        "z": -1.28111649
      },
      {
        "x": 2.88107681,
        "y": 0.788715541,
        "z": -1.2811166
      },
      {
        "x": 2.88107729,
        "y": 0.9005291,
        "z": -1.16639841
      },
      {
        "x": 2.73658657,
        "y": 0.9005295,
        "z": -1.16639829
      },
      {
        "x": 2.73658633,
        "y": 0.9005296,
        "z": -1.28111625
      },
      {
        "x": 2.881077,
        "y": 0.9005292,
        "z": -1.28111649
      }
    ]
  }
},
```