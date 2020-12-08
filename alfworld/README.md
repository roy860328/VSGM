# ALFWorld

## Command
textworld need to be download handly
```
pip install -r requirements.txt
apt install cmake g++ git make
pip install downward-faster_replan.zip
pip install TextWorld-handcoded_expert_integration.zip

export ALFRED_ROOT=/home/alfworld
export GRAPH_RCNN_ROOT=/home/graph-rcnn.pytorch/
python scripts/play_alfred_tw.py data/json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-14/trial_T20190908_231731_054988/ --domain data/alfred.pddl
```

### Train language
```
python dagger/train_dagger.py config/base_config.yaml
```

### Train vision
```
python dagger/train_vision_dagger.py config/vision_config.yaml
```

### INFOS
```
infos: {'admissible_commands': [['go to countertop 1', 'go to coffeemachine 1', 'go to cabinet 1', 'go to cabinet 2', 'go to cabinet 3', 'go to sink 1', 'go to cabinet 4', 'go to drawer 1', 'go to drawer 2', 'go to drawer 3', 'go to sinkbasin 1', 'go to cabinet 5', 'go to toaster 1', 'go to fridge 1', 'go to cabinet 6', 'go to cabinet 7', 'go to cabinet 8', 'go to microwave 1', 'go to cabinet 9', 'go to cabinet 10', 'go to cabinet 11', 'go to drawer 4', 'go to cabinet 12', 'go to stoveburner 1', 'go to drawer 5', 'go to stoveburner 2', 'inventory', 'look']], 'won': [False], 'goal_condition_success_rate': [0.0], 'extra.gamefile': ['../data/json_2.1.1/train/pick_and_place_simple-Lettuce-None-CounterTop-25/trial_T20190907_000040_764797'], 'expert_plan': [['go to fridge 1']]}
```

## visual semantic
agents/sgg/*
agents/semantic_graph/*

object_classes = "__background__" + objects
predicate_to_ind = "__background__" + relations
### Train Data
DATASET_SGG.md

### Train SGG
'''
cd graph-rcnn.pytorch/
python -m torch.distributed.launch --nproc_per_node=2 main.py --config-file configs/attribute.yaml
'''

### test semantic graph
```
cd $ALFRED_ROOT/agents/
python semantic_graph/semantic_graph.py config/semantic_graph_base.yaml config/semantic_graph.yaml
``` 

## Citations

**ALFWorld**
```
@inproceedings{ALFWorld20,
  title ={{ALFWorld: Aligning Text and Embodied
           Environments for Interactive Learning}},
  author={Mohit Shridhar and Xingdi Yuan and
          Marc-Alexandre C\^ot\'e and Yonatan Bisk and
          Adam Trischler and Matthew Hausknecht},
  booktitle = {arXiv},
  year = {2020},
  url = {https://arxiv.org/abs/2010.03768}
}
```  

**ALFRED**
```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```

**TextWorld**
```
@inproceedings{cote2018textworld,
  title={Textworld: A learning environment for text-based games},
  author={C{\^o}t{\'e}, Marc-Alexandre and K{\'a}d{\'a}r, {\'A}kos and Yuan, Xingdi and Kybartas, Ben and Barnes, Tavian and Fine, Emery and Moore, James and Hausknecht, Matthew and El Asri, Layla and Adada, Mahmoud and others},
  booktitle={Workshop on Computer Games},
  pages={41--75},
  year={2018},
  organization={Springer}
}
```
