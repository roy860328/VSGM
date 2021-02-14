# ALFWorld
## requirements
https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
download whl handly and install
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-geometric
```

## Word embedding 
For semantic_graph object class name to fasttext word embedding

### Get Object feature '.csv'
change graph_analysis/data/objects.txt
```
cd graph_analysis
python create_dgl_data.py --object
```
You will create csv file at 'graph_analysis/data_dgl/object_alfworld.csv'

## Get rgb_feature feature data
```
python sgg/alfred_data_format.py config/test_base.yaml --semantic_config_file config/semantic_graph.yaml --sgg_config_file sgg/graph-rcnn.pytorch/configs/attribute.yaml --not_save_config
```


## Test Memory Graph is OK
```
python semantic_graph/semantic_graph.py config/semantic_graph_base.yaml --semantic_config_file config/semantic_graph.yaml --not_save_config
```

- obj_cls_name_to_features
	- obj_cls_name would +1
	- cause
		- '__background__' = 0