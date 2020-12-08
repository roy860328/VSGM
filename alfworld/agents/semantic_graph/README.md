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


## Test Oracle Global Graph is OK
```
python semantic_graph/semantic_graph.py config/semantic_graph_base.yaml config/semantic_graph.yaml
```