# Prerequire
```
conda install -c menpo opencv
# https://github.com/dmlc/dgl/tree/master
conda install -c dglteam dgl-cuda9.0
pip install rdflib==4.2.2
pip install fasttext
pip install pytorch-nlp
pip install imageio
pip install icecream
pip install imageio-ffmpeg

from icecream import ic
ic(1)

```


# Run Model

## ALFRED_ROOT
```
cd gcn/alfred
# linux
export ALFRED_ROOT=/home/alfred/
export ALFWORLD_ROOT=/home/alfworld/
export GRAPH_ANALYSIS=/home/graph_analysis/
export GRAPH_RCNN_ROOT=/home/graph-rcnn.pytorch/

# windows
SET ALFRED_ROOT=D:\alfred\alfred
SET ALFWORLD_ROOT=D:\alfred\alfworld
SET GRAPH_ANALYSIS=D:\alfred\graph_analysis
SET GRAPH_RCNN_ROOT=D:\alfred\alfworld\agents\sgg\graph-rcnn.pytorch
```

## MOCA pre-download eval maskrcnn model
```
cd $ALFRED_ROOT
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt
```

## generate Semantic graph data
```
cd alfred/gen
python scripts/augment_meta_data_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
```
### extract exploration img to resnet feature
```
python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 32 --gpu --visual_model resnet18 --filename feat_exploration_conv.pt --img_folder exploration_meta
```

## MOCA

```
export ALFRED_ROOT=/home/moca/
cd /home/moca
CUDA_VISIBLE_DEVICES=1 python models/train/train_seq2seq.py --model seq2seq_im_mask --dout exp/moca_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --gpu
# eval
python models/eval/eval_seq2seq.py --model models.model.seq2seq_im_mask --model_path exp/moca_seq2seq_im_mask,name,pm_and_subgoals_01/best_seen.pth --eval_split valid_seen --gpu --num_threads 2
```


## MOCA + Semantic
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_semantic --dout exp/moca_memory{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 20 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --gpu

CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --model_path exp/moca_memoryseq2seq_im_moca_semantic,name,pm_and_subgoals_01_12-01-2021_03-37-50/best_seen.pth --model seq2seq_im_moca_semantic --data data/full_2.1.0/ --eval_split train --gpu
```

## MOCA + Importent nodes & rgb node feature
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/importent_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/importent_rgb_nodes_DynamicNode_moca --splits data/splits/oct21.json --batch 20 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --demb 100 --dhid 256 --gpu

CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/importent_semantic_graph.yaml --model_path exp/importent_nodes_moca_16-01-2021_11-02-31/best_seen.pth --model seq2seq_im_moca_importent_nodes --data data/full_2.1.0/ --eval_split train --gpu
```

## MOCA + Priori
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/priori_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/priori_moca --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --gpu --not_save_config

```

## MOCA + Priori + Graph attention
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/gan_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/graph_attention --splits data/splits/oct21.json --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu
```

---
---

# Decompose
## define
action_navi_low = ['<<pad>>', '<<seg>>', '<<stop>>', 'LookDown_15', 'MoveAhead_25', 'RotateLeft_90', 'LookUp_15', 'RotateRight_90']
action_operation_low = ['PickupObject', 'SliceObject', 'OpenObject', 'PutObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff']

1.index_to_word => [0-X]
action_navi_low_dict_word_to_index => {k:0, k1:1 ...}
action_operation_low_dict_word_to_index => {k:8, k1:8+1 ...}

2.old_action_low_index_to_navi_or_operation: redefine embed value for train
out_action_navi_or_operation: navi = 0, operation = 1,
ignore_index=0 for 'action_navi_low', 'action_operation_low' ...
ignore_index=-100 for 'action_navi_or_operation'

3.
```
feat['action_low'] # new action low index. for training data (gold)
feat['action_navi_low'] # for training data loss
feat['action_operation_low'] # for training data loss
feat['action_navi_or_operation'] # for training data loss
F.cross_entropy(, reduction='none', ignore_index=0)
```

## Decompose + Adj Relation + Priori + MOCA 
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/decompose_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_decomposed --dout exp/adj_relation_decomposed_priori --splits data/splits/oct21.json --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --action_navi_loss_wt 0.8 --action_oper_loss_wt 1 --action_navi_or_oper_loss_wt 1 --mask_loss_wt 1 --mask_label_loss_wt 1
# eval
CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/decompose_semantic_graph.yaml --model_path exp/fast_epoch_adj_relation_decomposed_priori_24-01-2021_05-45-58/best_seen.pth --model seq2seq_im_decomposed --data data/full_2.1.0/ --eval_split train --gpu
```
'action_low', 'action_navi_low', 'action_operation_low', 'action_navi_or_operation' need pad value = 0

## Decompose2 + fast train
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/fast_epoch_base.yaml --semantic_config_file models/config/decompose_semantic_graph2.yaml --data data/full_2.1.0/ --model seq2seq_im_decomposed --dout exp/fast_decomposed2 --splits data/splits/oct21.json --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --action_navi_loss_wt 0.8 --action_oper_loss_wt 1 --action_navi_or_oper_loss_wt 1 --mask_loss_wt 1 --mask_label_loss_wt 1
```

## eval sub goal
```
CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/fast_epoch_base.yaml --semantic_config_file models/config/decompose_semantic_graph.yaml --model_path exp/fast_decomposed2_26-01-2021_12-22-10/best_seen.pth --model seq2seq_im_decomposed --data data/full_2.1.0/ --eval_split train --gpu --subgoals GotoLocation,PickupObject
```

---
---
---

# Alfred baseline

## Seq2Seq
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model seq2seq_im_mask --dout exp/model,{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```

## GCN
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
python models/train/train_seq2seq.py --data data/json_feat_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 
```
--data/json_feat_2.1.0/

gcn visaul embedding
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --gcn_cat_visaul --gpu_id 1
```

## Heterograph GCN
```
python models/train/train_graph.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,heterograph_{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph
python models/train/train_graph.py --data data/json_feat_2.1.0/ --model gcn_im --dout exp/model,heterograph_{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph
```

### depth image
- You must generate depth image first

#### Gen depth image
```
cd alfred/gen
python scripts/augment_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
```
#### check current generate state
```
python scripts/check_augment_state.py --data_path ../data/full_2.1.0/
```
#### Run model
```
cd alfred
python models/train/train_graph.py --data data/full_2.1.0/ --model gcn_depth_im --dout exp/model,heterograph_depth_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --gpu_id 1
```
### HetG + depth + graph attention
```
python models/train/train_graph.py --data data/full_2.1.0/ --model gcn_depth_im --dout exp/model,heterograph_depth_attention_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg
# eval
python models/eval/eval_graph.py --model_path exp/model,heterograph_oratls_depth_attention_gcn_depth_im,name,pm_and_subgoals_01_19-10-2020_04-46-24/latest.pth --model models.model.gcn_depth_im --data data/full_2.1.0/ --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg --eval_split train
```

### HetG + graph attention
```
python models/train/train_graph.py --data data/full_2.1.0/ --model HeteG_noDepth_im --dout exp/model,heterograph_noDepth_attention_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg --gpu_id 0
# eval
python models/eval/eval_graph.py --model_path exp/model,heterograph_noDepth_attention_HeteG_noDepth_im,name,pm_and_subgoals_01_21-10-2020_05-24-02/latest.pth --model models.model.HeteG_noDepth_im --data data/full_2.1.0/ --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg --eval_split train
```

### HetG + graph attention + fasttext
```
python models/train/train_graph.py --data data/full_2.1.0/ --model fast_embedding_im --dout exp/model,heterograph__attention_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 1024 --demb 300 --HetLowSg --gpu_id 0
```

## Contrastive
- data
	- [[p1, p2, c4], [c0, h3, h1] ...]
- batch = 2
	- [p1, p2, c4, c0, h3, h1]

### Pretain HetG + graph attention + fasttext + contrastive
```
python models/train/train_parallel.py --data data/full_2.1.0/ --model contrastive_pretrain_im --dout exp/model,SimCLR_{model},heterograph__attention_,name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 128 --demb 300 --dframe 1000 --dhid 64 --HetLowSg --gpu --gpu_id 0 --DataParallelDevice 0 --DataParallelDevice 1
```

# Semantic graph
## generate Semantic graph data
```
cd alfred/gen
python scripts/augment_meta_data_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
```

## train Semantic graph
```
python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_semantic --dout exp/memory{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 128 --demb 300 --dframe 1000 --dhid 64 --HetLowSg --gpu --gpu_id 0 --DataParallelDevice 0 --DataParallelDevice 1
```

## MOCA + Semantic graph
https://github.com/gistvision/moca
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_semantic --dout exp/moca_memory{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --demb 100 --dhid 256 --gpu
```

# Eval
## Build AI2-THOR method
1. install https://github.com/allenai/ai2thor-docker
2. 		
(alfred: https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)
	- ...
	- user console run the X server
```
python3 scripts/startx.py
```

### result
XSERVTransSocketUNIXCreateListener: ...SocketCreateListener() failed
XSERVTransMakeAllCOTSServerListeners: server already running
(have running in docker)

## eval for train/valid_seen/valid_unseen
```
cd alfred
python models/eval/eval_seq2seq.py --model_path exp/json_feat_2/best_seen.pth --model models.model.seq2seq_im_mask --data data/json_feat_2.1.0 --gpu --gpu_id 0
python models/eval/eval_seq2seq.py --model_path exp/model,gcn_im,name,pm_and_subgoals_01,gcn_vial_False_19-09-2020_03-57-55/best_seen.pth --model models.model.gcn_im --data data/json_feat_2.1.0/ --gpu --gpu_id 0 --subgoals all
```
--eval_split ['train', 'valid_seen', 'valid_unseen', ]
--subgoals ['all', 'GotoLocation', 'PickupObject', ...]

## eval hete graph
```
python models/eval/eval_graph.py --model_path exp/{model_path}}/best_seen.pth --model models.model.gcn_im --data data/full_2.1.0/ --gpu --gpu_id 0 --model_hete_graph
```
--HETAttention --dgcnout 1024 --HetLowSg
--eval_split ['train', 'valid_seen', 'valid_unseen', ]
--subgoals ['all', 'GotoLocation', 'PickupObject', ...]
## Leaderboard
```
cd alfred
python models/eval/leaderboard.py --model_path <model_path>/model.pth --model models.model.seq2seq_im_mask --data data/full_2.1.0/ --gpu --num_threads 5
```

---
---

# Build Heterogeneous Graph

## fastText Word Embedding
- https://github.com/facebookresearch/fastText

### Download model
https://fasttext.cc/docs/en/english-vectors.html

### Create Alfred Objects Embedding (GCN Node feature)
You can get "./data/fastText_300d_108.json" & "./data/fastText_300d_108.h5"
```
cd graph_analysis
# download fastText English model & unzip data move to ./data/
./download_model.py en
# use fastText English model get word embedding
python alfred_dataset_vocab_analysis.py
```

## Visaul Genome dataset(GCN Edge Adjacency matrix)
https://visualgenome.org/
You have to download "scene graphs" data first from https://visualgenome.org/api/v0/api_home.html
You can get "relationship_matrics.csv" (Alfred objects relationship) & "A.csv" (Adjacency matrix)
```
cd graph_analysis/visual_genome
python visual_genome_scene_graphs_analysis.py 
```

## Create node & edge
You can get node & edge data at "./data_dgl"
```
cd graph_analysis
python create_dgl_data.py 
--object 
--verb
```

## analize verb in ALFRED
Prerequire
```
seaborn
scipy
```



---
---

# ALFRED

[<b>A Benchmark for Interpreting Grounded Instructions for Everyday Tasks</b>](https://arxiv.org/abs/1912.01734)  
[Mohit Shridhar](https://mohitshridhar.com/), [Jesse Thomason](https://jessethomason.com/), [Daniel Gordon](https://homes.cs.washington.edu/~xkcd/), [Yonatan Bisk](https://yonatanbisk.com/),  
[Winson Han](https://allenai.org/team.html), [Roozbeh Mottaghi](http://roozbehm.info/), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz), [Dieter Fox](https://homes.cs.washington.edu/~fox/)  
[CVPR 2020](http://cvpr2020.thecvf.com/)

**ALFRED** (**A**ction **L**earning **F**rom **R**ealistic **E**nvironments and **D**irectives), is a new benchmark for learning a mapping from natural language instructions and egocentric vision to sequences of actions for household tasks. Long composition rollouts with non-reversible state changes are among the phenomena we include to shrink the gap between research benchmarks and real-world applications.

For the latest updates, see: [**askforalfred.com**](https://askforalfred.com)

## Citation

If you find the dataset or code useful, please cite:

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
