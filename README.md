# Command

## ALFRED_ROOT
```
cd gcn/alfred
export ALFRED_ROOT=/home/host/gcn/alfred/
SET ALFRED_ROOT=D:\gcn\alfred
```

## Prerequire
```
conda install -c menpo opencv
```

## Seq2Seq
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model seq2seq_im_mask --dout exp/model,{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```

## GCN
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 
```
--data/json_feat_2.1.0/

gcn visaul embedding
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --gcn_cat_visaul --gpu_id 1
```

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
You can get "relationship_matrics.csv" (Alfred objects relationship).
```
cd graph_analysis/genome
python visual_genome_scene_graphs_analysis.py 
```
## Eval
### Build AI2-THOR method
1. install https://github.com/allenai/ai2thor-docker
2. sudo Xorg -noreset -sharevts -novtswitch -isolateDevice "PCI:1:0:0" :0 vt1 & sleep 1 sudo Xorg -noreset -sharevts -novtswitch -isolateDevice "PCI:2:0:0" :1 vt1 &

#### result
XSERVTransSocketUNIXCreateListener: ...SocketCreateListener() failed
XSERVTransMakeAllCOTSServerListeners: server already running
(have running in docker)

### eval for train/valid_seen/valid_unseen
```
cd alfred
python models/eval/eval_seq2seq.py --model_path exp/json_feat_2/best_seen.pth --model models.model.seq2seq_im_mask --data data/json_feat_2.1.0 --gpu
```
--eval_split ['train', 'valid_seen', 'valid_unseen', ]
--subgoals ['all', 'GotoLocation', 'PickupObject', ...]

### Leaderboard
```
cd alfred
python models/eval/leaderboard.py --model_path <model_path>/model.pth --model models.model.seq2seq_im_mask --data data/json_feat_2.1.0 --gpu --num_threads 5
```

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
