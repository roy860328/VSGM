# Command

## ALFRED_ROOT
```
export ALFRED_ROOT=/home/host/alfred/
SET ALFRED_ROOT=D:\alfred
```

## Prerequire
```
conda install -c menpo opencv
```

## Seq2Seq
```
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model seq2seq_im_mask --dout exp/model,{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```

## GCN
```
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 
```
gcn visaul embedding
```
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --gcn_cat_visaul --gpu_id 1
```

## fastText Word Embedding
- https://github.com/facebookresearch/fastText

### Download model
https://fasttext.cc/docs/en/english-vectors.html

## Eval
### Run THOR method 1
1. install https://github.com/allenai/ai2thor-docker
2. sudo Xorg -noreset -sharevts -novtswitch -isolateDevice "PCI:1:0:0" :0 vt1 & sleep 1 sudo Xorg -noreset -sharevts -novtswitch -isolateDevice "PCI:2:0:0" :1 vt1 &

#### result
XSERVTransSocketUNIXCreateListener: ...SocketCreateListener() failed
XSERVTransMakeAllCOTSServerListeners: server already running
(have running in docker)

### Run THOR method 2
https://github.com/askforalfred/alfred/tree/master/scripts
1. 
```
docker_build: user_name= "new_user_name"
sudo python3 scripts/docker_build.py -uid 1004 -gid 1004
```

2. 
```
docker_run: user_name= "new_user_name"
sudo python3 scripts/docker_run.py
```

#### result
Logging to /home/roy1/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-Thor/Player.log
No protocol specified

### eval
```
python models/eval/eval_seq2seq.py --model_path exp/json_feat_2/best_seen.pth --model models.model.seq2seq_im_mask --data data/json_feat_2.1.0 --gpu
```

### Leaderboard
```
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
