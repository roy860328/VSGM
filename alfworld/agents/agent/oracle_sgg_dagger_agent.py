import os
import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
import modules.memory as memory
from agent import TextDAggerAgent
from modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule
from modules.layers import NegativeLogLoss, masked_mean, compute_mask
import torchvision.transforms as T
from torchvision import models
from torchvision.ops import boxes as box_ops
import importlib

sys.path.insert(0, os.path.join(os.environ['ALFRED_ROOT'], 'agents'))
from agents.utils import tensorboard
sys.path.insert(0, os.path.join(os.environ['ALFRED_ROOT'], 'agents', 'semantic_graph'))
import pdb
from semantic_graph import SceneGraph
from sgg import alfred_data_format, sgg


class OracleSggDAggerAgent(TextDAggerAgent):
    '''
    Vision Agent trained with DAgger
    '''
    def __init__(self, config):
        super().__init__(config)

        assert self.action_space == "generation"

        self.use_gpu = config['general']['use_cuda']
        self.transform = T.Compose([T.ToTensor()])

        # choose vision model
        self.vision_model_type = config['vision_dagger']['model_type']
        self.use_exploration_frame_feats = config['vision_dagger']['use_exploration_frame_feats']
        self.sequence_aggregation_method = config['vision_dagger']['sequence_aggregation_method']

        '''
        NEW
        '''
        # Semantic graph create
        self.cfg_semantic = config['semantic_cfg']
        self.isORACLE = self.cfg_semantic.SCENE_GRAPH.ORACLE
        self.graph_embed_model = importlib.import_module(self.cfg_semantic.SCENE_GRAPH.MODEL).Net(self.cfg_semantic, config=config)
        if self.use_gpu:
            self.graph_embed_model.cuda()
        self.scene_graphs = []
        for i in range(config['general']['training']['batch_size']):
            scene_graph = SceneGraph(self.cfg_semantic)
            self.scene_graphs.append(scene_graph)

        # initialize model
        self.trans_MetaData = alfred_data_format.TransMetaData(self.cfg_semantic)
        if not self.isORACLE:
            self.cfg_sgg = config['sgg_cfg']
            self.detector = sgg.load_pretrained_model(
                self.cfg_sgg, self.trans_MetaData.transforms, self.trans_MetaData.ind_to_classes, 'cuda'
                )
            self.detector.eval()
            self.detector.cuda()

        self.load_pretrained = self.cfg_semantic.GENERAL.LOAD_PRETRAINED
        self.load_from_tag = self.cfg_semantic.GENERAL.LOAD_PRETRAINED_PATH

        self.summary_writer = tensorboard.TensorBoardX(config["general"]["save_path"])

    def reset_all_scene_graph(self):
        for scene_graph in self.scene_graphs:
            scene_graph.init_graph_data()

    def finish_of_episode(self, episode_no, batch_size):
        super().finish_of_episode(episode_no, batch_size)
        self.reset_all_scene_graph()

    def get_env_last_event_data(self, envs):
        store_state = {
            "rgb_image": [],
            "mask_image": [],
            "sgg_meta_data": [],
        }
        for i, thor in enumerate(envs):
            env = thor.env
            rgb_image = env.last_event.frame[:, :, ::-1]
            mask_image = env.last_event.instance_segmentation_frame
            sgg_meta_data = env.last_event.metadata['objects']
            store_state["rgb_image"].append(rgb_image)
            store_state["mask_image"].append(mask_image)
            store_state["sgg_meta_data"].append(sgg_meta_data)
        return store_state

    # visual features for state representation
    def extract_visual_features(self, envs=None, store_state=None, hidden_state=None):
        if envs is not None:
            store_state = self.get_env_last_event_data(envs)
        if store_state is None:
            raise NotImplementedError()

        graph_embed_features = []
        for i in range(len(store_state["rgb_image"])):
            scene_graph = self.scene_graphs[i]
            rgb_image = store_state["rgb_image"][i]
            # mask_image = store_state["mask_image"][i]
            # color_to_obj_id_type = {}
            # for color, object_id in env.last_event.color_to_object_id.items():
            #     color_to_obj_id_type[str(color)] = object_id
            if self.isORACLE:
                sgg_meta_data = store_state["sgg_meta_data"][i]
                target = self.trans_MetaData.trans_object_meta_data_to_relation_and_attribute(sgg_meta_data)
                scene_graph.add_oracle_local_graph_to_global_graph(rgb_image, target)
            else:
                rgb_image = rgb_image.unsqueeze(0)
                results = self.detector(rgb_image)
                result = results[0]
                scene_graph.add_local_graph_to_global_graph(rgb_image, result)
            global_graph = scene_graph.get_graph_data()
            graph_embed_feature = self.graph_embed_model(global_graph, hidden_state=hidden_state)
            graph_embed_features.append(graph_embed_feature)
        return graph_embed_features, store_state

    # without recurrency
    def train_dagger(self):
        raise NotImplementedError()

    # with recurrency
    def train_dagger_recurrent(self):

        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        # self.dagger_replay_batch_size, self.dagger_replay_sample_history_length = 64, 4
        sequence_of_transitions, contains_first_step = self.dagger_memory.sample_sequence(self.dagger_replay_batch_size, self.dagger_replay_sample_history_length)
        if sequence_of_transitions is None:
            return None

        batches = []
        for transitions in sequence_of_transitions:
            batch = memory.dagger_transition(*zip(*transitions))
            batches.append(batch)

        if self.action_space == "generation":
            return self.train_command_generation_recurrent_teacher_force([batch.observation_list for batch in batches], [batch.task_list for batch in batches], [batch.target_list for batch in batches], contains_first_step)
        else:
            raise NotImplementedError()

    # not recurrent loss
    def train_command_generation_teacher_force(self, observation_feats, task_desc_strings, target_strings):
        input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in target_strings]
        output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in target_strings]
        batch_size = len(observation_feats)

        aggregated_obs_feat = self.aggregate_feats_seq(observation_feats)
        h_obs = self.online_net.vision_fc(aggregated_obs_feat)
        # ['[SEP] clean some potato and put it in garbagecan.']
        # torch.Size([1, 14, 64])
        # tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]])
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")
        h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
        h_obs = h_obs.to(h_td_mean.device)
        # torch.Size([1, 2, 64])
        vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hi
        vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

        input_target = self.get_word_input(input_target_strings)
        ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
        target_mask = compute_mask(input_target)  # mask of ground truth should be the same
        pred = self.online_net.vision_decode(input_target, target_mask, vision_td, vision_td_mask, None)  # batch x target_length x vocab

        batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
        loss = torch.mean(batch_loss)

        if loss is None:
            return None, None
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(pred), to_np(loss)

    # loss
    def train_command_generation_recurrent_teacher_force(self, store_state, seq_task_desc_strings, seq_target_strings, contains_first_step=False, train_now=True):
        # pdb.set_trace()
        loss_list = []
        previous_dynamics = None
        batch_size = len(seq_target_strings[0])
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
        # with torch.autograd.set_detect_anomaly(True):
        for step_no in range(len(seq_target_strings)):
            input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in seq_target_strings[step_no]]
            output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in seq_target_strings[step_no]]
            observation_feats, _ = self.extract_visual_features(store_state=store_state[step_no], hidden_state=previous_dynamics)

            obs = [o.to(h_td.device) for o in observation_feats]
            aggregated_obs_feat = self.aggregate_feats_seq(obs)
            h_obs = self.online_net.vision_fc(aggregated_obs_feat)
            vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hid
            vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

            averaged_vision_td_representation = self.online_net.masked_mean(vision_td, vision_td_mask)
            current_dynamics = self.online_net.rnncell(averaged_vision_td_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_vision_td_representation)

            input_target = self.get_word_input(input_target_strings)
            ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            pred = self.online_net.vision_decode(input_target, target_mask, vision_td, vision_td_mask, current_dynamics)  # batch x target_length x vocab

            previous_dynamics = current_dynamics

            batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
            loss = torch.mean(batch_loss)
            loss_list.append(loss)

        if loss_list is None:
            return None
        loss = torch.stack(loss_list).mean()
        print("loss: ", loss)
        if train_now:
            loss = self.grad(loss)
            return loss
        else:
            return loss

    def grad(self, loss):
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(loss)

    # recurrent
    def command_generation_greedy_generation(self, observation_feats, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            # pdb.set_trace()
            # print(observation_feats to word)
            batch_size = len(observation_feats)

            # torch.Size([1, 1024])
            aggregated_obs_feat = self.aggregate_feats_seq(observation_feats)
            # torch.Size([1, 1, 64])
            h_obs = self.online_net.vision_fc(aggregated_obs_feat)
            # ['[SEP] clean some potato and put it in garbagecan.']
            # torch.Size([1, 14, 64])
            # tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]])
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
            h_obs = h_obs.to(h_td_mean.device)
            # torch.Size([1, 2, 64])
            vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hid
            vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

            if self.recurrent:
                averaged_vision_td_representation = self.online_net.masked_mean(vision_td, vision_td_mask)
                current_dynamics = self.online_net.rnncell(averaged_vision_td_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_vision_td_representation)
            else:
                current_dynamics = None

            # greedy generation
            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):

                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                # tensor([[[4.1473e-05, 3.1972e-05, 3.8759e-05,  ..., 3.6035e-05, 3.4417e-05, 6.3384e-05]]])
                # torch.Size([1, 1, 28996])
                pred = self.online_net.vision_decode(input_target, target_mask, vision_td, vision_td_mask, current_dynamics)  # batch x target_length x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            res = [self.tokenizer.decode(item) for item in input_target_list]
            res = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in res]
            res = [item.replace(" in / on ", " in/on " ) for item in res]
            return res, current_dynamics

    def get_vision_feat_mask(self, observation_feats):
        batch_size = len(observation_feats)
        num_vision_feats = [of.shape[0] for of in observation_feats]
        max_feat_len = max(num_vision_feats)
        mask = torch.zeros((batch_size, max_feat_len))
        for b, num_vision_feat in enumerate(num_vision_feats):
            mask[b,:num_vision_feat] = 1
        return mask

    def extract_exploration_frame_feats(self, exploration_frames):
        exploration_frame_feats = []
        for batch in exploration_frames:
            ef_feats = []
            for image in batch:
                raise NotImplementedError()
                # observation_feats, _ = self.extract_visual_features(envs=env.envs, store_state=store_state[step_no])
                ef_feats.append(self.extract_visual_features([image])[0])
            # cat_feats = torch.cat(ef_feats, dim=0)
            max_feat_len = max([f.shape[0] for f in ef_feats])
            stacked_feats = self.online_net.vision_fc.pad_and_stack(ef_feats, max_feat_len=max_feat_len)
            stacked_feats = stacked_feats.view(-1, self.online_net.vision_fc.in_features)
            exploration_frame_feats.append(stacked_feats)
        return exploration_frame_feats

    def aggregate_feats_seq(self, feats):
        if self.sequence_aggregation_method == "sum":
            return [f.sum(0).unsqueeze(0) for f in feats]
        elif self.sequence_aggregation_method == "average":
            return [f.mean(0).unsqueeze(0) for f in feats]
        elif self.sequence_aggregation_method == "rnn":
            max_feat_len = max([f.shape[0] for f in feats])
            feats_stack = self.online_net.vision_fc.pad_and_stack(feats, max_feat_len=max_feat_len)
            feats_h, feats_c = self.online_net.vision_feat_seq_rnn(feats_stack)
            aggregated_feats = feats_h[:,0,:].unsqueeze(1)
            return [b for b in aggregated_feats]
        else:
            raise ValueError("sequence_aggregation_method must be sum, average or rnn")
