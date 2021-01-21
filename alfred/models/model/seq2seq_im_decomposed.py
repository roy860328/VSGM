import os
import cv2
import torch
import numpy as np
import nn.vnn3 as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq_decomposed import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from PIL import Image
import gen.constants as constants
# # 1 background + 108 object + 10
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
from nn.resnet import Resnet
'''
semantic
'''
import sys
import importlib
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT']))
from agents.utils import tensorboard
from agents.agent import oracle_sgg_dagger_agent
import json
import glob
from icecream import ic

class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)
        '''
        semantic
        '''
        self.config = args.config_file
        self.config['general']['training']['batch_size'] = self.args.batch
        # for choose node attention input size
        self.config['general']['model']['block_hidden_dim'] = 2*args.dhid
        IMPORTENT_NDOES_FEATURE = self.config['semantic_cfg'].SCENE_GRAPH.EMBED_FEATURE_SIZE
        # Semantic graph create
        self.semantic_graph_implement = oracle_sgg_dagger_agent.SemanticGraphImplement(self.config)

        # encoder and self-attention
        self.enc_goal = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_instr = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att_goal = vnn.SelfAttn(args.dhid*2)
        self.enc_att_instr = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        if self.config['semantic_cfg'].GENERAL.DECODER == "DecomposeDec":
            decoder = vnn.DecomposeDec
        else:
            raise NotImplementedError()
        self.dec = decoder(self.emb, self.num_action_navi_or_operation,
                           args.dframe, 2*args.dhid,
                           self.semantic_graph_implement, IMPORTENT_NDOES_FEATURE,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.ce_loss = torch.nn.CrossEntropyLoss()

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'
        self.feat_exploration_pt = 'feat_exploration_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()
        self.extractor = None


    def finish_of_episode(self):
        self.semantic_graph_implement.reset_all_scene_graph()

    def _load_meta_data(self, root, list_img_traj, device):
        def sequences_to_one():
            print("_load with path", root)
            meta_datas = {
                "sgg_meta_data": [],
                "exploration_sgg_meta_data": [],
            }
            low_idx = -1
            for i, dict_frame in enumerate(list_img_traj):
                # 60 actions need 61 frames
                if low_idx != dict_frame["low_idx"]:
                    low_idx = dict_frame["low_idx"]
                else:
                    continue
                name_frame = dict_frame["image_name"].split(".")[0]
                file_path = os.path.join(root, "sgg_meta", name_frame + ".json")
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        meta_data = json.load(f)
                    meta_data = {
                        "rgb_image": [],
                        "sgg_meta_data": meta_data,
                    }
                    meta_datas["sgg_meta_data"].append(meta_data)
                else:
                    print("file is not exist: {}".format(file_path))
            meta_datas["sgg_meta_data"].append(meta_data)
            exploration_path = os.path.join(root, "exploration_meta", "*.json")
            exploration_file_paths = glob.glob(exploration_path)
            for exploration_file_path in exploration_file_paths:
                with open(exploration_file_path, 'r') as f:
                    meta_data = json.load(f)
                meta_data = {
                    "exploration_sgg_meta_data": meta_data,
                }
                meta_datas["exploration_sgg_meta_data"].append(meta_data)
            return meta_datas
        all_meta_data_path = os.path.join(root, "all_meta_data.json")
        if os.path.isfile(all_meta_data_path):
            with open(all_meta_data_path, 'r') as f:
                all_meta_data = json.load(f)
        else:
            all_meta_data = sequences_to_one()
            with open(all_meta_data_path, 'w') as f:
                json.dump(all_meta_data, f)
        exporlation_ims = torch.load(os.path.join(root, self.feat_exploration_pt)).to(device)
        all_meta_data["exploration_imgs"] = exporlation_ims
        return all_meta_data

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # subgoal completion supervision
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            feat['lang_goal'].append(lang_goal)
            feat['lang_instr'].append(lang_instr)

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                action_low, action_navi_low, action_operation_low, action_navi_or_operation = [], [], [], []
                for a in ex['num']['action_low']:
                    # operation
                    alow_index = self.old_action_low_index_to_navi_or_operation[a['action']]
                    # operation = 1
                    if alow_index[0]:
                        action_navi_low.append(self.pad)
                        action_operation_low.append(alow_index[1])
                    # navi = 0
                    else:
                        action_operation_low.append(self.pad)
                        action_navi_low.append(alow_index[1])
                    action_low.append(alow_index[1])
                    action_navi_or_operation.append(alow_index[0])
                # new action_low index
                # ic(action_low)
                # ic(action_navi_low)
                # ic(action_operation_low)
                # ic(action_navi_or_operation)
                feat['action_low'].append(action_low)
                feat['action_navi_low'].append(action_navi_low)
                feat['action_operation_low'].append(action_operation_low)
                feat['action_navi_or_operation'].append(action_navi_or_operation)

                # low-level action mask
                if load_mask:
                    indices = []
                    for a in ex['plan']['low_actions']:
                        if a['api_action']['action'] in ['MoveAhead', 'LookUp', 'LookDown', 'RotateRight', 'RotateLeft']:
                            continue
                        if a['api_action']['action'] == 'PutObject':
                            label = a['api_action']['receptacleObjectId'].split('|')
                        else:
                            label = a['api_action']['objectId'].split('|')
                        indices.append(classes.index(label[4].split('_')[0] if len(label) >= 5 else label[0]))
                    feat['action_low_mask_label'].append(indices)
                    feat['action_low_mask'].append([self.decompress_mask(a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])
                # low-level valid interact
                feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                all_meta_data = self._load_meta_data(root, ex["images"], device)
                feat['all_meta_datas'].append(all_meta_data)  # add stop frame

                im = torch.load(os.path.join(root, self.feat_pt))

                num_low_actions = len(ex['plan']['low_actions'])
                num_feat_frames = im.shape[0]

                if num_low_actions != num_feat_frames:
                    keep = [None] * len(ex['plan']['low_actions'])
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep.append(keep[-1])  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))
                else:
                    feat['frames'].append(torch.cat([im, im[-1].unsqueeze(0)], dim=0))  # add stop frame

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal', 'lang_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'action_low_mask_label'}:
                # label
                seqs = torch.tensor([vvv for vv in v for vvv in vv], device=device, dtype=torch.long)
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            elif k in {'action_navi_low', 'action_operation_low', 'action_navi_or_operation'}:
                seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
                # seqs = torch.tensor([vvv for vv in v for vvv in vv], device=device, dtype=torch.long)
                # feat[k] = seqs
            elif k in {'all_meta_datas'}:
                pass
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat


    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if not self.test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask


    def forward(self, feat, max_decode=300):
        cont_lang_goal, enc_lang_goal = self.encode_lang(feat)
        cont_lang_instr, enc_lang_instr = self.encode_lang_instr(feat)
        state_0_goal = cont_lang_goal, torch.zeros_like(cont_lang_goal)
        state_0_instr = cont_lang_instr, torch.zeros_like(cont_lang_instr)
        frames = self.vis_dropout(feat['frames'])
        res = self.dec(enc_lang_goal, enc_lang_instr, frames, feat['all_meta_datas'],
            max_decode=max_decode, gold=feat['action_low'], state_0_goal=state_0_goal, state_0_instr=state_0_instr)
        feat.update(res)
        return feat


    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang = feat['lang_goal']
        
        self.lang_dropout(emb_lang.data)
        
        enc_lang, _ = self.enc_goal(emb_lang)
        enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
        
        self.lang_dropout(enc_lang)
        
        cont_lang = self.enc_att_goal(enc_lang)

        return cont_lang, enc_lang

    def encode_lang_instr(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang = feat['lang_instr']
        
        self.lang_dropout(emb_lang.data)
        
        enc_lang, _ = self.enc_instr(emb_lang)
        enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
        
        self.lang_dropout(enc_lang)
        
        cont_lang = self.enc_att_instr(enc_lang)

        return cont_lang, enc_lang


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t_goal': None,
            'state_t_instr': None,
            'e_t': None,
            'cont_lang_goal': None,
            'enc_lang_goal': None,
            'cont_lang_instr': None,
            'enc_lang_instr': None,
        }


    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features (goal)
        if self.r_state['cont_lang_goal'] is None and self.r_state['enc_lang_goal'] is None:
            self.r_state['cont_lang_goal'], self.r_state['enc_lang_goal'] = self.encode_lang(feat)

        # encode language features (instr)
        if self.r_state['cont_lang_instr'] is None and self.r_state['enc_lang_instr'] is None:
            self.r_state['cont_lang_instr'], self.r_state['enc_lang_instr'] = self.encode_lang_instr(feat)

        # initialize embedding and hidden states (goal)
        if self.r_state['state_t_goal'] is None:
            self.r_state['state_t_goal'] = self.r_state['cont_lang_goal'], torch.zeros_like(self.r_state['cont_lang_goal'])

        # initialize embedding and hidden states (instr)
        if self.r_state['e_t'] is None and self.r_state['state_t_instr'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang_instr'].size(0), 1)
            self.r_state['state_t_instr'] = self.r_state['cont_lang_instr'], torch.zeros_like(self.r_state['cont_lang_instr'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        '''
        semantic graph
        '''
        # batch = 1
        all_meta_datas = feat['all_meta_datas']
        feat_global_graph = []
        feat_current_state_graph = []
        feat_history_changed_nodes_graph = []
        feat_priori_graph = []
        for env_index in range(len(all_meta_datas)):
            b_store_state = all_meta_datas[env_index]
            # get_meta_datas(cls, env, resnet):
            t_store_state = b_store_state["sgg_meta_data"]
            # cls.resnet.featurize([curr_image], batch=1).unsqueeze(0)
            t_store_state["rgb_image"] = feat['frames'][env_index, 0]
            self.semantic_graph_implement.store_data_to_graph(
                store_state=t_store_state,
                env_index=env_index
            )
            global_graph_importent_features, _ = \
                self.semantic_graph_implement.get_graph_feature(
                    chose_type="GLOBAL_GRAPH",
                    env_index=env_index,
                    )
            current_state_graph_importent_features, _ = \
                self.semantic_graph_implement.chose_importent_node_feature(
                    chose_type="CURRENT_STATE_GRAPH",
                    env_index=env_index,
                    hidden_state=self.r_state['state_t_instr'][0][env_index:env_index+1],
                    )
            history_changed_nodes_graph_importent_features, _ = \
                self.semantic_graph_implement.chose_importent_node_feature(
                    chose_type="HISTORY_CHANGED_NODES_GRAPH",
                    env_index=env_index,
                    hidden_state=self.r_state['state_t_goal'][0][env_index:env_index+1],
                    )
            priori_importent_features, _ = \
                self.semantic_graph_implement.chose_importent_node_feature(
                    chose_type="PRIORI_GRAPH",
                    env_index=env_index,
                    hidden_state=self.r_state['state_t_instr'][0][env_index:env_index+1],
                    )
            feat_global_graph.append(global_graph_importent_features)
            feat_current_state_graph.append(current_state_graph_importent_features)
            feat_history_changed_nodes_graph.append(history_changed_nodes_graph_importent_features)
            feat_priori_graph.append(priori_importent_features)
        feat_global_graph = torch.cat(feat_global_graph, dim=0)
        feat_current_state_graph = torch.cat(feat_current_state_graph, dim=0)
        feat_history_changed_nodes_graph = torch.cat(feat_history_changed_nodes_graph, dim=0)
        feat_priori_graph = torch.cat(feat_priori_graph, dim=0)

        # decode and save embedding and hidden states
        out_action_navi, out_action_oper, out_action_low_mask, out_action_navi_or_operation, out_action_low_masks_label,\
            state_t_goal, state_t_instr, lang_attn_t_goal, lang_attn_t_instr, *_ = \
            self.dec.step(
                self.r_state['enc_lang_goal'],
                self.r_state['enc_lang_instr'],
                feat['frames'][:, 0],
                e_t,
                self.r_state['state_t_goal'],
                self.r_state['state_t_instr'],
                feat_global_graph,
                feat_current_state_graph,
                feat_history_changed_nodes_graph,
                feat_priori_graph
            )
        w_t = self.dec.chose_embed_index(out_action_navi, out_action_oper, out_action_navi_or_operation)
        e_t = self.dec.emb(w_t)

        # save states
        self.r_state['state_t_goal'] = state_t_goal
        self.r_state['state_t_instr'] = state_t_instr
        self.r_state['e_t'] = e_t

        # output formatting
        feat['out_action_navi'] = out_action_navi.unsqueeze(0)
        feat['out_action_oper'] = out_action_oper.unsqueeze(0)
        feat['out_action_navi_or_operation'] = out_action_navi_or_operation.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        feat['out_action_low_masks_label'] = out_action_low_masks_label.unsqueeze(0)

        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow_navi_or_operation, alow_navi, alow_operation, alow_mask, alow_mask_label in \
            zip(batch, feat['out_action_navi_or_operation'].max(2)[1].tolist(), feat['out_action_navi_low'].max(2)[1].tolist(), feat['out_action_operation_low'].max(2)[1].tolist(), feat['out_action_low_mask'], feat['out_action_low_mask_label'].max(2)[1].tolist()):
            # remove padding tokens
            if self.pad in alow_navi:
                pad_start_idx = alow_navi.index(self.pad)
                alow_navi = alow_navi[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]
                alow_operation = alow_operation[:pad_start_idx]
                alow_navi_or_operation = alow_navi_or_operation[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow_navi:
                    stop_start_idx = alow_navi.index(self.stop_token)
                    alow_navi = alow_navi[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]
                    alow_operation = alow_operation[:stop_start_idx]
                    alow_navi_or_operation = alow_navi_or_operation[:stop_start_idx]

            # index to API actions
            # words = self.vocab['action_low'].index2word(alow)
            alow_navi_words = [self.action_low_index_to_word[alow] for alow in alow_navi]
            alow_operation_words = [self.action_low_index_to_word[alow] for alow in alow_operation]
            action_low = [alow_operation_words[i] if is_o else alow_navi_words[i] for i, is_o in enumerate(alow_navi_or_operation)]
            # ic(alow_navi_words)

            p_mask = [alow_mask[t].detach().cpu().numpy() for t in range(alow_mask.shape[0])]

            pred[self.get_task_and_ann_id(ex)] = {
                'action_low': ' '.join(action_low),
                'action_navi_low': ' '.join(alow_navi_words),
                'action_operation_low': ' '.join(alow_operation_words),
                'action_low_mask': p_mask,
                'alow_mask_label': alow_mask_label,
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.action_low_word_to_index[action], device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()
        # 'action_low'
        # 'action_navi_low'
        # 'action_operation_low'
        # 'action_navi_or_operation'
        # 'out_action_navi_low'
        # 'out_action_operation_low'
        # 'out_action_navi_or_operation'
        # GT and predictions
        p_alow_navi = out['out_action_navi_low'].view(-1, len(self.vocab['action_low']))
        l_alow_navi = feat['action_navi_low'].view(-1)
        p_alow_oper = out['out_action_operation_low'].view(-1, len(self.vocab['action_low']))
        l_alow_oper = feat['action_operation_low'].view(-1)
        p_alow_navi_or_operation = out['out_action_navi_or_operation'].view(-1, self.num_action_navi_or_operation)
        l_alow_navi_or_operation = feat['action_navi_or_operation'].view(-1)
        # action navi loss
        pad_valid = (l_alow_navi != self.pad)
        alow_navi_loss = F.cross_entropy(p_alow_navi, l_alow_navi, reduction='none')
        alow_navi_loss *= pad_valid.float()
        alow_navi_loss = alow_navi_loss.mean()
        losses['action_navi_low'] = alow_navi_loss * self.args.action_navi_loss_wt
        # action oper loss
        pad_valid = (l_alow_oper != self.pad)
        alow_oper_loss = F.cross_entropy(p_alow_oper, l_alow_oper, reduction='none')
        alow_oper_loss *= pad_valid.float()
        alow_oper_loss = alow_oper_loss.mean()
        losses['action_oper_low'] = alow_oper_loss * self.args.action_oper_loss_wt
        # navi_or_operation loss
        pad_valid = (l_alow_navi_or_operation != self.pad)
        alow_navi_or_operation_loss = F.cross_entropy(p_alow_navi_or_operation, l_alow_navi_or_operation, reduction='none')
        alow_navi_or_operation_loss *= pad_valid.float()
        alow_navi_or_operation_loss = alow_navi_or_operation_loss.mean()
        losses['action_navi_or_operation_low'] = alow_navi_or_operation_loss * self.args.action_navi_or_oper_loss_wt

        # mask
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']
        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0]*p_alow_mask.shape[1], *p_alow_mask.shape[2:])[valid_idxs]
        flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)
        alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask, flat_alow_mask)
        losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt

        # mask label
        p_alow_mask = out['out_action_low_mask_label']
        valid = feat['action_low_valid_interact']
        # mask label loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0] * p_alow_mask.shape[1], p_alow_mask.shape[2])[valid_idxs]
        losses['action_low_mask_label'] = self.ce_loss(flat_p_alow_mask, feat['action_low_mask_label']) * self.args.mask_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss
        return losses


    def weighted_mask_loss(self, pred_masks, gt_masks):
        '''
        mask loss that accounts for weight-imbalance between 0 and 1 pixels
        '''
        bce = self.bce_with_logits(pred_masks, gt_masks)
        flipped_mask = self.flip_tensor(gt_masks)
        inside = (bce * gt_masks).sum() / (gt_masks).sum()
        outside = (bce * flipped_mask).sum() / (flipped_mask).sum()
        return inside + outside


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}