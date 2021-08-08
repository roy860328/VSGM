import os
import cv2
import torch
import numpy as np
import nn.vnn5_graph_map as vnn
import collections
from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq_im_moca_semantic import Module as seq2seq_im_moca_semantic
import gen.constants as constants
# # 1 background + 108 object + 10
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
from PIL import Image
import json
import glob
from gen.utils.image_util import decompress_mask


class Module(seq2seq_im_moca_semantic):

    def __init__(self, args, vocab):
        super().__init__(args, vocab, importent_nodes=True)
        '''
        Seq2Seq agent
        '''
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att = vnn.SelfAttn(args.dhid*2)
        self.enc_goal = None
        self.enc_instr = None
        self.enc_att_goal = None
        self.enc_att_instr = None
        '''
        moca_graph_map ori
        '''
        IMPORTENT_NDOES_FEATURE = self.config['semantic_cfg'].SCENE_GRAPH.EMBED_FEATURE_SIZE
        if self.config['semantic_cfg'].GENERAL.DECODER == "SEQGRAPHMAP":
            decoder = vnn.SEQGRAPHMAP
        else:
            print("self.config['semantic_cfg'].GENERAL.DECODER not found\n", self.config['semantic_cfg'].GENERAL.DECODER)
            raise
        # else:
        #     decoder = vnn.ImportentNodes
        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           self.semantic_graph_implement, IMPORTENT_NDOES_FEATURE,
                           args.sgg_pool, args.gpu_id,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)
        if "FEAT_NAME" in self.config['semantic_cfg'].GENERAL and self.config['semantic_cfg'].GENERAL.FEAT_NAME != "feat_conv.pt":
            self.feat_pt = self.config['semantic_cfg'].GENERAL.FEAT_NAME
        else:
            self.feat_pt = 'feat_sgg_depth_instance_test.pt'
        self.device = torch.device(self.args.gpu_id) if self.args.gpu else torch.device('cpu')
        self.to(self.device)

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            self.r_state['state_t'] = self.r_state['cont_lang'], torch.zeros_like(self.r_state['cont_lang'])


        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        feat["frames_instance"] = torch.tensor(np.array(feat["frame_instance"])).unsqueeze(0).unsqueeze(0)
        if "RGB_FEAT" in self.config['semantic_cfg'].GENERAL and self.config['semantic_cfg'].GENERAL.RGB_FEAT:
            feat["frames_instance"] = torch.tensor(np.array(feat["frames_rgb"])).unsqueeze(0).unsqueeze(0)
        feat["frames_depth"] = torch.tensor(np.array(feat["frame_depth"])).unsqueeze(0).unsqueeze(0)
        '''
        semantic graph
        '''
        # batch = 1
        all_meta_datas = feat['all_meta_datas']
        frames_conv = {}
        feat_global_graph = []
        feat_current_state_graph = []
        feat_history_changed_nodes_graph = []
        feat_priori_graph = []
        feat_graph_map = []
        for env_index in range(len(all_meta_datas)):
            b_store_state = all_meta_datas[env_index]
            global_graph_importent_features, current_state_graph_importent_features, history_changed_nodes_graph_importent_features, priori_importent_features, graph_map_importent_features,\
                global_graph_dict_objectIds_to_score, current_state_dict_objectIds_to_score, history_changed_dict_objectIds_to_score, priori_dict_dict_objectIds_to_score, graph_map_dict_objectIds_to_score =\
                self.dec.store_and_get_graph_feature(
                    b_store_state, feat, 0, env_index, self.r_state['state_t'], frames_conv)
            feat_global_graph.append(global_graph_importent_features)
            feat_current_state_graph.append(current_state_graph_importent_features)
            feat_history_changed_nodes_graph.append(history_changed_nodes_graph_importent_features)
            feat_priori_graph.append(priori_importent_features)
            feat_graph_map.append(graph_map_importent_features)
        feat_global_graph = torch.cat(feat_global_graph, dim=0)
        feat_current_state_graph = torch.cat(feat_current_state_graph, dim=0)
        feat_history_changed_nodes_graph = torch.cat(feat_history_changed_nodes_graph, dim=0)
        feat_priori_graph = torch.cat(feat_priori_graph, dim=0)
        feat_graph_map = torch.cat(feat_graph_map, dim=0)

        out_action_low, out_action_low_mask, state_t, attn_score_t, subgoal_t, progress_t = \
            self.dec.step(
                self.r_state['enc_lang'],
                {k: torch.cat(v, dim=0).to(device=self.args.gpu_id) for k, v in frames_conv.items()},# frames[:, t],
                e_t,
                self.r_state['state_t'],
                feat_global_graph,
                feat_current_state_graph,
                feat_history_changed_nodes_graph,
                feat_priori_graph,
                feat_graph_map,)


        # save states
        self.r_state['state_t'] = state_t
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        assert len(all_meta_datas) == 1, "if not the analyze_graph object ind is error"
        global_graph_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            global_graph_dict_objectIds_to_score, graph_type="GLOBAL_GRAPH")
        current_state_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            current_state_dict_objectIds_to_score, graph_type="CURRENT_STATE_GRAPH")
        history_changed_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            history_changed_dict_objectIds_to_score, graph_type="HISTORY_CHANGED_NODES_GRAPH")
        priori_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            priori_dict_dict_objectIds_to_score, graph_type="PRIORI_GRAPH")

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        feat['out_subgoal_t'] = np.round(subgoal_t.view(-1).item(), decimals=2)
        feat['out_progress_t'] = np.round(progress_t.view(-1).item(), decimals=2)
        feat['global_graph_dict_ANALYZE_GRAPH'] = global_graph_dict_ANALYZE_GRAPH
        feat['current_state_dict_ANALYZE_GRAPH'] = current_state_dict_ANALYZE_GRAPH
        feat['history_changed_dict_ANALYZE_GRAPH'] = history_changed_dict_ANALYZE_GRAPH
        feat['priori_dict_ANALYZE_GRAPH'] = priori_dict_ANALYZE_GRAPH

        return feat

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            # sigmoid preds to binary mask
            alow_mask = torch.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

            task_id_ann = self.get_task_and_ann_id(ex)
            pred[task_id_ann] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
                'action_low_mask_label': p_mask,
                'action_navi_low': ".",
                'action_operation_low': ".",
                'action_navi_or_operation': [],
            }

        return pred

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device(self.args.gpu_id) if self.args.gpu else torch.device('cpu')
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
            lang_goal_instr = lang_goal + lang_instr
            feat['lang_goal_instr'].append(lang_goal_instr)
            feat['lang_goal'].append(lang_goal)
            feat['lang_instr'].append(lang_instr)

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

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

                images = self._load_img(os.path.join(root, 'instance_masks'), ex["images"], name_pt="feat_instance_tranform.pt", type_image=".png")
                if "RGB_FEAT" in self.config['semantic_cfg'].GENERAL and self.config['semantic_cfg'].GENERAL.RGB_FEAT:
                    images = self._load_img(os.path.join(root, 'raw_images'), ex["images"], name_pt="feat_conv.pt", type_image=".jpg")
                images_depth = self._load_img(os.path.join(root, 'depth_images'), ex["images"], name_pt="feat_depth_tranform.pt", type_image=".png")

                feat['frames_instance'].append(images)  # add stop frame
                feat['frames_depth'].append(images_depth)  # add stop frame

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal', 'lang_instr', 'lang_goal_instr'}:
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
            elif k in {'all_meta_datas'}:
                pass
            else:
                # default: tensorize and pad sequence
                seqs = [torch.as_tensor(vv, device=device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat

    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask

    def forward(self, feat, max_decode=120):
        cont_lang, enc_lang = self.encode_lang(feat)
        state_0 = cont_lang, torch.zeros_like(cont_lang)
        frames = {}
        for k, v in feat.items():
            if 'frames' in k:
                # frames[k] = self.vis_dropout(feat[k])
                frames[k] = feat[k]
        res = self.dec(enc_lang, frames, feat['all_meta_datas'], max_decode=120, gold=feat['action_low'], state_0=state_0)
        feat.update(res)
        return feat

    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = feat['lang_goal_instr']
        self.lang_dropout(emb_lang_goal_instr.data)
        # batch 2, batch 1 len = 145, batch 2 len = 95
        # len(emb_lang_goal_instr.data) = 240
        # enc_lang_goal_instr = (240, 145, 2, 2)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)
        # return back origin tensor
        # len(enc_lang_goal_instr) = 2, enc_lang_goal_instr[1].shape = torch.Size([145, 1024])
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        # torch.Size([2, 1024])
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)

        return cont_lang_goal_instr, enc_lang_goal_instr

    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].view(-1)
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0]*p_alow_mask.shape[1], *p_alow_mask.shape[2:])[valid_idxs]
        flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)
        alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask, flat_alow_mask)
        losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt

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

    def _load_img(self, path, list_img_traj, name_pt=None, type_image=".png"):
        path_pt = os.path.join(path, name_pt)
        def _load_with_path():
            print("_load with path", path_pt)
            frames_depth = None
            low_idx = -1
            for i, dict_frame in enumerate(list_img_traj):
                # 60 actions need 61 frames
                if low_idx != dict_frame["low_idx"]:
                    low_idx = dict_frame["low_idx"]
                else:
                    continue
                name_frame = dict_frame["image_name"].split(".")[0]
                frame_path = os.path.join(path, name_frame + type_image)
                if os.path.isfile(frame_path):
                    img_depth = Image.open(frame_path).convert("RGB")
                else:
                    print("file is not exist: {}".format(frame_path))
                # img_depth = \
                #     self.semantic_graph_implement.trans_MetaData.transforms(img_depth, None)[0]
                img_depth = torch.tensor(np.array(img_depth))
                img_depth = img_depth.unsqueeze(0)

                if frames_depth is None:
                    frames_depth = img_depth
                else:
                    frames_depth = torch.cat([frames_depth, img_depth], dim=0)
            frames_depth = torch.cat([frames_depth, img_depth], dim=0)
            torch.save(frames_depth, path_pt)
            return frames_depth

        def _load_with_pt():
            frames_depth = torch.load(path_pt)
            return frames_depth
        if os.path.isfile(path_pt):
            frames_depth = _load_with_pt()
        else:
            frames_depth = _load_with_path()
        # frames_depth = _load_with_path()
        return frames_depth

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
                file_agent_path = os.path.join(root, "agent_meta", name_frame + ".json")
                if os.path.isfile(file_path) and os.path.isfile(file_agent_path):
                    with open(file_path, 'r') as f:
                        meta_data = json.load(f)
                    with open(file_agent_path, 'r') as f:
                        agent_meta_data = json.load(f)
                    meta_data = {
                        "rgb_image": [],
                        "sgg_meta_data": meta_data,
                        "agent_meta_data": agent_meta_data,
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