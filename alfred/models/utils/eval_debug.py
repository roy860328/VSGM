import imageio
import os
import sys
import torch
import numpy as np
import cv2
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents', 'semantic_graph'))
from graph_map.graph_map import BasicGraphMap

SAVE_FOLDER_NAME = "eval_video"
font = cv2.FONT_HERSHEY_SIMPLEX
toptomLeftCornerOfText = (10, 20)
topmiddleLeftCornerOfText = (10, 50)
middleLeftCornerOfText = (10, 230)
bottomLeftCornerOfText = (10, 270)
topmiddleOfText = (600, 30)
fontScale = 0.7
r_fontColor = (255, 0, 0)
g_fontColor = (0, 255, 0)
lineType = 2


class EvalDebug():
    """docstring for EvalDebug"""
    def __init__(self):
        super(EvalDebug, self).__init__()
        self.graph_map = None
        self.reset_data()

    def reset_data(self):
        self.images = []
        self.depths = []
        self.list_actions = []
        self.lang_instr = []
        self.fail_reason_list = []
        self.fail_reason = ""

        self.list_row2_img_detection_graph_graphmap = []
        if self.graph_map:
            self.graph_map.reset_map()

    def set_graph_map(self, model):
        model.semantic_graph_implement.cfg_semantic.GRAPH_MAP.GRAPH_MAP_SIZE_S = 20
        model.semantic_graph_implement.cfg_semantic.GRAPH_MAP.GRID_MIN_SIZE_R = 0.1
        model.semantic_graph_implement.cfg_semantic.GRAPH_MAP.GRAPH_MAP_CLASSES = 108
        self.graph_map = BasicGraphMap(
            model.semantic_graph_implement.cfg_semantic,
            model.semantic_graph_implement.trans_MetaData.SGG_result_ind_to_classes,
            )


    def add_data(self, step, image, depth, dict_action, lang_instr, fail_reason):
        if fail_reason != "":
            # string-
            fail_reason = str(fail_reason)
            self.fail_reason += str(step) + ": " + fail_reason
            self.fail_reason += "\n"
        else:
            fail_reason = "."
        if dict_action["mask"] is None:
            dict_action["mask"] = (np.ones(depth.shape)*255).astype(np.uint8)
        else:
            dict_action["mask"] = (dict_action["mask"]*255).astype(np.uint8)
        self.images.append(image)
        self.depths.append(depth)
        self.list_actions.append(dict_action)
        self.lang_instr.append(lang_instr)
        self.fail_reason_list.append(fail_reason)


    def store_state_case(self, file_name, save_dir, goal_instr, step_instr):
        save_fail_case = os.path.join(save_dir, file_name + "_state.txt")
        with open(save_fail_case, 'w') as f:
            f.write("Save Dir: " + save_fail_case)
            f.write("\nGoal: " + goal_instr + "\nstep_instr: ")
            f.write("\nstep_instr: ".join(step_instr) + "\n")
            f.write(self.fail_reason)

    def store_current_state(self, file_name, save_dir, text):
        save_fail_case = os.path.join(save_dir, file_name + "_state.txt")
        with open(save_fail_case, 'a') as f:
            f.write("\n" + text)

    def record(self, save_dir, traj_data, goal_instr, step_instr, fail_reason, success, fps=2, eval_idx=""):
        # path
        if success:
            file_name = "S_"
        else:
            file_name = "F_"
        file_name += str(traj_data['repeat_idx']) + eval_idx
        fold_name = traj_data['task_type'] + '_' + traj_data['task_id']

        save_dir = os.path.join(save_dir, SAVE_FOLDER_NAME, fold_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.store_state_case(file_name, save_dir, goal_instr, step_instr)
        # self.save_row2_video(file_name, save_dir, goal_instr, fps)
        self.images_to_video(file_name, save_dir, goal_instr, fps)
        self.reset_data()

    def images_to_video(self, file_name, save_dir, goal_instr, fps):
        v_file_name = file_name + ".mp4"
        save_video_dir = os.path.join(save_dir, v_file_name)
        writer = imageio.get_writer(save_video_dir, fps=fps)
        i = 0
        sub_goal = False
        if len(self.list_row2_img_detection_graph_graphmap) == 0:
            self.list_row2_img_detection_graph_graphmap = self.images
            sub_goal = True
        # data
        for image, depth, dict_action, lang_instr, fail_reason, row2_img_detection_graph_graphmap in\
            zip(self.images, self.depths, self.list_actions, self.lang_instr, self.fail_reason_list, self.list_row2_img_detection_graph_graphmap):
            '''
            Process image
            '''
            depth = np.expand_dims(depth, axis=2)
            depth = np.tile(depth, (1, 3))
            mask = np.expand_dims(dict_action["mask"], axis=2)
            mask = np.tile(mask, (1, 3))
            mask = np.zeros([300, 300, 3])
            cat_image = np.concatenate([image, depth, mask], axis=1)
            '''
            Process string
            '''
            # action
            if len(dict_action["action_navi_or_operation"])<1\
                or dict_action["action_navi_or_operation"][0, 0]>dict_action["action_navi_or_operation"][0, 1]:
                color = r_fontColor
            else:
                color = g_fontColor
            if len(dict_action["action_navi_or_operation"])>0:
                p = dict_action["action_navi_or_operation"].tolist()
                p = np.round(p, decimals=2)
            else:
                p = []
            str_step = "step: " + str(i)
            str_action_low = dict_action["action_low"]
            str_navi_oper = "navi: " + dict_action["action_navi_low"] +\
                ", oper: " + dict_action["action_operation_low"]
            str_p_navi_or_operation = "p navi/oper: " + str(p)
            str_global_graph_dict_ANALYZE_GRAPH = "global " + str(dict_action["global_graph_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_current_state_dict_ANALYZE_GRAPH = "current " + str(dict_action["current_state_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_history_changed_dict_ANALYZE_GRAPH = "history " + str(dict_action["history_changed_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_priori_dict_ANALYZE_GRAPH = "priori " + str(dict_action["priori_dict_ANALYZE_GRAPH"]).replace(":", "").replace(" ", "")
            str_mask = str(dict_action["pred_class"]) + ", " + dict_action["object"]
            str_subgoal_progress = str(dict_action["subgoal_t"]) + ", " + str(dict_action["progress_t"])
            self.store_current_state(file_name, save_dir, str_step)
            self.store_current_state(file_name, save_dir, str_action_low)
            self.store_current_state(file_name, save_dir, str_mask)
            self.store_current_state(file_name, save_dir, "Fail: " + fail_reason)
            self.store_current_state(file_name, save_dir, str_global_graph_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_current_state_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_history_changed_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_priori_dict_ANALYZE_GRAPH)
            self.store_current_state(file_name, save_dir, str_subgoal_progress)
            i += 1
            '''
            write
            '''
            self.writeText(
                cat_image, str_step, toptomLeftCornerOfText, color)
            self.writeText(
                cat_image, str_action_low, (toptomLeftCornerOfText[0]+100, toptomLeftCornerOfText[1]), color)
            '''
            # navi oper
            self.writeText(
                cat_image, str_navi_oper, topmiddleLeftCornerOfText, r_fontColor)
            # action_navi_or_operation
            self.writeText(
                cat_image, str_p_navi_or_operation, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+30), r_fontColor)
            '''
            # ANALYZE_GRAPH
            '''
            self.writeText(
                cat_image, str_global_graph_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+60), r_fontColor, fontscale=0.6)
            self.writeText(
                cat_image, str_current_state_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+90), r_fontColor, fontscale=0.6)
            self.writeText(
                cat_image, str_history_changed_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+120), r_fontColor, fontscale=0.6)
            self.writeText(
                cat_image, str_priori_dict_ANALYZE_GRAPH, (topmiddleLeftCornerOfText[0], topmiddleLeftCornerOfText[1]+150), r_fontColor, fontscale=0.6)

            # fail_reason
            self.writeText(
                cat_image, fail_reason, middleLeftCornerOfText, r_fontColor)
            # goal_instr
            # self.writeText(
            #     cat_image, goal_instr, bottomLeftCornerOfText, r_fontColor)
            self.writeText(
                cat_image, lang_instr, (bottomLeftCornerOfText[0], bottomLeftCornerOfText[1]+20), r_fontColor)
            # dict_mask
            self.writeText(
                cat_image, str_mask, topmiddleOfText, r_fontColor)
            # goal persent: subgoal_t progress_t
            self.writeText(
                cat_image, str_subgoal_progress, (topmiddleOfText[0], topmiddleOfText[1]+270), r_fontColor)
            '''
            if sub_goal == False:
                cat_image = np.concatenate([cat_image, row2_img_detection_graph_graphmap], axis=0)
            writer.append_data(cat_image)
        writer.close()

    def writeText(self, img, string, position, color, fontscale=fontScale):
        cv2.putText(img, string,
                    position,
                    font,
                    fontscale,
                    color,
                    lineType)

    def row2_img_detection_graph_graphmap(self, model, image, depth, agent_sgg_meta_data):
        # depth = np.expand_dims(depth, axis=2)
        # depth = np.tile(depth, (1, 3))
        # mask = np.expand_dims(dict_action["mask"], axis=2)
        # mask = np.tile(mask, (1, 3))
        # cat_image = np.concatenate([image, depth, mask], axis=1)

        frames_instance = \
            model.semantic_graph_implement.trans_MetaData.transforms(image, None)[0]
        img_detection = model.semantic_graph_implement.detector.predict(frames_instance, ret_detection_img=True)
        img_graph_node = self.draw_graph_node(model)
        img_graphmap = self.draw_graphmap(model, image, depth, agent_sgg_meta_data)

        # img_graphmap[:,:,:].shape
        # (480, 640, 4)
        # img_detection = cv2.resize(np.array(img_detection), (640, 480))
        # img_graph_node = np.resize(img_graph_node[:, :, :3], (300, 300, 3))
        # img_graphmap = np.resize(img_graphmap[:, :, :3], (300, 300, 3))
        img_graph_node = img_graph_node[:, :, :3]
        img_graphmap = img_graphmap[:, :, :3]
        img_graph_node = cv2.resize(np.array(img_graph_node), (300, 300))
        img_graphmap = cv2.resize(np.array(img_graphmap), (300, 300))
        cat_image = np.concatenate([img_detection, img_graph_node, img_graphmap], axis=1)
        self.list_row2_img_detection_graph_graphmap.append(cat_image)

    def draw_graphmap(self, model, image, depth, agent_sgg_meta_data):
        frames_instance = \
            model.semantic_graph_implement.trans_MetaData.transforms(image, None)[0]
        sgg_results = model.semantic_graph_implement.detector.predict(frames_instance, 0)
        sgg_result = sgg_results[0]
        # import pdb; pdb.set_trace()
        target = {
            "bbox": sgg_result['bbox'],
            "labels": sgg_result['labels'],
        }

        depth = np.expand_dims(depth, axis=2)
        depth = np.tile(depth, (1, 3))
        cam_coords = self.graph_map.update_map(
            np.array(depth.reshape(300, 300, 3)),
            agent_sgg_meta_data,
            target)
        # import pdb; pdb.set_trace()
        self.graph_map.visualize_graph_map(None, None)
        buf_file = self.graph_map.buffer_plt[-1]
        plt_img = Image.open(buf_file)
        img_graphmap = np.array(plt_img)
        return img_graphmap

    def draw_graph_node(self, model):
        model.semantic_graph_implement.scene_graphs[0].global_graph
        model.semantic_graph_implement.scene_graphs[0].current_state_graph

        G = nx.Graph()
        node_connects = []
        edge_obj_to_obj = model.semantic_graph_implement.scene_graphs[0].current_state_graph.edge_obj_to_obj
        ind_to_obj_id = model.semantic_graph_implement.scene_graphs[0].current_state_graph.ind_to_obj_id

        G.add_nodes_from(
            [model.semantic_graph_implement.trans_MetaData.SGG_result_ind_to_classes[int(obj_id.split('_')[0])] for obj_id in list(ind_to_obj_id.values())])
        if edge_obj_to_obj is not None:
            for (src, dst) in edge_obj_to_obj.T.to('cpu').numpy().astype(int):
                src_obj_id = ind_to_obj_id[src]
                dst_obj_id = ind_to_obj_id[dst]
                # import pdb; pdb.set_trace()
                src_name = model.semantic_graph_implement.trans_MetaData.SGG_result_ind_to_classes[int(src_obj_id.split('_')[0])]
                dst_name = model.semantic_graph_implement.trans_MetaData.SGG_result_ind_to_classes[int(dst_obj_id.split('_')[0])]
                node_connects.append((src_name, dst_name))
            G.add_edges_from(node_connects)

        plt.cla()
        nx.draw(G, with_labels=True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        img_graph_node = Image.open(buf)
        img_graph_node = np.array(img_graph_node)
        return img_graph_node

    def save_row2_video(self, file_name, save_dir, goal_instr, fps):
        v_file_name = file_name + "_row2.mp4"
        save_video_dir = os.path.join(save_dir, v_file_name)
        writer = imageio.get_writer(save_video_dir, fps=fps)
        for image in self.list_row2_img_detection_graph_graphmap:
            writer.append_data(image)
        writer.close()