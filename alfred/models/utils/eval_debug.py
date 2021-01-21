import imageio
import os
import torch
import numpy as np
import cv2
SAVE_FOLDER_NAME = "eval_video"
font = cv2.FONT_HERSHEY_SIMPLEX
toptomLeftCornerOfText = (10, 30)
middleLeftCornerOfText = (10, 230)
bottomLeftCornerOfText = (10, 270)
topmiddleOfText = (600, 30)
fontScale = 0.7
fontColor = (255, 0, 0)
lineType = 2


class EvalDebug():
    """docstring for EvalDebug"""
    def __init__(self):
        super(EvalDebug, self).__init__()
        self.reset_data()

    def reset_data(self):
        self.images = []
        self.depths = []
        self.dict_masks = []
        self.list_actions = []
        self.fail_reason_list = []
        self.fail_reason = ""

    def add_data(self, step, image, depth, dict_mask, action, fail_reason):
        if action == "":
            action = "."
        if fail_reason != "":
            # string-
            fail_reason = str(fail_reason)
            self.fail_reason += str(step) + ": " + fail_reason
            self.fail_reason += "\n"
        else:
            fail_reason = "."
        if dict_mask["mask"] is None:
            dict_mask["mask"] = (np.ones(depth.shape)*255).astype(np.uint8)
        else:
            dict_mask["mask"] = (dict_mask["mask"]*255).astype(np.uint8)
        self.images.append(image)
        self.depths.append(depth)
        self.dict_masks.append(dict_mask)
        self.list_actions.append(action)
        self.fail_reason_list.append(fail_reason)


    def store_fail_case(self, file_name, save_dir):
        save_fail_case = os.path.join(save_dir, file_name + "_fail.txt")
        with open(save_fail_case, 'w') as f:
            f.write(self.fail_reason)

    def images_to_video(self, file_name, save_dir, goal_instr, fps):
        file_name += ".mp4"
        save_video_dir = os.path.join(save_dir, file_name)
        writer = imageio.get_writer(save_video_dir, fps=fps)
        # data
        for image, depth, dict_mask, action, fail_reason in zip(self.images, self.depths, self.dict_masks, self.list_actions, self.fail_reason_list):
            depth = np.expand_dims(depth, axis=2)
            depth = np.tile(depth, (1, 3))
            mask = np.expand_dims(dict_mask["mask"], axis=2)
            mask = np.tile(mask, (1, 3))
            cat_image = np.concatenate([image, depth, mask], axis=1)
            # action
            cv2.putText(cat_image, action,
                        toptomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            # fail_reason
            cv2.putText(cat_image, fail_reason,
                        middleLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            # goal_instr
            cv2.putText(cat_image, goal_instr,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            # dict_mask
            cv2.putText(cat_image, str(dict_mask["pred_class"]) + dict_mask["object"],
                        topmiddleOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            writer.append_data(cat_image)
        writer.close()

    def record(self, save_dir, traj_data, goal_instr, fail_reason, success, fps=2):
        # path
        if success:
            file_name = "S_"
        else:
            file_name = "F_"
        file_name += str(traj_data['repeat_idx'])
        fold_name = traj_data['task_type'] + '_' + traj_data['task_id']

        save_dir = os.path.join(save_dir, SAVE_FOLDER_NAME, fold_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.images_to_video(file_name, save_dir, goal_instr, fps)
        self.store_fail_case(file_name, save_dir)
        self.reset_data()
