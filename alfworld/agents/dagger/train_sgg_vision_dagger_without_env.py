import datetime
import os
import random
import time
import copy
import json
import glob
import importlib
import numpy as np

import sys
sys.path.insert(0, os.environ['ALFRED_ROOT'])
sys.path.insert(0, os.path.join(os.environ['ALFRED_ROOT'], 'agents'))

from agent import OracleSggDAggerAgent
import modules.generic as generic
import torch
from eval import evaluate_vision_dagger
from modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory
from agents.utils.misc import extract_admissible_commands
from agents.utils.traj_process import get_traj_train_data
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pdb

def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    # pdb.set_trace()
    agent = OracleSggDAggerAgent(config)
    env_type = "AlfredThorEnv"
    alfred_env = getattr(importlib.import_module("environment"), env_type)(config, train_eval="train")
    # env = alfred_env.init_env(batch_size=1)
    json_file_list = alfred_env.json_file_list

    id_eval_env, num_id_eval_game = None, 0
    ood_eval_env, num_ood_eval_game = None, 0
    if agent.run_eval:
        # in distribution
        if config['dataset']['eval_id_data_path'] is not None:
            alfred_env = getattr(importlib.import_module("environment"), env_type)(config, train_eval="eval_in_distribution")
            id_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_id_eval_game = alfred_env.num_games
        # out of distribution
        if config['dataset']['eval_ood_data_path'] is not None:
            alfred_env = getattr(importlib.import_module("environment"), env_type)(config, train_eval="eval_out_of_distribution")
            ood_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_ood_eval_game = alfred_env.num_games

    output_dir = config["general"]["save_path"]
    data_dir = config["general"]["save_path"]
    action_space = config["dagger"]["action_space"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    step_in_total = 0
    episode_no = 0
    running_avg_student_points = HistoryScoreCache(capacity=500)
    running_avg_student_steps = HistoryScoreCache(capacity=500)
    running_avg_dagger_loss = HistoryScoreCache(capacity=500)

    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_performance_so_far = 0.0

    # load model from checkpoint
    if agent.load_pretrained:
        print(data_dir + "/" + agent.load_from_tag + ".pt")
        print(os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"))
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            print("load model")
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    while(True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        print("reload")
        print("reload ends")
        batch_size = agent.batch_size
        tasks = random.sample(json_file_list, k=batch_size)
        save_frames_path = config['env']['thor']['save_frames_path']
        transition_caches = get_traj_train_data(tasks, save_frames_path)

        agent.train()
        agent.init(batch_size)
        previous_dynamics = None
        report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= (episode_no - batch_size) % agent.report_frequency)

        losses = []
        for transition_cache in transition_caches:
            agent.reset_all_scene_graph()
            store_states, task_desc_strings, expert_actions = transition_cache[0], transition_cache[1], transition_cache[2]
            loss = agent.train_command_generation_recurrent_teacher_force(
                store_states,
                task_desc_strings,
                expert_actions,
                train_now=False,
            )
            loss_copy = loss.clone().detach()
            losses.append(loss)
            running_avg_dagger_loss.push(loss_copy)
        loss = torch.stack(losses).mean()
        agent.grad(loss)

        agent.finish_of_episode(episode_no, batch_size)
        episode_no += batch_size
        print("episode_no: ", episode_no)

        if not report:
            continue
        time_2 = datetime.datetime.now()
        time_spent_seconds = (time_2-time_1).seconds
        eps_per_sec = float(episode_no) / time_spent_seconds
        # evaluate
        print("Save Model")
        if agent.run_eval:
            if id_eval_env is not None and episode_no/batch_size % 10 == 0:
                id_eval_res = evaluate_vision_dagger(id_eval_env, agent, num_id_eval_game)
                id_eval_game_points, id_eval_game_step = id_eval_res['average_points'], id_eval_res['average_steps']
            if ood_eval_env is not None and episode_no/batch_size % 10 == 0:
                ood_eval_res = evaluate_vision_dagger(ood_eval_env, agent, num_ood_eval_game)
                ood_eval_game_points, ood_eval_game_step = ood_eval_res['average_points'], ood_eval_res['average_steps']
            if id_eval_game_points >= best_performance_so_far:
                best_performance_so_far = id_eval_game_points
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + ".pt")
        else:
            if running_avg_dagger_loss.get_avg() >= best_performance_so_far:
                best_performance_so_far = running_avg_dagger_loss.get_avg()
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + ".pt")
        print("Save Model end")

        # write accuracies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "time spent seconds": time_spent_seconds,
                         "episodes": episode_no,
                         "episodes per second": eps_per_sec,
                         "loss": str(running_avg_dagger_loss.get_avg())})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()


if __name__ == '__main__':
    train()
