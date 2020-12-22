from tensorboardX import SummaryWriter
import numpy as np


class TensorBoardX():
    def __init__(self, output_dir):
        super(TensorBoardX, self).__init__()
        self.summary_writer = SummaryWriter(log_dir=output_dir)
        self.epoch = 0
        self.eval_num = 0

    def one_epoch(self, train_loss=None, optimizer=None):
        if train_loss:
            self.summary_writer.add_scalar("train/loss", train_loss.item(), self.epoch)
        if optimizer:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
            self.summary_writer.add_scalar("train/lr", lr, self.epoch)

        self.epoch += 1

    def eval(self, id_eval_game_points, id_eval_game_step, ood_eval_game_points, ood_eval_game_step):
        self.summary_writer.add_scalar("eval/id_eval_game_points", id_eval_game_points, self.eval_num)
        self.summary_writer.add_scalar("eval/id_eval_game_step", id_eval_game_step, self.eval_num)
        self.summary_writer.add_scalar("eval/ood_eval_game_points", ood_eval_game_points, self.eval_num)
        self.summary_writer.add_scalar("eval/ood_eval_game_step", ood_eval_game_step, self.eval_num)
        self.eval_num += 1