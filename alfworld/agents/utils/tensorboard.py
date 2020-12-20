from tensorboardX import SummaryWriter
import numpy as np


class TensorBoardX():
    def __init__(self, output_dir):
        super(TensorBoardX, self).__init__()
        self.summary_writer = SummaryWriter(log_dir=output_dir)
        self.epoch = 0

    def one_epoch(self, train_loss=None, optimizer=None):
        if train_loss:
            self.summary_writer.add_scalar("train/loss", train_loss.item(), self.epoch)
        if optimizer:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
            self.summary_writer.add_scalar("train/lr", lr, self.epoch)

        self.epoch += 1