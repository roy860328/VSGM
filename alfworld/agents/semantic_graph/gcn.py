import torch

class Net(torch.nn.Module):
    """docstring for Net"""
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        