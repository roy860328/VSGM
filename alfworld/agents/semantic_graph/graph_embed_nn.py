import importlib


class EmbedGlobalGraph(object):
    """docstring for EmbedGlobalGraph"""
    def __init__(self, cfg):
        super(EmbedGlobalGraph, self).__init__()
        self.cfg = cfg
        self.model = importlib.import_module(cfg.SCENE_GRAPH.MODEL).Net(cfg)