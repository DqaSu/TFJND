import torch
import os
import torch.nn as nn
import utils


class BaseSolver(object):
    def __init__(self, config):
        self.config = config
        self.phase = config.phase

        # GPU verify
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

        # experimental dirs
        self.log_sv_pth = config.save_log_dir
        self.model_sv_pth = config.save_model_dir
        self.log, self.writer = utils.set_save_path(self.log_sv_pth)

        self.is_print_network = config.is_print_network

        self.load_sd = config.load_sd

        self.multiple_gpu = config.multiple_gpu
        self.n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        self.timer = utils.Timer()

        self.model_spec = {}
        self.optimizer_spec = {}

    def train(self):
        pass

    def prepare_training(self):
        pass

    def prepare_test(self):
        pass

    def qa_val(self):
        pass

    def val(self, epoch):
        pass

    def save_checkpoint(self, epoch, epoch_save):
        pass

    def load(self):
        pass

    def print_network(self):
        pass

    def _forward(self, x):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n