import copy
import os
import sys
from collections import OrderedDict

import torch
import numpy as np
from numpy import random
path = os.getcwd() #current path
sys.path.append(os.path.abspath(os.path.join(path, os.pardir))) #import the parent directory

from model import binarization

class Server():

    def __init__(self, args, model):
        self.clients_list = np.arange(args.num_clients)
        self.args = args
        self.initial_model = model
        self.global_thresholds = OrderedDict()
        self.set_global_thresholds()
        self.global_difference = OrderedDict()


    def set_global_thresholds(self):
        for name, layer in self.initial_model.named_modules():
            if isinstance(layer, binarization.MaskedConv2d) or isinstance(layer, binarization.MaskedMLP):
                self.global_thresholds[name] = layer.threshold


    def sample_clients(self):
        """
        Return: array of integers, which corresponds to the indices of sampled deviecs
        """
        sampling_set = np.random.choice(self.args.num_clients, self.args.schedulingsize, replace = False)

        return sampling_set
    
    def broadcast(self, Clients_list, Clients_list_idx = None):
        """
        Input: a list of Client class
        Flow: Set the current global thresholds to every client
        """
        for idx in Clients_list_idx:
            client = Clients_list[idx]
            with torch.no_grad():
                for name, layer in client.model.named_modules():
                    if isinstance(layer, binarization.MaskedConv2d) or isinstance(layer, binarization.MaskedMLP):
                        layer.threshold.copy_(self.global_thresholds[name])


    def aggregation(self, Clients_list, sampling_set):
        """
        Input: sampling_set: array of integers, which corresponds to the indices of sampled devices and a list of Client class
        Flow: aggregate the updated threholds in the sampling set
        """
        threshold_dict = OrderedDict()
        for i, client in enumerate(sampling_set):
            local_model = Clients_list[client].model.state_dict()
            for name, layer in Clients_list[client].model.named_modules():
                if isinstance(layer, binarization.MaskedConv2d) or isinstance(layer, binarization.MaskedMLP):
                    if i == 0:
                        threshold_dict[name] = layer.threshold * 1/self.args.schedulingsize
                    else:
                        threshold_dict[name] += layer.threshold *1/self.args.schedulingsize

        for key in self.global_thresholds:
            self.global_difference[key] = threshold_dict[key] - self.global_thresholds[key]
        
        self.global_thresholds = threshold_dict
            
