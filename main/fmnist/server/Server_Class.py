import copy
from collections import OrderedDict

import torch
import numpy as np
from numpy import random

class Server():

    def __init__(self, args, model):
        self.clients_list = np.arange(args.num_clients)
        self.args = args
        self.initial_model = model
        self.global_thresholds = OrderedDict()
        self.set_global_thresholds()
        self.global_difference = OrderedDict()


    def set_global_thresholds(self):
        if self.args.model == 'lenet5':
            for key in self.initial_model.state_dict().keys():
                if key == 'conv1.threshold':
                    self.global_thresholds[key] = self.initial_model.state_dict()[key]
                elif key == 'conv2.threshold':
                    self.global_thresholds[key] = self.initial_model.state_dict()[key]
                elif key == 'fc3.threshold':
                    self.global_thresholds[key] = self.initial_model.state_dict()[key]
                elif key == 'fc4.threshold':
                    self.global_thresholds[key] = self.initial_model.state_dict()[key]

    def sample_clients(self):
        """
        Return: array of integers, which corresponds to the indices of sampled deviecs
        """
        sampling_set = np.random.choice(self.args.num_clients, self.args.schedulingsize, replace = False)

        return sampling_set
    
    def broadcast(self, Clients_list, Clients_list_idx):
        """
        Input: a list of Client class
        Flow: Set the current global thresholds to every client
        """
        for idx in Clients_list_idx:
            client = Clients_list[idx]
            with torch.no_grad():
                for key in self.global_thresholds:
                    if key =='conv1.threshold':
                        client.model.conv1.threshold.copy_(self.global_thresholds[key])
                    if key =='conv2.threshold':
                        client.model.conv2.threshold.copy_(self.global_thresholds[key])
                    if key =='fc3.threshold':
                        client.model.fc3.threshold.copy_(self.global_thresholds[key])
                    if key =='fc4.threshold':
                        client.model.fc4.threshold.copy_(self.global_thresholds[key])


    def aggregation(self, Clients_list, sampling_set):
        """
        Input: sampling_set: array of integers, which corresponds to the indices of sampled devices and a list of Client class
        Flow: aggregate the updated threholds in the sampling set
        """
        threshold_dict = OrderedDict()
        for i, client in enumerate(sampling_set):
            local_model = Clients_list[client].model.state_dict()

            if i == 0:
                for key in self.global_thresholds:
                    threshold_dict[key] = local_model[key] * 1/self.args.schedulingsize
            else:
                for key in self.global_thresholds:
                    threshold_dict[key] += local_model[key] *1/self.args.schedulingsize

        for key in self.global_thresholds:
            self.global_difference[key] = threshold_dict[key] - self.global_thresholds[key]
        
        self.global_thresholds = threshold_dict
        

