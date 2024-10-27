import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os

path = os.getcwd() #current path
sys.path.append(os.path.abspath(os.path.join(path, os.pardir))) #import the parent directory

from model import binarization


class Client():
    def __init__(self, args, model, loss, client_id, tr_loader, te_loader, device, scheduler = None):
        self.args = args
        self.model = model
        self.loss = loss
        self.scheduler = scheduler
        self.client_id = client_id
        self.tr_loader = tr_loader
        self.te_loader = te_loader
        self.device = device
        self.optimizer = None
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[80, 120], gamma=0.1)
        self.FLOPs = 0
        self.ratio_per_layer = {}
        self.density = 1
        self.num_weights, self.num_thresholds = self.get_model_numbers()
        self.threshold_dict ={}
        self.clip = self.args.clip
    
    def local_training(self, comm_rounds):
        """
        Flow: it freezes parameters or thresholds in a given model by conditioning on _iter
        Return: trained model 
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= self.args.learning_rate * self.args.lr_decay ** comm_rounds, 
                            momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        for epoch in range(1, self.args.local_epoch+1):

            for data, label in self.tr_loader:
                data, label = data.to(self.device), label.to(self.device)
                self.model.train()

                output = self.model(data)
                loss_val = self.loss(output, label)
                
                for name, layer in self.model.named_modules():
                    if isinstance(layer, binarization.MaskedMLP) or isinstance(layer, binarization.MaskedConv2d):     
                        self.ratio_per_layer[name] = layer.ratio   
                        loss_val += self.args.th_coeff * torch.sum(torch.exp(-layer.threshold)) #sigmoid


                self.optimizer.zero_grad()
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                

                if self.scheduler is not None:
                    self.scheduler.step()
                
            

        #Calculate number of FLOPs in this local training 
        self.FLOP_count_weight()
        self.FLOP_count_threshold()

    def local_test(self):

        total_acc = 0.0
        num = 0
        self.model.eval()
        std_loss = 0. 
        iteration = 0.
        with torch.no_grad():
            for data, label in self.te_loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = torch.max(output, dim=1)[1]
                te_acc = (pred.cpu().numpy()== label.cpu().numpy()).astype(np.float32).sum()

                total_acc += te_acc
                num += output.shape[0]

                std_loss += self.loss(output, label)
                iteration += 1
        std_acc = total_acc/num*100.
        std_loss /= iteration
        std_loss = std_loss.cpu().detach().numpy()

        density = self.get_density()
        return std_acc, std_loss, density
        

    def get_model_numbers(self):
        total_weights = 0
        total_thresholds = 0
        for layer in self.model.modules():
            if isinstance(layer, binarization.MaskedMLP) or isinstance(layer, binarization.MaskedConv2d):
                total_weights += layer.weight.numel()
                total_thresholds += layer.threshold.numel()

        return total_weights, total_thresholds

    def get_density(self):
        ratio = 0
        layer_number = 0
        for layer in self.model.modules():
            if isinstance(layer, binarization.MaskedMLP) or isinstance(layer, binarization.MaskedConv2d):
                ratio += layer.ratio
                layer_number += 1
        self.density = (ratio/layer_number).cpu().detach().numpy()

        return (ratio/layer_number).cpu().detach().numpy()
    
    def th_update(self, global_difference):
        with torch.no_grad():
            for name, layer in self.model.named_modules():
                if isinstance(layer, binarization.MaskedConv2d) or isinstance(layer, binarization.MaskedMLP):
                    weight_shape = layer.weight.shape
                    weight = layer.weight
                    weight = weight.view(weight_shape[0], -1)

                    weight_sum_sign = torch.sign(torch.sum(weight, 1))
                    weight_sum_sign = weight_sum_sign.view(weight_shape[0], -1)
                    weight_sum_sign = torch.mul(weight_sum_sign, torch.ones(weight.shape).to(self.device))

                    threshold_dir = global_difference[name].view(weight_shape[0], -1)
                    threshold_dir = torch.mul(threshold_dir, weight_sum_sign)
                    if isinstance(layer, binarization.MaskedConv2d):
                        update_direction = threshold_dir *1/(layer.kernel_size[0]**2)
                    else:
                        update_direction = threshold_dir *1/(layer.in_size)

                    update_direction = update_direction.view(weight_shape)
                    layer.weight += update_direction * (-1)

    def FLOP_count_weight(self): #Conv4
        forward = 0
        for name, layer in self.model.named_modules():
            if isinstance(layer, binarization.MaskedConv2d):
                forward += self.args.local_epoch * len(self.tr_loader.dataset) *layer.ratio * layer.out_channels * layer.in_channels * layer.kernel_size[0]**2 * layer.output_dim**2
            elif isinstance(layer, binarization.MaskedMLP):
                forward += self.args.local_epoch *len(self.tr_loader.dataset) * layer.ratio * layer.in_size * layer.out_size
        backward = 2 * forward
        self.FLOPs += forward + backward
            

    def FLOP_count_threshold(self):             
            self.FLOPs += self.num_thresholds * len(self.tr_loader) + self.num_weights          
