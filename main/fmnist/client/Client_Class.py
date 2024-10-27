import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
import math

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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= self.args.learning_rate, 
                            momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.FLOPs = 0
        # self.ratio_per_layer = {}
        self.density = 1.0
        self.num_weights, self.num_thresholds = self.get_model_numbers()
        self.zero_list = [0]*self.args.comm_rounds

    
    def local_training(self, comm_rounds):

        for epoch in range(1, self.args.local_epoch+1):

            for data, label in self.tr_loader:
                data, label = data.to(self.device), label.to(self.device)
                self.model.train()

                output = self.model(data)
                loss_val = self.loss(output, label)

                for name, layer in self.model.named_modules():
                    if isinstance(layer, binarization.MaskedMLP) or isinstance(layer, binarization.MaskedConv2d):     

                        loss_val += self.args.th_coeff *1/(1+np.exp(5 - 10*comm_rounds/self.args.penalty_scheduler)) * torch.sum(torch.exp(-layer.threshold)) #sigmoid


                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                

                if self.scheduler is not None:
                    self.scheduler.step()
                
        zero_count = 0
        for name, layer in self.model.named_modules():
            if isinstance(layer, binarization.MaskedMLP) or isinstance(layer, binarization.MaskedConv2d):   
                zero_count += layer.zero_count
        self.zero_list[comm_rounds] = zero_count
        
        
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
        ratio = 0. 
        layer_number = 0
        for layer in self.model.modules():
            if isinstance(layer, binarization.MaskedMLP) or isinstance(layer, binarization.MaskedConv2d):
                ratio += layer.ratio
                layer_number += 1
        self.density = (ratio/layer_number).cpu().detach().numpy()

        return (ratio/layer_number).cpu().detach().numpy()
        # return (self.density if type(self.density) == float else self.density.cpu().detach().numpy())
    
    def th_update(self, global_difference):
        with torch.no_grad():
            for key in global_difference:
                if key == 'conv1.threshold':
                    weight_shape = self.model.conv1.weight.shape
                    weight = self.model.conv1.weight
                    weight = weight.view(weight_shape[0], -1)
                    
                    weight_sum_sign = torch.sign(torch.sum(weight, 1))
                    weight_sum_sign = weight_sum_sign.view(weight_shape[0], -1) #match dim with threhsold difference
                    weight_sum_sign = torch.mul(weight_sum_sign, torch.ones(weight.shape).to(self.device)) #allocates direction fo weights
                    
                    threshold_dir = global_difference[key].view(weight_shape[0],-1)
                    threshold_dir = torch.mul(threshold_dir, weight_sum_sign)
                    update_direction = threshold_dir * 1/(self.model.conv1.kernel_size[0]**2)
                    update_direction = update_direction.view(weight_shape)
                    self.model.conv1.weight += update_direction * (-1)


                if key == 'conv2.threshold':

                    weight_shape = self.model.conv2.weight.shape
                    weight = self.model.conv2.weight
                    weight = weight.view(weight_shape[0], -1)
                    
                    weight_sum_sign = torch.sign(torch.sum(weight, 1))
                    weight_sum_sign = weight_sum_sign.view(weight_shape[0], -1) #match dim with threhsold difference
                    weight_sum_sign = torch.mul(weight_sum_sign, torch.ones(weight.shape).to(self.device)) #allocates direction fo weights
                    
                    threshold_dir = global_difference[key].view(weight_shape[0],-1)
                    threshold_dir = torch.mul(threshold_dir, weight_sum_sign)
                    update_direction = threshold_dir * 1/(self.model.conv2.kernel_size[0]**2)
                    update_direction = update_direction.view(weight_shape)
                    self.model.conv2.weight += update_direction * (-1)


                if key == 'fc3.threshold':
                    weight_shape = self.model.fc3.weight.shape
                    weight = self.model.fc3.weight
                    threshold = global_difference[key].view(weight_shape[0], -1)
                    
                    weight_sum_sign = torch.sign(torch.sum(weight, 1))
                    weight_sum_sign = weight_sum_sign.view(weight_shape[0], -1)
                    weight_sum_sign = torch.mul(weight_sum_sign, torch.ones(weight.shape).to(self.device))

                    threshold_dir = global_difference[key].view(weight_shape[0], -1)
                    threshold_dir = torch.mul(threshold_dir, weight_sum_sign )
                    update_direction = threshold_dir *1/(self.model.fc3.in_size)
                    self.model.fc3.weight += update_direction *(-1)

                if key == 'fc4.threshold':

                    weight_shape = self.model.fc4.weight.shape
                    weight = self.model.fc4.weight
                    threshold = global_difference[key].view(weight_shape[0], -1)
                    
                    weight_sum_sign = torch.sign(torch.sum(weight, 1))
                    weight_sum_sign = weight_sum_sign.view(weight_shape[0], -1)
                    weight_sum_sign = torch.mul(weight_sum_sign, torch.ones(weight.shape).to(self.device))

                    threshold_dir = global_difference[key].view(weight_shape[0], -1)
                    threshold_dir = torch.mul(threshold_dir, weight_sum_sign )
                    update_direction = threshold_dir *1/(self.model.fc4.in_size)
                    self.model.fc4.weight += update_direction *(-1)

    def FLOP_count_weight(self):
        forward = 0
        for name, layer in self.model.named_modules():
            if isinstance(layer, binarization.MaskedConv2d):
                forward += self.args.local_epoch * len(self.tr_loader.dataset) *layer.ratio * layer.out_channels * layer.in_channels * layer.kernel_size[0]**2 * layer.output_dim**2
            elif isinstance(layer, binarization.MaskedMLP):
                forward += self.args.local_epoch *len(self.tr_loader.dataset) * layer.ratio * layer.in_size * layer.out_size
        backward = 2 * forward
        self.FLOPs += forward + backward


    def FLOP_count_threshold(self):
        self.FLOPs += self.num_thresholds * len(self.tr_loader)  + self.num_weights

    def print_cuda_memory(self):
    # Get current GPU memory usage
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        self.used_memory += current_memory
        self.peak_memory = peak_memory
        print(f"Current CUDA memory usage: {current_memory / (1024 ** 2):.2f} MB")
        print(f"Peak CUDA memory usage: {peak_memory / (1024 ** 2):.2f} MB")