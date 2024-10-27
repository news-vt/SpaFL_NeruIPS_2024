import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from time import time
import numpy as np
import copy
from model import net
from server import Server_Class
from client import Client_Class
from utils import*
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import wandb

class Simulator():
    def __init__(self, args, logger, local_tr_data_loaders, local_te_data_loaders, device):
        self.args = args
        self.logger = logger
        self.Clients_list = None
        self.Clients_list_fedavg = None
        self.Server = None
        self.local_tr_data_loaders = local_tr_data_loaders
        self.local_te_data_loaders = local_te_data_loaders
        self.device = device


    def initialization(self, model):

        loss = nn.CrossEntropyLoss()

        self.Server = Server_Class.Server(self.args, model)
        
        
        self.Clients_list = [Client_Class.Client(self.args, copy.deepcopy(self.Server.initial_model), loss, 
                                    client_id, tr_loader, te_loader, self.device, scheduler=None)
                                    for (client_id, (tr_loader, te_loader)) in enumerate(zip(self.local_tr_data_loaders, self.local_te_data_loaders))]
        

    def FL_loop(self):

        best_acc = 0.
        keep_ratio_at_best_acc = 0.
        best_keep_ratio = 1.
        acc_at_best_keep_ratio = 0.
        acc_history = []
        density_history = []

        for rounds in np.arange(self.args.comm_rounds):
            begin_time = time()
            avg_acc =[]
            avg_loss =[]
            avg_density = []
            self.logger.info("-"*30 + "Epoch start" + "-"*30)

            sampled_clients = self.Server.sample_clients()
            self.Server.broadcast(self.Clients_list, sampled_clients)

            for client_idx in sampled_clients:
                self.Clients_list[client_idx].local_training(rounds)        


            self.Server.aggregation(self.Clients_list, sampled_clients)


            if self.args.th_update == 1:
                for client in self.Clients_list:
                    client.th_update(self.Server.global_difference)

            for client_idx, client in enumerate(self.Clients_list):
                acc, loss, density = client.local_test()
                avg_acc.append(acc), avg_loss.append(loss), avg_density.append(density)


            avg_acc_round = np.mean(avg_acc)
            avg_density_round = np.mean(avg_density)
            avg_loss_round = np.mean(avg_loss)
            acc_history.append(avg_acc_round) #save the current average accuracy to the history
            density_history.append(avg_density_round)

            #                                                                                               avg_density_round, time()-begin_time))
            self.logger.info('round: %d, avg_acc: %.3f, avg_density: %.3f, spent: %.2f' %(rounds, avg_acc_round,
                                                                                                          avg_density_round, time()-begin_time))
            wandb.log({"round": round, "te_accuracy": avg_acc_round, "te_loss": avg_loss_round, "density": avg_density_round})
            

            cur_acc = avg_acc_round
            current_keep_ratio = avg_density_round
            if cur_acc > best_acc:
                best_acc = cur_acc
                keep_ratio_at_best_acc = current_keep_ratio
                print(f"new best acc model with acc: {best_acc} and density: {keep_ratio_at_best_acc}")
                # filename = os.path.join(args.model_folder, 'best_acc_model.pth')
                # save_model(model, filename)
            if  current_keep_ratio < best_keep_ratio:
                best_keep_ratio = current_keep_ratio
                acc_at_best_keep_ratio = cur_acc
                print(f"new most sparse model with acc: {acc_at_best_keep_ratio} and density: {best_keep_ratio}")

                # filename = os.path.join(args.model_folder, 'best_keepratio_model.pth')
                # save_model(model, filename)

            if rounds==140:
                self.pattern()

        
        avg_total_FLOPS = self.FLOP_cal()

        self.logger.info(">>>>> Training process finish")
        self.logger.info("Best keep ratio {:.4f}, acc at best keep ratio {:.4f}".format(best_keep_ratio, acc_at_best_keep_ratio))
        self.logger.info("Best acc {:.4f}, keep ratio at best acc {:.4f}".format(best_acc, keep_ratio_at_best_acc))
        
        # filename = os.path.join(self.args.model_folder, 'final_model.pth')
        # save_model(model, filename)

        wandb.log({"average FLOPs": avg_total_FLOPS})
        wandb.log({"best_keep_ratio": best_keep_ratio, "accuracy":acc_at_best_keep_ratio})
        wandb.log({"best_acc":best_acc, "density":keep_ratio_at_best_acc} )


        self.logger.info(">>>>> Accuracy history during training")
        self.logger.info(acc_history)
        self.logger.info(">>>>> Density history during training")
        self.logger.info(density_history)
        self.logger.info(">>>>> Average FLOPs")
        self.logger.info(avg_total_FLOPS)
        
        clients_sparsity_list = self.get_clients_sparsity()
        self.logger.info(">>>>> Client sparsity list")
        self.logger.info(clients_sparsity_list)


        # self.pattern4()
        # self.pattern()

    def get_sparsity(self, round):
        avg_density = 0 
        for client_idx, client in enumerate(self.Clients_list):
            density = print_layer_keep_ratio(self.Clients_list[client_idx].model, round, client_idx, self.logger)
            avg_density += density
        avg_density_round = avg_density/self.args.num_clients
        self.logger.info('round: %d, avg_density: %.4f' %(round, avg_density_round))

    def FLOP_cal(self):
        FLOPs = 0
        for client in self.Clients_list:
            FLOPs += client.FLOPs
        FLOPs *= 1/self.args.num_clients

        return FLOPs
    
    def FLOP_cal_fedavg(self):
        FLOPs = 0
        for client in self.Clients_list_fedavg:
            FLOPs += client.FLOPs
        FLOPs *= 1/self.args.num_clients            
        
        return FLOPs
    
    def get_clients_sparsity(self):
        clients_sparsity_list = []

        for client in self.Clients_list:
            density = client.get_density()
            sparsity = 1 - density
            clients_sparsity_list.append(sparsity)

        return clients_sparsity_list

    def pattern(self):
        pattern_dict = []
        for client_idx, client in enumerate(self.Clients_list):
            filter_list = []
            mask = client.model.conv1.mask
            for filter in mask:
                if filter[0][0][0] == 1:
                    filter_list.append(1)
                else:
                    filter_list.append(0)
            pattern_dict.append(filter_list)
        with open(f'pattern_conv_1_{self.args.alpha}.json', 'w') as json_file:
            json.dump(pattern_dict, json_file)


