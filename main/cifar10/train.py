import os
import logging
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision

from train_argument import parser, print_args

import random
import copy
import numpy as np
import wandb

from time import time
from model import net
from utils import *
from Simulator import Simulator
from Split_Data import Non_iid_split_cifar, data_stats

def main(args):
    save_folder = args.affix
    
    log_folder = os.path.join(args.log_root, save_folder) #return a new path 
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)


    setattr(args, 'log_folder', log_folder) #setattr(obj, var, val) assign object attribute to its value, just like args.'log_folder' = log_folder
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger) #It prints arguments

       
    tr_dataset = torchvision.datasets.CIFAR10(args.data_root,
                                                train=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Pad(4),
                                            torchvision.transforms.RandomCrop(32),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]  )]),
                                            download = True)
    
    te_dataset = torchvision.datasets.CIFAR10(args.data_root,
                                                train=False,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]   )]),
                                            download = True)  
    num_classes = 10
    
    Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split_cifar(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
    
    client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
    client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)

    while np.min(client_total_samples) < args.batch_size: #if a batch has only one sample, we have an error in BN layers
        print('re-sampling data to satisfy Batch Normalization layer requirements.....', np.min(client_total_samples))
        Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split_cifar(
        10, args.num_clients, tr_dataset, te_dataset, args.alpha)
        client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, 10, args.num_clients)
        client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, 10, args.num_clients)
        
    client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
    client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)

    local_tr_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size, 
                                        shuffle = True, drop_last=True)
                    for dataset in Non_iid_tr_datasets]
    local_te_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size, 
                                        shuffle = True)
                    for dataset in Non_iid_te_datasets]

    print("tr_data counts: ", client_total_samples)
    print("te_data_counts: ", client_total_te_samples) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("currrent device: ", device)


    if args.model =='conv4':
        model = net.masked_Conv4().to(device) 
    elif args.model =='vit':
        model = net.masked_vit(image_size = 32, patch_size = 4, num_classes = 10, dim = 128, depth = 6, heads = 8, mlp_dim = 256,
    dropout = 0, emb_dropout = 0).to(device)
    else:
        print("model not supported")
    
    logger.info(model)

    trainer = Simulator(args, logger, local_tr_data_loaders, local_te_data_loaders, device)
    trainer.initialization(copy.deepcopy(model))
    trainer.FL_loop()


if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    run = wandb.init(
    # Set the project where this run will be logged
        project="SpaFL",
        # Track hyperparameters and run metadata
        config={
            "model": args.model,
            "learning_rate": args.learning_rate,
            "epochs": args.comm_rounds,
            "dirichelt": args.alpha,
            "local_epoch": args.local_epoch,
            "th_update": args.th_update,
            "grad_clip": args.clip
        },
)
    main(args)