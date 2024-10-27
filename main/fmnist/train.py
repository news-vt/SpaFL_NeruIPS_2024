import os
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision

from train_argument import parser, print_args

import random
import copy

from time import time
from model import lenet
from utils import *
from Simulator import Simulator
from Split_Data import Non_iid_split
import wandb

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
       
    tr_dataset = torchvision.datasets.FashionMNIST(args.data_root, 
                                    train=True, 
                                    transform=torchvision.transforms.ToTensor(), 
                                    download=True)

    # evaluation during training
    te_dataset = torchvision.datasets.FashionMNIST(args.data_root, 
                                    train=False, 
                                    transform=torchvision.transforms.ToTensor(), 
                                    download=True)    
        
       
    Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
    
    local_tr_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size, 
                                        shuffle = True)
                    for dataset in Non_iid_tr_datasets]
    local_te_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size, 
                                        shuffle = True)
                    for dataset in Non_iid_te_datasets]
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "lenet5":
        model = lenet.LeNet_5_Masked().to(device)

    else:
        print("model not supported")

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
            "th_update": args.th_update
                            },
)
    main(args)