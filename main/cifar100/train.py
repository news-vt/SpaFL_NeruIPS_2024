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
import wandb

from time import time
from model import net
from utils import *
from Simulator import Simulator
from Split_data import load_partition_data_cifar100

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


    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_partition_data_cifar100(args.data_root, args.dist,
                                                args.alpha, args.num_clients,
                                                args.batch_size)

    print("tr_data counts: ", train_data_local_num_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("currrent device: ", device)

    if args.model =='resnet18':
        model = net.ResNet18().to(device)
    else:
        print("not supported model")
    logger.info(model)

    trainer = Simulator(args, logger, train_data_local_dict, test_data_local_dict, device)
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