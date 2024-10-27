import logging
import math

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from datasets import CIFAR100_truncated





def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(100):
            if i in unq:
                tmp.append( unq_cnt[np.argwhere(unq==i)][0,0])
            else:
                tmp.append(0)
        net_cls_counts.append ( tmp)
    # logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts



def _data_transforms_cifar100():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar100_data(datadir):
    train_transform, test_transform = _data_transforms_cifar100()

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data( datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "dir":
        min_size = 0
        K = 100
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


def get_dataloader_CIFAR100(datadir, train_bs, test_bs, dataidxs=None,test_idxs=None, cache_train_data_set=None,cache_test_data_set=None):
    transform_train, transform_test = _data_transforms_cifar100()

    dataidxs=np.array(dataidxs)
    # rand_perm = np.random.permutation(len(dataidxs))
    # train_dataidxs=dataidxs[rand_perm[:int(len(dataidxs) * 0.8)]]

    logging.info("train_num{}  test_num{}".format(len(dataidxs),len(test_idxs)))
    train_ds = CIFAR100_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True,cache_data_set=cache_train_data_set)
    # test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True,cache_data_set=cache_test_data_set)
    test_ds = CIFAR100_truncated(datadir, dataidxs=test_idxs, train=False, transform=transform_test, download=True,
                      cache_data_set=cache_test_data_set)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True, drop_last=False)
    # logging.info("train_loader{}  test_loader{}".format(len(train_dl), len(test_dl)))
    return train_dl, test_dl




def load_partition_data_cifar100( data_dir, partition_method, partition_alpha, client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)
    # class_num = len(np.unique(y_train))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    #
    # train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))
    # test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_cifar100()

    cache_train_data_set=CIFAR100(data_dir, train=True, transform=transform_train, download=True)
    cache_test_data_set = CIFAR100(data_dir, train=False, transform=transform_test, download=True)
    idx_test = [[] for i in range(100)]
    # checking
    for label in range(100):
        idx_test[label] = np.where(y_test == label)[0]
    test_dataidxs = [[] for i in range(client_number)]
    for client_idx in range(client_number):
        for label in range(100):
            # each has 100 pieces of testing data
            label_num = math.ceil(traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * 100)
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]]))
        dataidxs = net_dataidx_map[client_idx]
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader_CIFAR100( data_dir, batch_size, batch_size,
                                                 dataidxs,test_dataidxs[client_idx] ,cache_train_data_set=cache_train_data_set,cache_test_data_set=cache_test_data_set)
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    # test= [0 for i in range(10000)]
    # for idx in test_dataidxs:
    #     for value in idx:
    #         test[value]+=1
    # print(np.count_nonzero(test))
    return None, None, None, None, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts
