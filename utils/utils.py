#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import os
import csv
from ast import arg
import copy
import errno
import torch
import os
import sys
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from models.vae import Acc_Mnist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from ignite.metrics import FID
from pytorch_fid.inception import InceptionV3
import lpips
import matplotlib
matplotlib.use('Agg')


class WrapperInceptionV3(nn.Module):
    """
    Wrapper class of inception (v3) network used for caculating FID.
    """    
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3
    
    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y

def fid_wrapper(num_features=2048, device="cuda"):
    """
    Instantiate an FID object
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[num_features]
    model = InceptionV3([block_idx]).to(device)
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.eval()
    return FID(num_features=num_features, feature_extractor=wrapper_model, device=device)


def append_to_csv(data, filename):
    with open(filename, 'ab') as f:
        # np.savetxt(f, data, delimiter=",")
        np.savetxt(f, data, newline=",")
        f.write(b'\n')

class Logger(object):
    """
    Log class used for recordding the prints 
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample

def get_dataset(name, type='train', download=True, capacity=None, dir='./dataset', verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    # data_name = 'mnist' if name=='mnist28' else name
    if name == 'mnist' or name == 'mnist28':
        dataset_class = datasets.MNIST  # AVAILABLE_DATASETS[data_name]
        dataset_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])
        dataset = dataset_class(dir, train=False if type=='test' else True,  # '{dir}/{name}'.format(dir=dir, name=data_name)
                                download=download, transform=dataset_transform, target_transform=target_transform)
    elif name == 'fashion':
        dataset_class = datasets.FashionMNIST
        dataset_transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
    ])
        dataset = dataset_class(dir, train=False if type=='test' else True,  # '{dir}/{name}'.format(dir=dir, name=data_name)
                                download=download, transform=dataset_transform, target_transform=target_transform)
    elif name == 'svhn':
        dataset_class = datasets.SVHN
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = dataset_class(dir, split=type,  # '{dir}/{name}'.format(dir=dir, name=data_name)
                            transform=dataset_transform, target_transform=target_transform,download=download)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

def get_data_loader(dataset, batch_size, cuda=True, collate_fn=None, drop_last=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''
    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    dataset_ = dataset
    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_, batch_size=batch_size, shuffle=True,
        collate_fn=(collate_fn or default_collate), drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def imshow(dataset_name, data_gen, save_dir, task_id, nrow=10):
    if dataset_name == 'mnist' or dataset_name == 'fashion':
        img_gen = data_gen.cpu().data.view(data_gen.size(0), 1, 32, 32) # for mnist & fashion
    elif dataset_name == 'svhn' or dataset_name == 'cifar10':
        img_gen = data_gen.cpu().data.view(data_gen.size(0), 3, 32, 32) # for svhn & cifar10
    grid_img = torchvision.utils.make_grid(img_gen, nrow=nrow, padding=0)
    plt.imshow(grid_img.permute(1, 2, 0))
    path = os.path.join(save_dir, "Image_task{}.png".format(task_id))
    plt.savefig(path)
    plt.show()

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def caculate_fid(args,task_ID,dataset,labels,decoder):
    """ 
    Calculates Frechet Inception Distance. More details can be found in the Ignite package:
        https://pytorch.org/ignite/generated/ignite.metrics.FID.html
        https://github.com/pytorch/ignite/issues/2423
    """
    print("Testing with FID ...")
    testFID_loader = get_data_loader(
        SubDataset(dataset, labels, target_transform=None),
        args.batch_size,
        cuda=True,
        drop_last=True)
    loss_log = {'FID': 0.0}
    m = fid_wrapper(num_features=2048, device="cuda")
    with torch.no_grad():
        for x, y in testFID_loader:
            y = y.cuda()
            y_onehot = torch.eye(args.class_dim)[y].cuda()
            z = torch.Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).cuda()
            if args.dataset == 'mnist' or args.dataset == 'fashion':
                x = x.cuda().repeat(1, 3, 1, 1)
                x_hat = decoder(z, y_onehot).view(x.size(0), 1, 32, 32).repeat(1, 3, 1, 1)
            elif args.dataset == 'svhn' or args.dataset == 'cifar10' or args.dataset == 'lsun':
                x = x.cuda()
                x_hat = decoder(z, y_onehot).view(x.size(0), 3, 32, 32)
            m.update((x, x_hat))
    loss_log["FID"] = m.compute()
    print('FID: %.3f' % loss_log["FID"])
    append_to_csv([task_ID, loss_log["FID"]],os.path.join(args.modeldir, "fid.csv"))

def caculate_ACC(args,task_ID,train_data,test_data,labels,decoder):
    """
    Calculates the accuracy, where the classifier is evaluated on the real dataset 
    before it is well trained on the generated dateset.
    """
    print("Testing with Acc and rAcc ...")
    trainCLS_loader = get_data_loader(
        SubDataset(train_data, labels, target_transform=None),
        args.batch_size,
        cuda=True,
        drop_last=True)
    testCLS_loader = get_data_loader(
        SubDataset(test_data, labels, target_transform=None),
        args.batch_size,
        cuda=True,
        drop_last=True)
    rCLS = Acc_Mnist(args).cuda()
    optimizer_rCLS = torch.optim.Adam(rCLS.parameters(), lr=args.lr, betas=(0.9, 0.999))
    for epoch in range(5):
        loss_log = {'CLS/loss': 0.0, 'CLS/r_loss': 0.0, 'ACC/FN': 0.0, 'ACC/FP': 0.0, 'ACC/TP': 0.0, 'ACC/TN': 0.0}
        for x, y in trainCLS_loader:
            x = x.cuda().view(x.size(0), -1)
            y = y.cuda()
            y_onehot = torch.eye(args.class_dim)[y].cuda()
            z = torch.Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).cuda()
            x_hat = decoder(z, y_onehot).cuda().view(args.batch_size, -1)
            y_rcls = rCLS(x_hat)
            loss_rcls = torch.nn.CrossEntropyLoss()(y_rcls, y)
            optimizer_rCLS.zero_grad()
            loss_rcls.backward()
            optimizer_rCLS.step()

            loss_log['CLS/r_loss'] += loss_rcls.item()
    num_FN = 0
    num_FP = 0
    rCLS.eval()
    with torch.no_grad():
        for i, data in enumerate(testCLS_loader):
            x = data[0].cuda().view(args.batch_size, -1)
            y = data[1]
            y_FP_soft = rCLS(x)
            y_FP = y_FP_soft.argmax(dim=1).cpu()
            num_FP += (y.data == y_FP).sum()

        acc_FN = num_FN / (args.batch_size * (i + 1)) * 100
        acc_FP = num_FP / (args.batch_size * (i + 1)) * 100
    loss_log['ACC/FP'] = acc_FP.item()
    print('[CLS Epoch%2d]\t LCS/loss: %.3f \t CLS/r_loss: %.3f \t ACC: %.2f%% \t rACC: %.2f%%'
        % (epoch + 1, loss_log['CLS/loss'], loss_log['CLS/r_loss'], loss_log['ACC/FN'], loss_log['ACC/FP'],))
    append_to_csv([task_ID,loss_log["ACC/FN"], loss_log['ACC/FP']],os.path.join(args.modeldir, "ACC.csv"))

def generat_img(args,task_ID,labels,decoder):
    """
    Generate the image with the trained generator.
    """
    print("Testing in the generated figure.") 
    y = torch.tensor(list(range(labels[-1] + 1)))
    z = torch.Tensor(np.random.normal(0, 1, (10 * y.size(0), args.latent_dim))).cuda()
    y = y.repeat(10).reshape(10, y.size(0)).transpose(1, 0).reshape(1, 10 * y.size(0)).squeeze()
    y = torch.eye(args.class_dim)[y].cuda()
    with torch.no_grad():
        data_gen = decoder(z, y)
    imshow(args.dataset,data_gen, args.modeldir, task_ID)
