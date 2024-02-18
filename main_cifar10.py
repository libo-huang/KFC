#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import random
import tqdm
import copy
import errno
import torch
import os
import sys
import numpy as np
import argparse
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from models.vae import ConditionalVAE_cifar
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import torch.autograd as autograd
from utils.utils import append_to_csv, fid_wrapper
import matplotlib
matplotlib.use('Agg')
import time


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
    elif name == 'cifar10':
        dataset_class = datasets.CIFAR10
        dataset_transform = transforms.Compose([transforms.Resize(178),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.CenterCrop(148),
                                        transforms.Resize(64),
                                        transforms.ToTensor(),])
        
        dataset = dataset_class(dir, train= (type=='train') ,  # '{dir}/{name}'.format(dir=dir, name=data_name)
                            transform=dataset_transform,target_transform=target_transform,download=download)
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

def compute_gradient_penalty(dataset,D, real_samples, fake_samples, syn_label):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor
    if dataset == 'mnist' or dataset == 'fashion':
        alpha = Tensor(np.random.random((real_samples.size(0), 1))) # for fashin & mnist
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False) # for fashin & mnist
    elif dataset == 'svhn' or dataset == 'cifar10' :
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))) # for svhn & cifar10
        fake = Variable(Tensor(real_samples.shape[0]).fill_(1.0), requires_grad=False)  # for svhn & cifar10
    #real_samples = real_samples.reshape(fake_samples.shape) # fashion && conv
    #print('alpha.shape: ',alpha.shape,'real_samples.shape: ',real_samples.shape, 'fake_samples.shape: ',fake_samples.shape)
    # Get random interpolation between real and fake samples
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, syn_label)    
    # Get gradient w.r.t. interpolates
    gradients = \
        autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True,
                      retain_graph=True,
                      only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

def imshow(dataset_name, data_gen, save_dir, task_id, nrow=10):
    """
    Function used to show a set of images
    """
    if dataset_name == 'mnist' or dataset_name == 'fashion':
        img_gen = data_gen.cpu().data.view(data_gen.size(0), 1, 32, 32) # for mnist & fashion
    elif dataset_name == 'svhn' or dataset_name == 'cifar10':
        # img_gen = data_gen.cpu().data.view(data_gen.size(0), 3, 32, 32) # for svhn & cifar10
        img_gen = data_gen.cpu().data.view(data_gen.size(0), 3, 64, 64)
    grid_img = torchvision.utils.make_grid(img_gen, nrow=nrow, padding=0, normalize = True)
    plt.imshow(grid_img.permute(1, 2, 0))
    path = os.path.join(save_dir, "Image_task{}.png".format(task_id))
    plt.savefig(path)
    path = os.path.join(save_dir, "Image_task{}_norm.png".format(task_id))
    plt.savefig(path)
    # plt.show()

def draw_cifar(data_gen, save_dir, task_id, name):
    """
    Function used to show a set of cifar-10 images
    """
    img_gen = data_gen.cpu().data  # .view(data_gen.size(0), 3, 32, 32)  # for svhn & cifar10
    grid_img = torchvision.utils.make_grid(img_gen, nrow=10, padding=0)
    plt.imshow(grid_img.permute(1, 2, 0))
    path = os.path.join(save_dir, "IMG_{}{}.png".format(name, task_id))
    plt.savefig(path)
    plt.show()

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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


def main_LGLvKR(args):
    """ 
    Main procedure of Lifelong Generative Learning via Knowledge Reconstruction (LGLvKR)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Prepare sequential tasks
    print(75 * '=' + '\n' + '|| \t model dir:\t%s\t ||\n' % args.modeldir + 75 * '=')
    permutation = np.array(list(range(10)))
    target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
    data_train = get_dataset(args.dataset, type="train", dir=args.data_dir, target_transform=target_transform, verbose=False)
    data_test = get_dataset(args.dataset, type="test", dir=args.data_dir, target_transform=target_transform, verbose=False)
    labels_per_task = [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    labels_per_task_test = [list(np.array(range(labels_per_task[i][-1] + 1))) for i in range(args.tasks)]

    # Built logger class
    sys.stdout = Logger(os.path.join(args.modeldir, 'log_{}_{}.txt'.format(args.name,args.op)))
    
    # Main sequential training
    time_cost = 0
    for task_i in range(args.f_task,args.tasks):
        print(40 * '=' + ' Task %1d ' % (task_i) + 40 * '=')
        if not os.path.exists(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i)):
            # train on each task i if there does not exist trained model i
            train_loader = get_data_loader(
                SubDataset(data_train, labels_per_task[task_i], target_transform=None),
                args.batch_size,
                cuda=True,
                drop_last=True)
            print("training(n=%5d)..." % len(train_loader.dataset))
            if task_i == 0:
                cvae = ConditionalVAE_cifar(args)
                print("there are {} params with {} elems in the cvae".format(
                    len(list(cvae.parameters())), sum(p.numel() for p in cvae.parameters() if p.requires_grad))
                )
            else:
                cvae = torch.load(os.path.join(args.modeldir, args.name + '_%1d.pth' % (task_i - 1)))
                cvae_old = deepcopy(cvae)
                cvae_old.eval()
                cvae_old = freeze_model(cvae_old)
                print("there are {} params with {} elems in the cvae, {} params with {} elems in the cvae_old".format(
                    len(list(cvae.parameters())), sum(p.numel() for p in cvae.parameters() if p.requires_grad),
                    len(list(cvae_old.parameters())), sum(p.numel() for p in cvae_old.parameters() if p.requires_grad))
                )
            cvae = cvae.cuda()
            optimizer_CVAE = torch.optim.Adam(cvae.parameters(), lr=args.lr, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_CVAE, gamma=0.95)
            start = time.time()
            for epoch in range(args.epochs):
                # Main loops with several epochs
                loss_log = {'V/loss': 0.0, 'V/loss_rec': 0.0, 'V/loss_var': 0.0, 'V/loss_var_hat': 0.0, 'V/loss_aug': 0.0}
                for x, y in train_loader:
                    # train on each batch dataset
                    x = x.cuda()
                    y = torch.eye(args.class_dim)[y].cuda()
                    if task_i == 0:
                        # train with conventional VAE at the initial task since there is no knowledge needed to remember
                        x_rec, mu, logvar = cvae(x, y)
                        _, mu_hat, logvar_hat = cvae(x_rec, y)

                        loss_rec = torch.nn.MSELoss(reduction='sum')(x, x_rec) / x.size(0)
                        loss_var = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
                        loss_var_hat = (-0.5 * torch.sum(1 + logvar_hat - mu_hat ** 2 - logvar_hat.exp())) / x_rec.size(0)
                        loss_cvae = loss_rec + args.alpha_var * loss_var + args.alpha_var_hat * loss_var_hat

                    else:
                        # train with LGLvKR when the task is not the initial one
                        _, mu, logvar = cvae(x, y)
                        z = cvae.reparameterize(mu, logvar)

                        y_pre = torch.randint(0, labels_per_task[task_i][0], (args.batch_size,))
                        y_pre = torch.eye(args.class_dim)[y_pre].cuda()
                        z_pre = torch.empty((args.batch_size, args.latent_dim)).normal_(mean=0, std=1).cuda()
                        z_pre_ = torch.cat([z_pre, y_pre], dim=1)
                        xPre_old = cvae_old.decode(z_pre_)

                        z_merged = torch.cat((z_pre, z))
                        y_merged = torch.cat((y_pre, y))
                        z_merged_ = torch.cat([z_merged, y_merged], dim=1)
                        xRec_merged = cvae.decode(z_merged_)
                        _, mu_hat, logvar_hat = cvae(xRec_merged[:args.batch_size], y)

                        loss_rec = torch.nn.MSELoss(reduction='sum')(x, xRec_merged[args.batch_size:]) / x.size(0)
                        loss_var = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
                        loss_var_hat = (-0.5 * torch.sum(1 + logvar_hat - mu_hat ** 2 - logvar_hat.exp())) / x.size(0)
                        loss_aug = torch.nn.MSELoss(reduction='sum')(xPre_old, xRec_merged[:args.batch_size]) / xPre_old.size(0)
                        loss_aug = loss_aug * task_i

                        loss_cvae = loss_rec + args.alpha_var * loss_var + args.alpha_var_hat * loss_var_hat + args.alpha_aug * loss_aug
                    optimizer_CVAE.zero_grad()
                    loss_cvae.backward()
                    optimizer_CVAE.step()

                    loss_log['V/loss'] += loss_cvae.item()
                    loss_log['V/loss_rec'] += loss_rec.item()
                    loss_log['V/loss_var'] += loss_var.item() * args.alpha_var
                    loss_log['V/loss_var_hat'] += loss_var_hat.item() * args.alpha_var_hat
                    loss_log['V/loss_aug'] += loss_aug.item() * args.alpha_aug if task_i != 0 else 0
                scheduler.step()
                print('[VAE Epoch%2d]\t V/loss: %.3f\t V/loss_rec: %.3f\t V/loss_var: %.3f\t V/loss_var_hat: %.3f\t V/loss_aug: %.3f'
                      % (epoch + 1,
                         loss_log['V/loss'],
                         loss_log['V/loss_rec'],
                         loss_log['V/loss_var'],
                         loss_log['V/loss_var_hat'],
                         loss_log['V/loss_aug']))
            time_cost += time.time() - start
            torch.save(cvae, os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i))
        else:
            # load model i if we already trained on task i
            cvae = torch.load(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i))

        # Evaluate with different matrics
        cvae.eval()
        ##################### Test with FID #########################################################################
        """ 
        Calculates Frechet Inception Distance. More details can be found in the Ignite package:
        https://pytorch.org/ignite/generated/ignite.metrics.FID.html
        https://github.com/pytorch/ignite/issues/2423
        """
        if args.fid and task_i == args.tasks - 1:
            print("Testing with FID ...")
            m_r = fid_wrapper(num_features=2048, device="cuda")  # cuda with 12s while cpu with 1460s
            m_g = fid_wrapper(num_features=2048, device="cuda")

            test_loader = get_data_loader(
                SubDataset(data_test, labels_per_task_test[task_i], target_transform=None),
                args.batch_size,
                cuda=True,
                drop_last=True)
            loss_log = {'FID_g': 0.0, 'FID_r': 0.0}
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()  # too small to use FID with default parameters
                    y_onehot = torch.eye(args.class_dim)[y].cuda()
                    z = torch.Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).cuda()
                    x_g = cvae.decode(torch.cat([z, y_onehot], dim=1))  # .view(x.size(0), 3, 32, 32)
                    x_r, _, _ = cvae(x, y_onehot)
                    m_g.update((x, x_r))
                    m_r.update((x, x_g))
            loss_log['FID_g'] = m_g.compute()
            loss_log['FID_r'] = m_r.compute()
            print('FID_g: {:.3f} \t FID_r: {:.3f}'.format(loss_log['FID_g'], loss_log['FID_r']))
            append_to_csv([task_i, loss_log["FID_g"], loss_log["FID_r"]], os.path.join(args.modeldir, "fid.csv"))
        print(100 * '-')
    print('Total training time is: %.3f seconds' % time_cost)

parser = argparse.ArgumentParser(description='')

common = parser.add_argument_group("common parameters group")
network = parser.add_argument_group("network parameters group")
train = parser.add_argument_group("training parameters group")
common.add_argument('-modeldir', default='./checkpoints/cifar10/', help='')
common.add_argument('-dataset',type=str, default='cifar10', help='dataset name')
common.add_argument('-data_dir', type=str, default='./dataset/cifar10', help='data directory')  # TODO
common.add_argument('-tasks', type=int, default=4, help='number of tasks')
common.add_argument('-f_task', type=int, default=0, help='number of tasks')
common.add_argument('-batch_size', type=int, default=64, help='batch size')
network.add_argument('-feat_dim', type=int, default=32*32*3, help='input features dimension')
network.add_argument('-latent_dim', type=int, default=128, help='latent variable dimension')
network.add_argument('-class_dim', type=int, default=10, help='class or one-hot label dimension')
network.add_argument('-hidden_dim', type=int, default=256, help='hidden dimension')
train.add_argument('-lr', type=float, default=0.005, help='learning rate')
train.add_argument('-alpha_var', type=float, default=0.1, help='alpha parameter for variational loss')
train.add_argument('-alpha_var_hat', type=float, default=0, help="alpha parameter for variational loss of reconstructed data")
train.add_argument('-alpha_aug', type=float, default=10., help="alpha parameter for the augmented loss")
train.add_argument('-epochs', default=1, type=int, metavar='N', help='number of epochs')  # TODO
train.add_argument("-gpu", type=str, default='2', help='which gpu to use')
train.add_argument("-name", type=str, default='4tasks', help='the name of the temporary saved model')
train.add_argument("-seed", type=int, default=1024, help='')
train.add_argument('-method',type=str,default='KFC', choices=['KFC_Fine','KFC_joint','KFC','CVAE_Dy_','CVAE_Dy_Fine','CVAE_Dy_joint'])

common.add_argument('-op',type=str, default='tarin', choices= ['train','eval_'])
common.add_argument('-fid', default=False, action = 'store_true', help='Whether fid needs to be calculated')
common.add_argument('-generate', default=False, action='store_true', help = 'Whether imgs need to be generated')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.method == 'KFC':
    args.modeldir = args.modeldir + '0_KFC/' + '{}epoch'.format(args.epochs)
    main_LGLvKR(args)