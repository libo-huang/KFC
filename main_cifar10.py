#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import random
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
from ignite.metrics import FID
from utils.utils import append_to_csv
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

def compute_fisher_info_cgan(args, generator, discriminator, generator_old_para, optimizer_G, optimizer_D, dataloader, fisher_pre):
    fisher = {n: torch.zeros(p.shape).cuda() for n, p in generator.named_parameters()
              if p.requires_grad}
    # n_samples_batches = (num_samples // dataloader.batch_size + 1) if num_samples > 0 \
    #         else (len(dataloader.dataset) // dataloader.batch_size) # number of samples to compute fisher matrix in each batch
    generator.eval()
    discriminator.eval()
    # old_batch_size = dataloader.batch_size
    # dataloader.batch_size = 1
    for i, data in enumerate(dataloader, 0):
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        x = data[0].cuda().view(data[0].size(0), -1)
        y = data[1]
        y_onehot = torch.eye(args.class_dim)[y].cuda()
        # print(x.shape, y_onehot.shape)
        # exit()
        for j in range(x.shape[0]):
            x_ = x[j].unsqueeze(0)
            y_onehot_ = y_onehot[j].unsqueeze(0)
            # print(x_.shape, y_onehot_.shape)
            # exit()
            valid = Variable(torch.cuda.FloatTensor(1, 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.cuda.FloatTensor(1, 1).fill_(0.0), requires_grad=False)
            # valid = Variable(torch.cuda.FloatTensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(torch.cuda.FloatTensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

            z_g = torch.normal(0, 1, (x_.shape[0], args.latent_dim)).cuda()
            x_fake = generator(z_g, y_onehot_)
            fake_validity = discriminator(x_fake, y_onehot_)
            g_loss = torch.nn.MSELoss()(fake_validity, valid)

            # loss_ewc = torch.tensor(0)
            # if fisher_pre != None:
            #     loss_ewc = compute_ewc_loss(generator,generator_old_para,fisher_pre)
            #     g_loss += args.alpha_ewc * loss_ewc

            # torch.backends.cudnn.enabled = False
            g_loss.backward()

            for n, p in generator.named_parameters():
                if p.grad is not None:
                    # print('p.grad',p.grad)
                    fisher[n] += p.grad.pow(2) / int(len(dataloader))
                    # n_samples = len(dataloader)
    fisher = {n: p for n, p in fisher.items()}
    # dataloader.batch_size = old_batch_size
    return fisher
def compute_ewc_loss(model, model_old_para, fisher):
    ewc_loss = 0
    for n, p in model.named_parameters():
        _loss = fisher[n] * (p - model_old_para[n]) ** 2
        ewc_loss += _loss.sum()
    return ewc_loss

class Logger(object):
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

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(75 * '=' + '\n' + '|| \t model dir:\t%s\t ||\n' % args.modeldir + 75 * '=')
    permutation = np.array(list(range(10)))
    target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
    mnist_train = get_dataset(args.dataset, type="train", dir=args.data_dir, target_transform=target_transform, verbose=False)
    mnist_test = get_dataset(args.dataset, type="test", dir=args.data_dir, target_transform=target_transform, verbose=False)
    labels_per_task = [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    labels_per_task_test = [list(np.array(range(labels_per_task[i][-1] + 1))) for i in range(args.tasks)]

    sys.stdout = Logger(os.path.join(args.modeldir, 'log_{}_{}.txt'.format(args.name,args.op)))
    # training
    time_cost = 0
    for task_i in range(args.f_task,args.tasks):
        print(40 * '=' + ' Task %1d ' % (task_i) + 40 * '=')
        if not os.path.exists(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i)):
            train_loader = get_data_loader(
                SubDataset(mnist_train, labels_per_task[task_i], target_transform=None),
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
                loss_log = {'V/loss': 0.0, 'V/loss_rec': 0.0, 'V/loss_var': 0.0, 'V/loss_var_hat': 0.0, 'V/loss_aug': 0.0}
                for x, y in train_loader:
                    x = x.cuda()
                    y = torch.eye(args.class_dim)[y].cuda()
                    if task_i == 0:
                        x_rec, mu, logvar = cvae(x, y)
                        _, mu_hat, logvar_hat = cvae(x_rec, y)

                        loss_rec = torch.nn.MSELoss(reduction='sum')(x, x_rec) / x.size(0)
                        loss_var = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
                        loss_var_hat = (-0.5 * torch.sum(1 + logvar_hat - mu_hat ** 2 - logvar_hat.exp())) / x_rec.size(0)
                        loss_cvae = loss_rec + args.alpha_var * loss_var + args.alpha_var_hat * loss_var_hat

                    else:
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
            cvae = torch.load(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i))

        cvae.eval()
        ##################### Test with FID #########################################################################
        if args.fid and task_i == args.tasks - 1:
            print("Testing with FID ...")
            test_loader = get_data_loader(
                SubDataset(mnist_test, labels_per_task_test[task_i], target_transform=None),
                args.batch_size,
                cuda=True,
                drop_last=True)
            loss_log = {'FID_gINTER': 0.0, 'FID_rINTER': 0.0}
            fid_gINTER = 0
            fid_rINTER = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    x = data[0].cuda()  # too small to use FID with default parameters
                    x_INTER = F.interpolate(x, size=128).repeat(1, 1, 1, 1).cuda()
                    y = data[1].cuda()
                    y_onehot = torch.eye(args.class_dim)[y].cuda()
                    z = torch.Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).cuda()
                    x_g = cvae.decode(torch.cat([z, y_onehot], dim=1))  # .view(x.size(0), 3, 32, 32)
                    x_gINTER = F.interpolate(x_g, size=128).repeat(1, 1, 1, 1)
                    x_r, _, _ = cvae(x, y_onehot)
                    x_rINTER = F.interpolate(x_r, size=128).repeat(1, 1, 1, 1)

                    m_gINTER = FID(device="cuda")
                    m_gINTER.update((x_gINTER, x_INTER))
                    fid_gINTER += m_gINTER.compute()

                    m_rINTER = FID(device="cuda")
                    m_rINTER.update((x_rINTER, x_INTER))
                    fid_rINTER += m_rINTER.compute()

            loss_log["FID_gINTER"] = fid_gINTER / (i + 1)
            loss_log["FID_rINTER"] = fid_rINTER / (i + 1)
            print('FID_gINTER: %.3f \t FID_rINTER: %.3f' % (loss_log["FID_gINTER"], loss_log["FID_rINTER"]))
            append_to_csv([task_i, loss_log["FID_gINTER"], loss_log["FID_rINTER"]], os.path.join(args.modeldir, "fid.csv"))
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
train.add_argument('-method',type=str,default='LGLvKR', choices=['LGLvKR_Fine','LGLvKR_joint','LGLvKR','CVAE_Dy_','CVAE_Dy_Fine','CVAE_Dy_joint'])

common.add_argument('-op',type=str, default='tarin', choices= ['train','eval_'])
common.add_argument('-fid', default=False, action = 'store_true', help='Whether fid needs to be calculated')
common.add_argument('-generate', default=False, action='store_true', help = 'Whether imgs need to be generated')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.method == 'LGLvKR':
    args.modeldir = args.modeldir + '0_LGLvKR/' + '{}epoch'.format(args.epochs)
    main_LGLvKR(args)