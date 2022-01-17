#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import os
import sys
import numpy as np
import argparse
from copy import deepcopy
from torchvision import transforms
from models.vae import ConditionalVAE, ConditionalVAE_conv
from utils import *
import time

def main_LGLvKR_Fine(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(75 * '=' + '\n' + '|| \t model dir:\t%s\t ||\n' % args.modeldir + 75 * '=')
    permutation = np.array(list(range(10)))
    target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
    train_data = get_dataset(args.dataset, type="train", dir=args.data_dir, target_transform=target_transform,
                              verbose=False)
    test_data = get_dataset(args.dataset, type="test", dir=args.data_dir, target_transform=target_transform,
                             verbose=False)
    classes_per_task = int(np.floor(10 / args.tasks))
    labels_per_task = [list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in
                       range(args.tasks)]
    labels_per_task_test = [list(np.array(range(classes_per_task + classes_per_task * task_id))) for task_id in
                            range(args.tasks)]
    
    sys.stdout = Logger(os.path.join(args.modeldir, '{}_log_{}.txt'.format(args.name,args.op)))
    # training
    time_cost = 0
    for task_i in range(args.tasks):
        print(40 * '=' + ' Task %1d ' % (task_i+1) + 40 * '=')
        if not os.path.exists(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i)):
            train_loader = get_data_loader(
                SubDataset(train_data, labels_per_task[task_i], target_transform=None),
                args.batch_size,
                cuda=True,
                drop_last=True)
            print("training(n=%5d)..." % len(train_loader.dataset))

            if task_i == 0:
                if args.dataset == 'mnist' or args.dataset == 'fashion':
                    cvae = ConditionalVAE(args)
                elif args.dataset == 'svhn' or args.dataset == 'cifar10':
                    cvae = ConditionalVAE_conv(args)
                print("there are {} params with {} elems in the cvae".format(
                    len(list(cvae.parameters())), sum(p.numel() for p in cvae.parameters() if p.requires_grad))
                )
            else:
                cvae = torch.load(os.path.join(args.modeldir, args.name + '_%1d.pth' % (task_i - 1)))
                print("there are {} params with {} elems in the cvae".format(
                    len(list(cvae.parameters())), sum(p.numel() for p in cvae.parameters() if p.requires_grad))
                )
            cvae = cvae.cuda()
            optimizer_CVAE = torch.optim.Adam(cvae.parameters(), lr=args.lr, betas=(0.9, 0.999))

            start = time.time()
            for epoch in range(args.epochs):
                loss_log = {'V/loss': 0.0, 'V/loss_rec': 0.0, 'V/loss_var': 0.0}
                for x, y in train_loader:
                    if args.dataset == 'mnist' or args.dataset == 'fashion':
                        x = x.cuda().view(x.size(0), -1)    
                    elif args.dataset == 'svhn' or args.dataset == 'cifar10':
                        x = x.cuda()
                    y = torch.eye(args.class_dim)[y].cuda()
                    x_rec, mu, logvar = cvae(x, y)
                    loss_rec = torch.nn.MSELoss(reduction='sum')(x, x_rec) / x.size(0)
                    loss_var = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
                    loss_cvae = loss_rec + args.alpha_var * loss_var
                    optimizer_CVAE.zero_grad()
                    loss_cvae.backward()
                    optimizer_CVAE.step()

                    loss_log['V/loss'] += loss_cvae.item()
                    loss_log['V/loss_rec'] += loss_rec.item()
                    loss_log['V/loss_var'] += loss_var.item() * args.alpha_var

                print('[VAE Epoch%2d]\t V/loss: %.3f\t V/loss_rec: %.3f\t V/loss_var: %.3f'
                      % (epoch + 1,
                         loss_log['V/loss'],
                         loss_log['V/loss_rec'],
                         loss_log['V/loss_var']))
            time_cost += time.time() - start
            torch.save(cvae, os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i))
            
        else:
            print(40 * '=' + ' Task %1d ' % (task_i+1) + 40 * '=')
            cvae = torch.load(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i))

        cvae.eval()
        ##################### Test with LPIPS #########################################################################
        if args.LPIPS:
            caculate_LPIPS(args,task_i,test_data,labels_per_task_test[task_i],cvae.decode)

        ##################### Test with FID #########################################################################
        if args.fid:
            caculate_fid(args,task_i,test_data,labels_per_task_test[task_i],cvae.decode)

        ####################### Test with Acc and rAcc ####################################
        if args.ACC:
            caculate_ACC(args,task_i,train_data,test_data,labels_per_task_test[task_i],cvae.decode)
        # ####################### Test as generated pictures ####################################
        if args.generate:
            generat_img(args,task_i,labels_per_task[task_i],cvae.decode)
        print(100 * '-')
    print('Total training time is: %.3f seconds'%time_cost)

def main_LGLvKR_joint(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(75 * '=' + '\n' + '|| \t model dir:\t%s\t ||\n' % args.modeldir + 75 * '=')
    permutation = np.array(list(range(10)))
    target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
    train_data = get_dataset(args.dataset, type="train", dir=args.data_dir, target_transform=target_transform,
                              verbose=False)
    test_data = get_dataset(args.dataset, type="test", dir=args.data_dir, target_transform=target_transform,
                             verbose=False)
    classes_per_task = int(np.floor(10 / args.tasks))
    labels_per_task = [list(np.array(range(classes_per_task+ classes_per_task * task_id))) for task_id in range(args.tasks)]
    labels_per_task_test = [list(np.array(range(classes_per_task + classes_per_task * task_id))) for task_id in range(args.tasks)]

    sys.stdout = Logger(os.path.join(args.modeldir, '{}_log_{}.txt'.format(args.name,args.op)))
    # training
    time_cost = 0
    for task_i in range(args.tasks):
        print(40 * '=' + ' Task %1d ' % (task_i+1) + 40 * '=')
        if not os.path.exists(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i)):
            train_loader = get_data_loader(
                SubDataset(train_data, labels_per_task[task_i], target_transform=None),
                args.batch_size,
                cuda=True,
                drop_last=True)
            print("training(n=%5d)..." % len(train_loader.dataset))
            if args.dataset == 'mnist' or args.dataset == 'fashion':
                    cvae = ConditionalVAE(args)
            elif args.dataset == 'svhn' or args.dataset == 'cifar10':
                cvae = ConditionalVAE_conv(args)
            print("there are {} params with {} elems in the cvae".format(
                len(list(cvae.parameters())), sum(p.numel() for p in cvae.parameters() if p.requires_grad))
            )
            cvae = cvae.cuda()
            optimizer_CVAE = torch.optim.Adam(cvae.parameters(), lr=args.lr, betas=(0.9, 0.999))
            start = time.time()
            for epoch in range(args.epochs):
                loss_log = {'V/loss': 0.0, 'V/loss_rec': 0.0, 'V/loss_var': 0.0}
                for x, y in train_loader:
                    if args.dataset == 'mnist' or args.dataset == 'fashion':
                        x = x.cuda().view(x.size(0), -1)    
                    elif args.dataset == 'svhn' or args.dataset == 'cifar10':
                        x = x.cuda()
                    y = torch.eye(args.class_dim)[y].cuda()
                    x_rec, mu, logvar = cvae(x, y)
                    loss_rec = torch.nn.MSELoss(reduction='sum')(x, x_rec) / x.size(0)
                    loss_var = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
                    loss_cvae = loss_rec + args.alpha_var * loss_var
                    optimizer_CVAE.zero_grad()
                    loss_cvae.backward()
                    optimizer_CVAE.step()

                    loss_log['V/loss'] += loss_cvae.item()
                    loss_log['V/loss_rec'] += loss_rec.item()
                    loss_log['V/loss_var'] += loss_var.item() * args.alpha_var

                print('[VAE Epoch%2d]\t V/loss: %.3f\t V/loss_rec: %.3f\t V/loss_var: %.3f'
                      % (epoch + 1,
                         loss_log['V/loss'],
                         loss_log['V/loss_rec'],
                         loss_log['V/loss_var']))
            time_cost += time.time() - start
            torch.save(cvae, os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i))
        else:
            cvae = torch.load(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i))

        cvae.eval()
        ##################### Test with LPIPS #########################################################################
        if args.LPIPS:
            caculate_LPIPS(args,task_i,test_data,labels_per_task_test[task_i],cvae.decode)

        ##################### Test with FID #########################################################################
        if args.fid:
            caculate_fid(args,task_i,test_data,labels_per_task_test[task_i],cvae.decode)

        ####################### Test with Acc and rAcc ####################################
        if args.ACC:
            caculate_ACC(args,task_i,train_data,test_data,labels_per_task_test[task_i],cvae.decode)
        # ####################### Test as generated pictures ####################################
        if args.generate:
            generat_img(args,task_i,labels_per_task[task_i],cvae.decode)

        print(100 * '-')
    print('Total training time is: %.3f seconds'%time_cost)

def main_LGLvKR(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(75 * '=' + '\n' + '|| \t model dir:\t%s\t ||\n' % args.modeldir + 75 * '=')
    permutation = np.array(list(range(10)))
    target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
    train_data = get_dataset(args.dataset, type="train", dir=args.data_dir, target_transform=target_transform, verbose=False)
    test_data = get_dataset(args.dataset, type="test", dir=args.data_dir, target_transform=target_transform, verbose=False)
    classes_per_task = int(np.floor(10 / args.tasks))
    labels_per_task = [list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(args.tasks)]
    labels_per_task_test = [list(np.array(range(classes_per_task + classes_per_task * task_id))) for task_id in range(args.tasks)]

    sys.stdout = Logger(os.path.join(args.modeldir, 'log_{}_{}.txt'.format(args.name,args.op)))
    # training
    time_cost = 0
    for task_i in range(args.tasks):
        print(40 * '=' + ' Task %1d ' % (task_i + 1) + 40 * '=')
        if not os.path.exists(os.path.join(args.modeldir, args.name + '_%1d.pth' % task_i)):
            train_loader = get_data_loader(
                SubDataset(train_data, labels_per_task[task_i], target_transform=None),
                args.batch_size,
                cuda=True,
                drop_last=True)
            print("training(n=%5d)..." % len(train_loader.dataset))
            if task_i == 0:
                if args.dataset == 'mnist' or args.dataset == 'fashion':
                    cvae = ConditionalVAE(args)
                elif args.dataset == 'svhn' or args.dataset == 'cifar10':
                    cvae = ConditionalVAE_conv(args)
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
            optimizer_CVAE = torch.optim.Adam(cvae.parameters(), lr=args.lr, betas=(0.9, 0.999))
            start = time.time()
            for epoch in range(args.epochs):
                loss_log = {'V/loss': 0.0, 'V/loss_rec': 0.0, 'V/loss_var': 0.0, 'V/loss_var_hat': 0.0, 'V/loss_aug': 0.0}
                for x, y in train_loader:
                    if args.dataset == 'mnist' or args.dataset == 'fashion':
                        x = x.cuda().view(x.size(0), -1)    
                    elif args.dataset == 'svhn' or args.dataset == 'cifar10':
                        x = x.cuda()
                    y = torch.eye(args.class_dim)[y].cuda()
                    if task_i == 0:
                        x_rec, mu, logvar = cvae(x, y)
                        mu_hat, logvar_hat = cvae.encode(x_rec, y)

                        loss_rec = torch.nn.MSELoss(reduction='sum')(x, x_rec) / x.size(0)
                        loss_var = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
                        loss_var_hat = (-0.5 * torch.sum(1 + logvar_hat - mu_hat ** 2 - logvar_hat.exp())) / x_rec.size(0)
                        loss_cvae = loss_rec + args.alpha_var * loss_var + args.alpha_var_hat * loss_var_hat

                    else:
                        mu, logvar = cvae.encode(x, y)
                        z = cvae.reparameterize(mu, logvar)

                        y_pre = torch.randint(0, labels_per_task[task_i][0], (args.batch_size,))
                        y_pre = torch.eye(args.class_dim)[y_pre].cuda()
                        z_pre = torch.empty((args.batch_size, args.latent_dim)).normal_(mean=0, std=1).cuda()
                        xPre_old = cvae_old.decode(z_pre, y_pre)
                        z_merged = torch.cat((z_pre, z))
                        y_merged = torch.cat((y_pre, y))
                        xRec_merged = cvae.decode(z_merged, y_merged)
                        mu_hat, logvar_hat = cvae.encode(xRec_merged[:args.batch_size], y)

                        loss_rec = torch.nn.MSELoss(reduction='sum')(x, xRec_merged[args.batch_size:]) / x.size(0)
                        loss_var = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())) / x.size(0)
                        loss_var_hat = (-0.5 * torch.sum(1 + logvar_hat - mu_hat ** 2 - logvar_hat.exp())) / x.size(0)
                        loss_aug = torch.dist(xRec_merged[:args.batch_size], xPre_old, 2) / xPre_old.size(0)
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
        ##################### Test with LPIPS #########################################################################
        if args.LPIPS:
            caculate_LPIPS(args,task_i,test_data,labels_per_task_test[task_i],cvae.decode)

        ##################### Test with FID #########################################################################
        if args.fid:
            caculate_fid(args,task_i,test_data,labels_per_task_test[task_i],cvae.decode)

        ####################### Test with Acc and rAcc ####################################
        if args.ACC:
            caculate_ACC(args,task_i,train_data,test_data,labels_per_task_test[task_i],cvae.decode)
        # ####################### Test as generated pictures ####################################
        if args.generate:
            generat_img(args,task_i,labels_per_task[task_i],cvae.decode)
        print(100 * '-')
    print('Total training time is: %.3f seconds'%time_cost)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    common = parser.add_argument_group("common parameters group")
    network = parser.add_argument_group("network parameters group")
    train = parser.add_argument_group("training parameters group")
    common.add_argument('-modeldir', default='./checkpoints/SplitMNIST/', help='') 
    common.add_argument('-dataset',type=str, default='mnist', help='dataset name & uses dataset mnist/svhn/fasion')
    common.add_argument('-data_dir', type=str, default='./dataset/mnist', help='data directory')
    common.add_argument('-tasks', type=int, default=10, help='number of tasks')
    common.add_argument('-batch_size', type=int, default=128, help='batch size')
    common.add_argument('-op',type = str, default = 'tarin', choices= ['train','eval_'])
    common.add_argument('-LPIPS', default = False, action = 'store_true', help = 'Whether LPIPS needs to be calculated')
    common.add_argument('-fid', default = False, action = 'store_true', help = 'Whether fid needs to be calculated')
    common.add_argument('-ACC', default = False, action = 'store_true', help = 'Whether ACC needs to be calculated')
    common.add_argument('-generate', default = False, action = 'store_true', help = 'Whether imgs need to be generated')
    network.add_argument('-feat_dim', type=int, default=32 * 32, help='input features dimension')
    network.add_argument('-latent_dim', type=int, default=2, help='latent variable dimension')
    network.add_argument('-class_dim', type=int, default=10, help='class or one-hot label dimension')
    network.add_argument('-hidden_dim', type=int, default=256, help='hidden dimension')
    train.add_argument('-lr', type=float, default=0.0002, help='learning rate')
    train.add_argument('-alpha_var', type=float, default=1., help='alpha parameter for variational loss')
    train.add_argument('-alpha_var_hat', type=float, default=1., help="alpha parameter for variational loss of reconstructed data")
    train.add_argument('-alpha_aug', type=float, default=1., help="alpha parameter for the augmented loss")
    train.add_argument('-epochs', default=10, type=int, metavar='N', help='number of epochs')  
    train.add_argument("-gpu", type=str, default='0', help='which gpu to use')
    train.add_argument("-name", type=str, default='10tasks', help='the name of the temporary saved model')
    train.add_argument('-method',type = str,default = 'LGLvKR',choices=['LGLvKR_Fine','LGLvKR_joint','LGLvKR','LGLvKR_noFC','LGLvKR_noKR'])
    
    args = parser.parse_args()

    if args.dataset == 'mnist' or args.dataset =='mnist28':
        args.feat_dim = 32*32*1
        model_dir = './checkpoints/SplitMNIST/'
    elif args.dataset == 'fashion':
        args.feat_dim = 32*32*1
        model_dir = './checkpoints/fashion/'
    elif args.dataset == 'svhn':
        args.feat_dim = 32*32*3
        args.data_dir = './dataset/svhn'
        model_dir = './checkpoints/svhn/'

    if args.method == 'LGLvKR_fine' :
        args.modeldir = model_dir + 'LGLvKR_finetuning/' + '{}epoch'.format(args.epochs)
        main_LGLvKR_Fine(args)
    elif args.method == 'LGLvKR_joint':
        args.modeldir = model_dir + 'LGLvKR_jointTraining/' + '{}epoch'.format(args.epochs)
        main_LGLvKR_joint(args)

    elif args.method == 'LGLvKR':
        args.modeldir = model_dir + 'LGLvKR/' + '{}epoch'.format(args.epochs)
        main_LGLvKR(args)
    elif args.method == 'LGLvKR_noFC':
        args.modeldir = model_dir + 'LGLvKR_noFC/' + '{}epoch'.format(args.epochs)
        args.alpha_var_hat = 0.
        main_LGLvKR(args)
    elif args.method == 'LGLvKR_noKR':
        args.modeldir = model_dir + 'LGLvKR_noKR/' + '{}epoch'.format(args.epochs)
        args.alpha_aug = 0.
        main_LGLvKR(args)