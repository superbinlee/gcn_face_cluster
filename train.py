###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import model
from feeder.feeder import Feeder
from utils import to_numpy
from utils.logging import Logger
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint

from sklearn.metrics import precision_score, recall_score


def main(args):
    """
    在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    如果设置初始化，则每次初始化都是固定的。
    if args.seed is not None:    　　
    random.seed(args.seed) #    　　
    torch.manual_seed(args.seed)  #为CPU设置种子用于生成随机数，以使得结果是确定的   　　
    torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子；  　　
    cudnn.deterministic = True#如果使用多个GPU，应该使用
    torch.cuda.manual_seed_all()为所有的GPU设置种子。
    #如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    """
    cuDNN 是英伟达专门为深度神经网络所开发出来的 GPU 加速库，
    针对卷积、池化等等常见操作做了非常多的底层优化，比一般的 GPU 程序要快很多。
    大多数主流深度学习框架都支持 cuDNN，PyTorch 自然也不例外。
    在使用 GPU 的时候，PyTorch 会默认使用 cuDNN 加速。
    但是，在使用 cuDNN 的时候，torch.backends.cudnn.benchmark 模式是为 False。
    所以就意味着，我们的程序可能还可以继续提速！
    """
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    trainset = Feeder(args.feat_path,
                      args.knn_graph_path,
                      args.label_path,
                      args.seed,
                      args.k_at_hop,
                      args.active_connection)
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size,
        num_workers=args.workers, shuffle=True, pin_memory=True)

    net = model.gcn().cuda()
    opt = torch.optim.SGD(net.parameters(), args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()

    save_checkpoint({'state_dict': net.state_dict(), 'epoch': 0, }, False, fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(0)))
    for epoch in range(args.epochs):
        adjust_lr(opt, epoch)
        train(trainloader, net, criterion, opt, epoch)
        save_checkpoint({'state_dict': net.state_dict(), 'epoch': epoch + 1, }, False, fpath=osp.join(args.logs_dir, 'epoch_{}.ckpt'.format(epoch + 1)))


def train(loader, net, crit, opt, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    net.train()
    end = time.time()
    for i, ((feat, adj, cid, h1id), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(), (feat, adj, cid, h1id, gtmat))
        pred = net(feat, adj, h1id)
        labels = make_labels(gtmat).long()
        loss = crit(pred, labels)
        p, r, acc = accuracy(pred, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(epoch, i, len(loader),
                                                                        batch_time=batch_time,
                                                                        data_time=data_time,
                                                                        losses=losses,
                                                                        accs=accs,
                                                                        precisions=precisions,
                                                                        recalls=recalls))


def make_labels(gtmat):
    # https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view
    # 展开成1维
    return gtmat.view(-1)


def adjust_lr(opt, epoch):
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in [1, 2, 3, 4]:
        args.lr *= 0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(pred, label):
    # 返回输入张量中所有元素的最大值的索引。dim=1表示返回每行的最大元素的索引。
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    # 精度是比率tp / (tp + fp)，其中tp是真阳性数，fp是假阳性数。精度直观地是分类器不将负样本标记为正样本的能力。
    p = precision_score(label, pred)
    # 召回率是 tp / (tp + fn) 的比率，其中 tp 是真阳性数，fn是假阴性数。召回率直观地是分类器找到所有正样本的能力。
    r = recall_score(label, pred)
    return p, r, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--print_freq', default=200, type=int)
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=4)
    # Training args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--feat_path', type=str, metavar='PATH', default=osp.join(working_dir, '../facedata/CASIA.feas.npy'))
    parser.add_argument('--knn_graph_path', type=str, metavar='PATH', default=osp.join(working_dir, '../facedata/knn.graph.CASIA.kdtree.npy'))
    parser.add_argument('--label_path', type=str, metavar='PATH', default=osp.join(working_dir, '../facedata/CASIA.labels.npy'))
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[200, 10])
    parser.add_argument('--active_connection', type=int, default=10)
    # 解析
    args = parser.parse_args()
    main(args)
