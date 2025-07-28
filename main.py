import random
import time
import warnings
import sys
import argparse
import shutil
import os
import numpy as np
import os.path as osp
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import utils

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from utils import Classifier, SupConLoss, PlcLoss
from transform import get_train_transform, get_val_transform
from common.grl.grl import DomainDiscriminator, DomainAdversarialLoss

device = torch.device("cuda")

is_domain = True

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    cudnn.benchmark = True

    train_transform = {'IQ': get_train_transform}
    val_transform = get_val_transform
    print("train transform: ", train_transform)
    print("val_transform: ", val_transform)
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, test_source_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=args.workers)
    test_source_loader = DataLoader(test_source_dataset, batch_size=2, shuffle=False, num_workers=args.workers)

    len_source_loader = len(train_source_loader)
    len_target_loader = len(train_target_loader)
    min_n_batch = min(len_source_loader, len_target_loader)
    max_n_batch = max(len_source_loader, len_target_loader)
    print('min_n_batch: ', min_n_batch, ' max_n_batch：', max_n_batch)
    if min_n_batch != 0:
        args.iters_per_epoch = max_n_batch

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)

    classifier = Classifier(backbone, num_classes, args.bottleneck_dim).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define loss function
    domain_loss = DomainAdversarialLoss(domain_discri).to(device)
    optimizer = SGD(classifier.get_parameters()+ domain_discri.get_parameters(), args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    # define optimizer and lr scheduler
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    psc_fn = SupConLoss(temperature=0.1).cuda()
    plc_fn = PlcLoss(threshold=args.T).cuda()

    # load checkpoint
    save_pth_latest = logger.get_checkpoint_path(
        f'latest-model_{args.model_suffix}-batch_{args.batch_size}-lr_{args.lr}'
        f'-alpha_{args.alpha}-pre_epoch_{args.pretrain_epoch}-ST_{args.source}_{args.target}-matrix_aug0-{args.pth_suffix}')
    save_pth_best = logger.get_checkpoint_path(f'best-model_{args.model_suffix}-batch_{args.batch_size}-lr_{args.lr}'
        f'-alpha_{args.alpha}-pre_epoch_{args.pretrain_epoch}-ST_{args.source}_{args.target}-matrix_aug0-{args.pth_suffix}')

    # resume from the best checkpoint
    if args.phase == 'analysis':
        print('load checkpoint: best model')
        checkpoint = torch.load(save_pth_best, map_location='cpu')
        classifier.load_state_dict(checkpoint['net'].state_dict())
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)  # 定义特征提取器
        source_feature, labels_s = collect_feature(test_source_loader, feature_extractor, device)
        target_feature, labels_t = collect_feature(test_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, f'ours_tnse_{args.target}.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        # tsne.visualize1(source_feature, target_feature, tSNE_filename)
        # tsne.visualize2(source_feature, target_feature, labels_s, labels_t, tSNE_filename)
        # tsne.visualize_target_only(target_feature, labels_t, tSNE_filename, tSNE_filename1)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print(f"Mean A-distance ={A_distance}")
        return

    # image classification test
    if args.phase == 'test':
        print('load checkpoint: best model')
        checkpoint = torch.load(save_pth_best, map_location='cpu')
        classifier.load_state_dict(checkpoint)
        acc = utils.validate(test_loader, classifier, args, device)
        # print("Classification Accuracy = {:0.4f}".format(acc))
        return

    checkpoint_epoch = 0
    best_acc = 0.

    is_checkpoint = True
    if is_checkpoint:
        checkpoint_path = save_pth_best
        if os.path.exists(checkpoint_path):
            if input('Continue from the checkpoint?') in ['y', 'Y', 'yes', 'YES']:
                checkpoint = torch.load(checkpoint_path)
                classifier.load_state_dict(checkpoint['net'].state_dict())
                # 替换最后一层以适应新的 num_class
                # data_args.model.fc = nn.Linear(512*data_args.model.block.expansion, 128)
                classifier.to(device)  # 创建了新的FC需要重新将模型放到CUDA上
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                # 添加 in 8.22
                checkpoint_epoch = checkpoint['epoch']
                # load best_acc
                best_acc = checkpoint['best']
                print(f"lr:{lr_scheduler.get_last_lr()[0]}, epoch:{checkpoint_epoch}, best:{best_acc:3%}")
            else:
                if input('Remove all checkpoint?') in ['y', 'Y', 'yes', 'YES']:
                    os.remove(checkpoint_path)
                    print('Remove checkponts!')

    # start training
    best_epoch = checkpoint_epoch
    for epoch in range(checkpoint_epoch, args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, psc_fn, plc_fn,
              optimizer, lr_scheduler, epoch, args, domain_loss)

        # evaluate on validation set
        acc = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        save_pth(save_path=save_pth_latest, model=classifier, optimizer=optimizer, scheduler=lr_scheduler, epoch=epoch,
                 best_acc=best_acc)
        if acc > best_acc:
            shutil.copy(save_pth_latest, save_pth_best)
            best_acc = acc
            best_epoch = epoch
            print(f"Best model save success, best acc is {best_acc}, current best epoch is {epoch}!")
        print(f"Current acc is {acc}, best acc is {best_acc}, current epoch is {epoch}, best epoch is {best_epoch}!")
    print("best_accu = {:.3f}!".format(best_acc))

    print("*****************************TEST BEST MODEL*******************************")
    classifier.load_state_dict(torch.load(save_pth_best)['net'].state_dict())
    acc = utils.validate(test_loader, classifier, args, device)
    print("best model test_accu = {:.3f}".format(acc))
    print("*****************************TEST LATEST MODEL*****************************")
    classifier.load_state_dict(torch.load(save_pth_latest)['net'].state_dict())
    acc = utils.validate(test_loader, classifier, args, device)
    print("latest model test_accu = {:.3f}".format(acc))

    logger.close()

def save_pth(save_path, model, optimizer, scheduler, epoch, best_acc=None):
    torch.save({
        'epoch': epoch,
        'net': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best': best_acc
    }, save_path)
    print(f"{save_path.split('/')[-1]} save success, current epoch is {epoch}!")

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: Classifier,
           psc_fn: nn.Module, plc_fn: nn.Module, optimizer: SGD,
           lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, domain_fn: nn.Module):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Total Loss', ':3.2f')
    plc_losses = AverageMeter('Plc Loss', ':5.4f')
    domain_losses = AverageMeter('Domain Loss', ':5.4f')
    psc_losses = AverageMeter('Psc Loss', ':5.4f')
    sce_losses = AverageMeter('Sce Loss', ':5.4f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, sce_losses, domain_losses, psc_losses, plc_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):

        x_s_ori, x_s, _, _, labels_s = next(train_source_iter)
        x_t_ori, x_t_w, x_t_s1, _, labels_t = next(train_target_iter)
        bsz = labels_s.shape[0]

        x_t = torch.cat([x_t_w, x_t_s1], dim=0)
        x_s = x_s.to(device)  # 只用到原域弱增强数据
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)
        y_t_w, y_t_s1 = torch.split(y_t, [bsz, bsz], dim=0)
        f_t_w, f_t_s1 = torch.split(f_t, [bsz, bsz], dim=0)

        if epoch > args.pretrain_epoch:
            plc_loss, mask, pseudo_labels= plc_fn(y_str=y_t_s1, y_wea=y_t_w)
        else:
            plc_loss = torch.tensor([0]).cuda()
            mask = torch.zeros(bsz, device=device)
            pseudo_labels = None

        # source domain discriminative loss
        sce_loss = F.cross_entropy(y_s, labels_s)
        domain_loss = domain_fn(f_s, f_t_w)

        selected_indices = (mask == 1).nonzero(as_tuple=True)[0]
        if selected_indices.numel() > 0:
            f_t_w_selected = f_t_w[selected_indices]
            f_t_s1_selected = f_t_s1[selected_indices]
            selected_features = torch.cat([f_t_w_selected, f_t_s1_selected], dim=0)
            selected_labels = pseudo_labels[selected_indices]
            psc_loss = psc_fn(features=selected_features, labels=selected_labels)
        else:
            psc_loss = torch.tensor(0., device=device)

        loss = sce_loss + domain_loss + args.alpha * (psc_loss + plc_loss)
        losses.update(loss.item(), labels_s.size(0))
        plc_losses.update(plc_loss.item(), labels_s.size(0))
        domain_losses.update(domain_loss.item(), labels_s.size(0))
        psc_losses.update(psc_loss.item(), labels_s.size(0))
        sce_losses.update(sce_loss.item(), labels_s.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

if __name__ == '__main__':
    model_name = 'resnet50'
    parser = argparse.ArgumentParser(description='CACL for cross domain user identificaiton')
    # dataset parameters
    parser.add_argument('-pth_suffix', default='', help='Model weight suffix')
    parser.add_argument('-T', default=0.9, help='threshold of pseudo label')
    parser.add_argument('-root', metavar='DIR', default='.', help='root path of dataset')

    parser.add_argument('-s', '--source', default='15-0', help='source domain')
    parser.add_argument('-t', '--target', default='15+30', help='target domain')

    parser.add_argument('--alpha', type=float, default=0.9, help='loss balance alpha')

    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--backbone_feature_len', type=int, default='512')

    parser.add_argument('--model_suffix', type=str, default=f'{model_name}')

    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default=f'{model_name}',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             f' (default: {model_name})')  # backbone
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('-b', '--batch-size', default=18, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--pretrain-epoch', default=5, type=int,
                        help='pretrain epoch for discriminative feature learning')  # 预训练次数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',  # epochs
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=2, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', default=True, action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)


