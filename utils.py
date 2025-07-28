import sys
import time
import timm

import torch
import torch.nn as nn
from typing import Optional, Sequence
import torch.nn.functional as F
from common.modules.classifier import Classifier as ClassifierBase

import common.vision.models as models
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from data_loader import get_rd_dataset, get_rd_class_name

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()

def get_model(model_name):
    if model_name in models.__dict__:
        # load models from common_mic.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        return Exception('error in build model process')

    return backbone

def get_dataset(root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform

    train_source_dataset = get_rd_dataset(root, source=True, dataset_name=source, transform=train_source_transform, appli='train')
    train_target_dataset = get_rd_dataset(root, source=False, dataset_name=target, transform=train_target_transform, appli='train')
    val_dataset = get_rd_dataset(root, source=False, dataset_name=target, transform=val_transform, appli='val')
    # test_dataset = val_dataset
    test_source_dataset = get_rd_dataset(root, source=True, dataset_name=source, transform=val_transform, appli='test')
    test_dataset = get_rd_dataset(root, source=False, dataset_name=target, transform=val_transform, appli='test')
    class_names = get_rd_class_name(target)
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, test_source_dataset, num_classes, class_names

def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    confmat = ConfusionMatrix(len(args.class_names))
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            confmat.update(target, output.argmax(1))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        h,acc_global, accs = confmat.compute()
        all_acc = torch.mean(accs).item() * 100

        print(h)
        print(confmat.format(args.class_names))
        print(' * All test acc {all:.3f}'.format(all=all_acc))

    return all_acc

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class GradientReverseLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, x):
        return GradientReverse.apply(x)


class Classifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout()
        )
        super(Classifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.grl = GradientReverseLayer()

    def forward(self, x):

        features1 = self.backbone(x)
        features = self.bottleneck(features1)
        outputs = self.head1(features)
        if self.training:
            return outputs, features
        else:
            return outputs

class PlcLoss(nn.Module):
    def __init__(self, threshold=0.96):
        super(PlcLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y_str, y_wea):
        confidence, pseudo_labels = F.softmax(y_wea.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (F.cross_entropy(y_str, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] // 2

        features = F.normalize(features, dim=1)
        f_t1, f_t2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f_t1.unsqueeze(1), f_t2.unsqueeze(1)], dim=1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (36,256)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # prevent computing log(0), which will produce Nan in the loss
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

