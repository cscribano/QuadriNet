# -*- coding: utf-8 -*-
# ---------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# thanks to University of Tokyo Doi Kento
# They refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(logpt)

        w = Variable(self.alpha*target + (1-self.alpha)*(1-target))

        # compute the loss
        loss_tmp = -1 * w * ((1-pt)**self.gamma) * logpt
        
        # averaging (or not) loss
        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss
