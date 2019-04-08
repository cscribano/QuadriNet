# -*- coding: utf-8 -*-
# ---------------------

from typing import *
from path import Path

import torch
from torch import tensor
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

np.warnings.filterwarnings('ignore')

Seq = Sequence[Union[int, float]]

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    # type: (np.array, np.array, int) -> Tuple(float, float)

    imPred = imPred.copy()
    imLab = imLab.copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def compute_metrics(y_true, y_pred):
    #type: (tensor, tensor) -> Dict[str, Union[int, float]]
    """
    :param y_true: Ground-thru segmentation mask (1,1,W,H)
    :param y_pred: Predicted segmentation mask (1,1,W,H)
    :return: a dictionary of metrics, 'met', related to segmentation
             the the available metrics are:
             (1) met['pp'] = pixel precision
             (2) met['iou'] = intersection over union
    """

    imPred = y_pred.cpu().numpy().squeeze()
    imTrue = y_true.cpu().numpy().squeeze()

    #normalize prediction
    imPred = sigmoid(imPred)
    imPred[imPred > 0.5] = 1
    imPred[imPred < 0.5] = 0

    #compute IoU
    intersection, union = intersectionAndUnion(imPred, imTrue, 2)

    #compute IoU only for class '1' since '0' is background
    if union[1] != 0:
        iou = intersection[1]/union[1]
    else:
        iou = 0

    #compute pixel accuracy
    ac = accuracy(imPred, imTrue)[0]

    # build the metrics dictionary
    metrics = {
        'iou': iou, 'ac':ac
    }

    return metrics

def main():
    a = np.load('26.npy')
    #b = np.zeros_like(a)
    b = a.copy()

    c = compute_metrics(a,b)
    print(c)

if __name__ == '__main__':
    main()
