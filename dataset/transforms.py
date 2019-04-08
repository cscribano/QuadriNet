# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

import numpy as np
import random
from math import ceil

def resize_with_pad_pil(image, height=1080, width=1920):
    # type: (Image, int, int) -> np.array
    """
    Resize and image adding a zero padding to keep aspect ratio

    :param image: PIL Image
    :param height: desires output height
    :param width: desired output width
    :return: PIL Image
    """

    #target padding
    top, bottom, left, right = (0, 0, 0, 0)

    w, h = image.size

    dh = abs(h - height)
    dw = abs(w - width)
    closest_edge = min(dh, dw)

    #resise the longest shape to the target size and the shortest accordingly to kep aspect ratio
    # w is the closest shape -> set width
    if dh != closest_edge:
        new_height = ceil(h * width/w)

        image = image.resize((width, new_height))

        # pad the height
        delta_h = height - new_height
        top = delta_h // 2
        bottom = delta_h - top

    # h is the closest shape (or image is squared) -> set height
    else:#elif w < longest_edge:
        new_width = ceil(w * height/h)
        image = image.resize((new_width, height))

        # pad the height
        delta_w = width - new_width
        left = delta_w // 2
        right = delta_w - left

    BLACK = [0, 0, 0]
    #apply padding
    image = F.pad(image, padding=(left, top, right, bottom))

    return image


def pad_if_smaller(image, height=1080, width=1920):
    # type: (Image, int, int) -> np.array
    """
    Resize and image adding a zero padding to keep aspect ratio

    :param image: PIL Image
    :param height: desires output height
    :param width: desired output width
    :return: PIL Image
    """

    #target padding
    top, bottom, left, right = (0, 0, 0, 0)
    w, h = image.size

    # pad the height
    delta_h = height - h
    top = delta_h // 2
    bottom = delta_h - top

    # pad the height
    delta_w = width - w
    left = delta_w // 2
    right = delta_w - left

    BLACK = [0, 0, 0]
    #apply padding
    image = F.pad(image, padding=(left, top, right, bottom))

    return image

class Transform(object):
    def __init__(self, flip_prob=0, degrees=10, minscale=0.6):
        # type: (float, int, float) -> None
        """
        :param flip_prob: Probability of H-Flip
        :param degrees: max degrees of rotation, min is assumed -degrees
        :param minscale: Minimum crop percentage of original dimensions
        """
        self.flip_prob = flip_prob
        self.degrees = degrees
        self.minscale = minscale
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __call__(self, image, target):
        # type: (Image, Image) -> (Tensor, Tensor)

        assert image.size == target.size, "Image and target must have the same dimension"
        width, heigh = image.size

        #shit happens
        if width != 720 or heigh != 405:
            image = resize_with_pad_pil(image, 405, 720)
            target = resize_with_pad_pil(target, 405, 720)

            width, heigh = image.size

        #random HFlip
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)

        #Random Resized Crop
        w = int(image.size[0] * random.uniform(self.minscale, 1))
        h = int(image.size[1] * random.uniform(self.minscale, 1))

        if w != width and h != heigh:
            i = random.randint(0,  image.size[1] - h)#top margin
            j = random.randint(0, image.size[0] - w)#left margin

            # margins: (left, upper, right, lower)
            image = image.crop((j, i, j + w, i + h))
            target = target.crop((j, i, j + w, i + h))

            image = pad_if_smaller(image, width=width, height=heigh)
            target = pad_if_smaller(target, width=width, height=heigh)

        #random Rotate
        angle = random.uniform(-1*self.degrees, self.degrees)
        #print("Angle{}".format(angle))
        if angle != 0:
            image = F.rotate(image, angle, Image.NEAREST)
            target = F.rotate(target, angle, Image.NEAREST)

        #Normalize and convert to tensor
        image = self.to_tensor(image)
        target = self.to_tensor(target).type(torch.FloatTensor)
        image = self.normalize(image)

        return image, target
