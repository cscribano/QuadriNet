# -*- coding: utf-8 -*-
# ---------------------

from math import ceil
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import cv2
import PIL
from PIL.Image import Image

from path import Path
from torchvision.transforms import ToTensor
from typing import *

#RandomHorizontalFlip = RandomHorizontalFlip

def imread(path):
    # type: (Union[Path, str]) -> Image
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def resize_with_pad(image, height=1080, width=1920):
    # type: (np.array, int, int, str) -> np.array
    """
    Resize and image adding a zero padding to keep aspect ratio

    :param image: numpy array (3,H,W) usually obtained by OpenCV
    :param height: desires output height
    :param width: desired output width
    :return: numpy array (3,height, width)
    """

    #target padding
    top, bottom, left, right = (0, 0, 0, 0)

    h, w, _ = image.shape

    dh = abs(h - height)
    dw = abs(w - width)
    closest_edge = min(dh, dw)

    #resise the longest shape to the target size and the shortest accordingly to kep aspect ratio
    # w is the closest shape -> set width
    if dh != closest_edge:
        new_height = ceil(h * width/w)
        image = cv2.resize(image, (width, new_height))

        # pad the height
        delta_h = height - new_height
        top = delta_h // 2
        bottom = delta_h - top

    # h is the closest shape (or image is squared) -> set height
    else:#elif w < longest_edge:
        new_width = ceil(w * height/h)
        image = cv2.resize(image, (new_width, height))

        # pad the height
        delta_w = width - new_width
        left = delta_w // 2
        right = delta_w - left

    BLACK = [0, 0, 0]
    #apply padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return image

def pyplot_to_numpy(figure):
    # type: (figure.Figure) -> np.ndarray
    """
    :param figure: pyplot figure
    :return: numpy array
    """
    figure.canvas.draw()
    x = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(figure):
    # type: (figure.Figure) -> Tensor
    x = pyplot_to_numpy(figure=figure)
    x = ToTensor()(x)
    return x


def apply_colormap_to_tensor(x, cmap='jet', range=(None, None)):
    # type: (Tensor, str, Optional[Tuple[float, float]]) -> Tensor
    """
    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the color map you want to apply
    :param range: tuple of (minimum possible value in x, maximum possible value in x)
    :return: Tensor with shape (1, 3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=range[0], vmax=range[1])
    try:
        x = x.detach().numpy()
    except:
        x = x.cpu().detach().numpy()

    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    x = ToTensor()(x).type('torch.FloatTensor')
    return x

def tensor_to_img_resize(x, shape=(240, 135,)):
    # type: (Tensor, Tuple) -> Tensor
    """
    :param x: input image Tensor (1, 3, H, W)
    :param shape: Desired output shape (H,W)
    :return: Tensor with shape (1, 3, H, W)
    """
    try:
        x = x.detach().numpy()
    except:
        x = x.cpu().detach().numpy()

    x = x.squeeze()
    x = x.transpose(1,2,0)
    x = cv2.resize(x, (240, 135,))
    x = ToTensor()(x)
    return x.unsqueeze(0)


def overlap_image_heatmap(x, heatmap, cmap='jet'):
    # type: (np.array, Tensor) -> Tensor
    """
    :param x: input image Tensor (1, 3, H, W)
    :param heatmap: heatmap tensor(1,H,W)
    :param cmap: color map
    :return: Tensor (1, 3, H, W)
    """
    heatmap = apply_colormap_to_tensor(heatmap).numpy()

    try:
        x = x.detach().numpy()
    except:
        x = x.cpu().detach().numpy()

    heatmap = heatmap.squeeze()
    x = x.squeeze()
    x = x.transpose(1,2,0)
    heatmap = heatmap.transpose(1,2,0)

    x = cv2.addWeighted(heatmap, 0.5, x, 0.5, 0)
    x = ToTensor()(x)

    return x.unsqueeze(0)

def overlap_image_mask(x, prediction):
    # type: (np.array, Tensor) -> Tensor
    """
    :param x: input image Tensor (1, 3, H, W)
    :param heatmap: heatmap tensor(1,H,W)
    :param cmap: color map
    :return: Tensor (1, 3, H, W)
    """

    try:
        prediction = prediction.detach().numpy()
    except:
        prediction = prediction.cpu().detach().numpy()

    x = x.numpy().copy()
    prediction = prediction.copy()

    prediction = prediction.squeeze(0)
    x = x.squeeze()
    x = x.transpose(1,2,0)
    prediction = prediction.transpose(1,2,0)

    x[:,:,1] = cv2.addWeighted(prediction, 0.5, x[:,:,1], 0.5, 0)
    x = ToTensor()(x)

    return x.unsqueeze(0)

def main():
    frame = PIL.Image.open('../jta_dataset/frames/train/seq_0/1.jpg')
    pose = PIL.Image.open('test.jpg')

    #resize Tensor -> Tensor
    frame = ToTensor()(frame) #3,H,W
    print(frame.max())
    frame.unsqueeze_(0) #1,3,W,H

    newframe = tensor_to_img_resize(frame)
    print(newframe.dtype) #([1, 3, 135, 240])

    e = overlap_image_heatmap(newframe, ToTensor()(pose))
    print(e.shape) #([1, 3, 135, 240])

    e = e.numpy().squeeze().transpose(1,2,0)
    plt.imshow(cv2.cvtColor(e, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    main()
