# -*- coding: utf-8 -*-
# ---------------------

import random
from typing import *
from path import Path

from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from conf import Conf
if __name__ == '__main__':
    from transforms import Transform
else:
    from dataset.transforms import Transform

class QuadriDataset(Dataset):
    """
    Dataset composed of pairs (x, y) in which:
    * x: RGB image of size 720x405 representing a the scene to segment
    * y: the segmentation mask corresponding to x
    """

    def __init__(self, cnf):
        # type: (Conf) -> None
        """
        :param cnf: configuration object
        """
        self.cnf = cnf
        self.imgs_path = cnf.dataset_path/'images'
        self.masks_path = cnf.dataset_path/'masks'

        #retrieve list of available data items
        self.images_list = sorted(self.imgs_path.files())
        self.masks_list = sorted(self.masks_path.files())

        #sanity checks
        assert(len(self.images_list) == len(self.masks_list))
        self.len = len(self.images_list)

        for image, mask in zip(self.images_list, self.masks_list):
            image_name = image.name.split('.')[0]
            mask_name = mask.name.split('.')[0]
            assert (image_name == mask_name)

    def __len__(self):
        # type: () -> int
        return self.len

    def __getitem__(self, item):
        # type: (int) -> Tuple(Tensor, Tensor)

        image = Image.open(self.images_list[item]).convert('RGB')
        mask = np.load(self.masks_list[item])
        mask = Image.fromarray(mask)

        #Apply data augmentation
        image, mask = Transform(flip_prob=0.5, degrees=10, minscale=0.6)(image, mask)

        return image, mask

def main():
    cnf = Conf(conf_file_path='../conf/default.yaml', exp_name='default')
    d = QuadriDataset(cnf)
    print(len(d))

    im, msk = d[36]
    print(im.max(), im.min())
    print(msk.max(), msk.min())

    im = im.numpy().transpose(1,2,0)
    msk = msk.numpy().transpose(1,2,0).squeeze()

    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.imshow(msk, alpha=0.5)

    plt.show()


if __name__=='__main__':
    main()