# -*- coding: utf-8 -*-
# ---------------------

import torch
from torchvision import transforms

import time
import cv2
import os
import numpy as np
from path import Path
from matplotlib import pyplot as plt

from metrics import sigmoid
from conf import Conf
from utils import resize_with_pad
from models import BaseModel, QuadriFcn

class LiveTester:

    def __init__(self, cnf, model, transform=None):
        # type: (Conf, BaseModel, transforms) -> LiveTester

        self.cnf = cnf
        self.log_path = cnf.exp_log_path
        self.transform = transform

        # init model
        self.model = model
        self.model = self.model.to(self.cnf.device)

        # possibly load checkpoint
        self.load_model_ck()

    def load_model_ck(self):
        """
        load training checkpoint
        """
        ck_path = Path('best.pth')
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.model.load_state_dict(ck)

    def livetest(self, source):
        # type: (torch.Tensor) ->  np.array

        t = time.time()
        self.model.eval()
        self.model.requires_grad(False)

        pred = None

        #apply transofrm to input
        source = self.transform(source) #1080,1920,3 -> 3x1080x1920
        source.unsqueeze_(0)

        #pass compute model's output
        source = source.to(self.cnf.device)
        pred = self.model.forward(source) #1,1,135,240

        print("Inference time: {}".format(time.time() - t))

        return pred.squeeze()

def imtest(resolution = (1920,1080), conf_file_path = None, exp_name = None):

    cnf = Conf(conf_file_path=conf_file_path, exp_name=exp_name)
    print("Using log path: {}" .format(cnf.exp_log_path))

    model = QuadriFcn()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tester = LiveTester(cnf, model, transform)

    frame = cv2.imread('../Videotest/image3.png', cv2.IMREAD_COLOR) #BGR
    frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB) #RGB
    frame = resize_with_pad(frame, height=405, width=720) #1920x1080 -> 720x405

    #Get network prediction
    mask = tester.livetest(frame).cpu().numpy() #torch tensor -> np.array

    #preliminary post processing
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    #cool stuff
    mask = sigmoid(mask)
    mask[mask>0.5] = 1
    mask[mask<=0.5] = 0

    #Apply predicted mask to image
    mask = mask.astype(np.uint8)
    frame = cv2.cvtColor(frame, code=cv2.COLOR_RGB2BGR) #RGB

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3, cv2.LINE_4, hierarchy, 100)

    frame[:,:,1] = cv2.addWeighted(mask, 100, frame[:,:,1], 1, 0)

    #plt.show()
    cv2.imshow("window", frame)
    cv2.waitKey()

def videotest(resolution = (1920,1080), conf_file_path = None, exp_name = None):

    cnf = Conf(conf_file_path=conf_file_path, exp_name=exp_name)
    print("Using log path: {}" .format(cnf.exp_log_path))

    model = QuadriFcn()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tester = LiveTester(cnf, model, transform)
    cap = cv2.VideoCapture('../Videotest/videos/media2.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (720, 405))

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)  # RGB
        frame = resize_with_pad(frame, height=405, width=720)  # 1920x1080 -> 720x405

        # Get network prediction
        mask = tester.livetest(frame).cpu().numpy()  # torch tensor -> np.array

        # preliminary post processing
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # cool stuff
        mask = sigmoid(mask)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        # Apply predicted mask to image
        mask = mask.astype(np.uint8)
        frame = cv2.cvtColor(frame, code=cv2.COLOR_RGB2BGR)  # RGB

        # Draw the contours on the image
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3, cv2.LINE_4, hierarchy, 100)

        frame[:, :, 1] = cv2.addWeighted(mask, 100, frame[:, :, 1], 1, 0)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    videotest(exp_name='local')