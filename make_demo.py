# -*- coding: utf-8 -*-
# ---------------------
import click

import torch
from torchvision import transforms
from models import BaseModel

import time
import cv2
import numpy as np
from path import Path

from metrics import sigmoid
from utils import resize_with_pad
from models import BaseModel, QuadriFcn, QuadriNetFancy

###############################
## PLEASE CONSULT Readme.TXT ##
###############################

VExtensions = ["avi", "mp4"]
IExtensions = ["png", "jpg", "jpeg"]

@click.command()
@click.option('--source', '-s', type=click.Path(exists=True), default=None)
@click.option('--device', '-d', type=str, default='cpu', help="'cpu' or 'cuda'")
@click.option('--net', '-n', type=str, default='our', help="'our' or 'fcn8'")
@click.option('--weights', '-w', type=click.Path(exists=True), default=None, help="Pre-trained network weights")
@click.option('--output', '-o', type=str, default='output', help="Destination file")
def main(source, device, net, weights, output):

    #get source path
    if source is None:
        source = input('>> Insert source path: ')

    #check the provided input file
    source_extension = source.split(".")[-1]
    video = False

    if source_extension in VExtensions:
        video = True
    elif source_extension in IExtensions:
        pass
    else:
        print("Invalid input format, please retry")
        exit(1)

    if weights is None:
        weights = input('>> Insert Pre-trained wights path (.pth): ')

    output = output+'.'+source_extension

    print(f'>> Selected output file {output}')

    #Load the nn module
    if net == 'our':
        model = QuadriNetFancy()
    else:
        model = QuadriFcn()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tester = LiveTester(model, Path(weights), device, transform)

    #call the source processer
    if video:
        videotest(tester=tester, source=source, out=output)
    else:
        imtest(tester=tester, source=source, out=output)

    print(">> All done, bye!")

def imtest(tester, source, out):

    frame = cv2.imread(source, cv2.IMREAD_COLOR) #BGR
    frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB) #RGB
    frame = resize_with_pad(frame, height=405, width=720) #1920x1080 -> 720x405

    #Get network prediction
    mask = tester.livetest(frame).cpu().numpy() #torch tensor -> np.array
    frame = postprocess(mask, frame)

    cv2.imwrite(out, frame)

def videotest(tester, source, out):

    cap = cv2.VideoCapture(source)
    lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    dest = cv2.VideoWriter(out, fourcc, 30.0, (720, 405))

    print(">> Processing video, wait or press 'q' to abort")

    step = 0
    while cap.isOpened():
        _, frame = cap.read()
        step += 1

        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)  # RGB
        frame = resize_with_pad(frame, height=405, width=720)  # 1920x1080 -> 720x405

        # Get network prediction and postprocess
        mask = tester.livetest(frame).cpu().numpy()  # torch tensor -> np.array
        frame = postprocess(mask, frame)

        #█████-----
        progress = (step + 1) / lenght
        progress_bar = ('█' * int(50 * progress)) + ('-' * (50 - int(50 * progress)))
        print('\r│{}│ : {}/{}'.format(progress_bar, step, lenght), end='')

        dest.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class LiveTester:

    def __init__(self, model, weights, device='cpu', transform=None):
        # type: (BaseModel, Path, str, transforms) -> LiveTester

        self.device = device
        self.transform = transform
        self.ck_path = weights

        # init model
        self.model = model
        self.model = self.model.to(device)

        # load trained weights
        self.load_model_weights()

    def load_model_weights(self):
        """
        load network weights
        """
        if self.ck_path.exists():
            print(f'[loading weights \'{self.ck_path}\']')
            try:
                ck = torch.load(self.ck_path)
                self.model.load_state_dict(ck)
            except():
                print(">> Can't load nn weights, aborting")
                exit(1)

            print("Loaded OK")

    def livetest(self, source):
        # type: (torch.Tensor) ->  np.array

        t = time.time()
        self.model.eval()
        self.model.requires_grad(False)

        pred = None

        #apply transofrm to input
        source = self.transform(source)
        source.unsqueeze_(0)

        #pass compute model's output
        source = source.to(self.device)
        pred = self.model.forward(source)

        return pred.squeeze()

def postprocess(mask, frame):

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

    #putting together
    frame[:,:,1] = cv2.addWeighted(mask, 100, frame[:,:,1], 1, 0)
    return frame

if __name__ == '__main__':
    main()
