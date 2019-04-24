# -*- coding: utf-8 -*-
# ---------------------

import math
from datetime import datetime
from time import time

import torch
import torchvision as tv
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from conf import Conf
from dataset.quadri_dataset import DSMode, QuadriDataset
from models import QuadriFcn, QuadriNetFancy

from utils import *
from metrics import compute_metrics
from focalloss import FocalLoss

"""
EXP:
	default_01: rete fcn-8
	default_02: rete fcn-8 con focal loss (quali parametri?)
	default_03_b: rete nostra con bug somma
	default_03: rete nostra senza bug e senza residual3
	default: rete nostra senza bug e CON residual3
	default_* (todo): rete fcn-8 trainata da 0 per 100 epoche
"""

NUM_LOG_IMGS = 10

class Trainer(object):

	def __init__(self, cnf, test_mode=False):
		# type: (Conf) -> Trainer

		self.cnf = cnf
		self.test_mode = test_mode

		# init model
		self.model = QuadriNetFancy(n_class=1)#QuadriFcn()
		self.model = self.model.to(cnf.device)

		# init optimizer
		self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr)

		#Load and split dataset in train - test
		train_dataset = QuadriDataset(cnf=cnf, mode=DSMode.TRAIN)
		test_dataset = QuadriDataset(cnf=cnf, mode=DSMode.VAL)


		# init train loader
		self.train_loader = DataLoader(
			dataset=train_dataset, batch_size=cnf.batch_size, num_workers=cnf.n_workers, shuffle=True
		)

		# init test loader
		self.test_loader = DataLoader(
			dataset=test_dataset, batch_size=1, num_workers=cnf.n_workers, shuffle=False
		)

		#define inverse normalization function, userful for displayng purposes
		self.inv_transform = tv.transforms.Compose([
			tv.transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
			tv.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
		])

		# init logging stuffs
		self.log_path = cnf.exp_log_path
		print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
		self.sw = SummaryWriter(self.log_path)
		self.log_freq = len(self.train_loader)
		self.train_losses = []
		self.test_losses = []

		# starting values values
		self.epoch = 0
		self.best_test_IoU = None

		if not test_mode:
			print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
			self.sw = SummaryWriter(self.log_path)

		# in test model load the best model (if available)
		if test_mode:
			ck_path = 'best.pth'
			# ck_path = self.log_path/'best.pth'
			if Path(ck_path).exists():
				print(f'[loading checkpoint \'{ck_path}\']')
				ck = torch.load(ck_path)
				self.model.load_state_dict(ck)
		else:
			# possibly load checkpoint
			self.load_ck()


	def load_ck(self):
		"""
		load training checkpoint
		"""
		ck_path = self.log_path/'training.ck'
		if ck_path.exists():
			ck = torch.load(ck_path)
			print(f'[loading checkpoint \'{ck_path}\']')
			self.epoch = ck['epoch']
			self.model.load_state_dict(ck['model'])
			self.optimizer.load_state_dict(ck['optimizer'])
			self.best_test_IoU = self.best_test_IoU


	def save_ck(self):
		"""
		save training checkpoint
		"""
		ck = {
			'epoch': self.epoch,
			'model': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'best_test_IoU': self.best_test_IoU
		}
		torch.save(ck, self.log_path/'training.ck')


	def train(self):
		"""
		train model for one epoch on the Training-Set.
		"""
		start_time = time()
		self.model.train()
		self.model.requires_grad(True)

		times = []
		for step, sample in enumerate(self.train_loader):
			t = time()

			self.optimizer.zero_grad()

			x, y_true = sample
			x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)

			y_pred = self.model.forward(x)
			loss = nn.BCEWithLogitsLoss()(y_pred, y_true)
			#loss = FocalLoss(alpha=0.4, gamma=1)(y_pred, y_true)
			loss.backward()
			self.train_losses.append(loss.item())

			self.optimizer.step(None)

			# print an incredible progress bar
			progress = (step + 1)/len(self.train_loader)
			progress_bar = ('█'*int(50*progress)) + ('┈'*(50 - int(50*progress)))
			times.append(time() - t)
			if self.cnf.log_each_step or (not self.cnf.log_each_step and progress == 1):
				print('[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}% │ Loss: {:.6f} │ ↯: {:5.2f} step/s'.format(
					datetime.now().strftime("%m-%d@%H:%M"), self.epoch, step + 1,
					progress_bar, 100*progress,
					np.mean(self.train_losses), 1/np.mean(times),
					e=math.ceil(math.log10(self.cnf.epochs)),
					s=math.ceil(math.log10(self.log_freq)),
				), end='\r', flush=True)

		# log average loss of this epoch
		#TODO: save berst IoU not best TestLoss
		mean_epoch_loss = np.mean(self.train_losses)  # type: float
		self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
		self.train_losses = []

		# log epoch duration
		print(f' │ T: {time()-start_time:.2f} s')


	def test(self):
		"""
		test model on the Test-Set
		"""

		self.model.eval()
		self.model.requires_grad(False)

		t = time()
		limgs = 0

		self.test_losses = []
		test_ac = []
		test_iou = []

		for step, sample in enumerate(self.test_loader):
			x, y_true = sample
			x, y_true = x.to(self.cnf.device), y_true.to(self.cnf.device)
			y_pred = self.model.forward(x)

			loss = nn.BCEWithLogitsLoss()(y_pred, y_true)
			#loss = FocalLoss(alpha=0.4, gamma=1)(y_pred, y_true)
			self.test_losses.append(loss.item())

			#compute metrics
			metrics = compute_metrics(y_true, y_pred)
			ac = metrics['ac']
			iou = metrics['iou']
			test_ac.append(ac)
			test_iou.append(iou)

			# Log exactly num_logimgs images in TB
			global NUM_LOG_IMGS
			print_cond = step % (len(self.test_loader) // NUM_LOG_IMGS) == 0
			print_cond = print_cond and (self.test_mode is False)

			if (print_cond and limgs < NUM_LOG_IMGS):
				# draw results for this step in a 3 rows grid:
				# row #1: input (x)
				# row #2: predicted_output (y_pred)
				# row #3: target (y_true)

				# de-normalize
				for i in x:
					self.inv_transform(i)

				x = x.cpu()
				yp = overlap_image_mask(x, y_pred)
				yt = overlap_image_mask(x, y_true)

				grid = torch.cat([x, yp, yt], dim=0)
				grid = tv.utils.make_grid(grid, normalize=True, range=(0, 1), nrow=x.shape[0])
				self.sw.add_image(tag=f'results_{step}', img_tensor=grid, global_step=self.epoch)
				limgs += 1

		# log average loss on test set
		mean_test_loss = float(np.mean(self.test_losses))
		mean_test_ac = float(np.mean(test_ac))
		mean_test_iou = float(np.mean(test_iou))

		# print test metrics
		print(
			f'\t● AVG (LOSS, AC, IOU) on TEST-set: '
			f'({mean_test_loss:.6f}, '
			f'{mean_test_ac:.6f}, '
			f'{mean_test_iou:.6f}) ',
			end=''
		)
		print(f'│ T: {time() - t:.2f} s')

		if not self.test_mode:
			self.sw.add_scalar(tag='test/mLoss', scalar_value=mean_test_loss, global_step=self.epoch)
			self.sw.add_scalar(tag='test/mAccuracy', scalar_value=mean_test_ac, global_step=self.epoch)
			self.sw.add_scalar(tag='test/mIOU', scalar_value=mean_test_iou, global_step=self.epoch)

			# save best model
			if self.best_test_IoU is None or mean_test_iou > self.best_test_IoU:
				self.best_test_IoU = mean_test_iou
				torch.save(self.model.state_dict(), self.log_path/'best.pth')


	def run(self):
		"""
		start model training procedure (train > test > checkpoint > repeat)
		"""
		if self.test_mode:
			self.test()
		else:
			for _ in range(self.epoch, self.cnf.epochs):
				self.train()
				self.test()
				self.epoch += 1
				self.save_ck()
