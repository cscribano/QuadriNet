# -*- coding: utf-8 -*-
# ---------------------

import os

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
	os.environ['PYTHONPATH'] = PYTHONPATH
else:
	os.environ['PYTHONPATH'] += (':' + PYTHONPATH)

import yaml
import socket
import random
import torch
import numpy as np
from path import Path
from typing import Optional


def set_seed(seed=None):
	# type: (Optional[int]) -> int
	"""
	set the random seed using the required value (`seed`)
	or a random value if `seed` is `None`
	:return: the newly set seed
	"""
	if seed is None:
		seed = random.randint(1, 10000)
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	return seed


class Conf(object):
	HOSTNAME = socket.gethostname()
	OUT_PATH = Path('./')


	def __init__(self, conf_file_path=None, seed=None, exp_name=None, log=True, device='cuda'):
		# type: (str, int, str, bool, str) -> None
		"""
		:param conf_file_path: optional path of the configuration file
		:param seed: desired seed for the RNG; if `None`, it will be chosen randomly
		:param exp_name: name of the experiment
		:param log: `True` if you want to log each step; `False` otherwise
		:param device: torch device you want to use for train/test
			:example values: 'cpu', 'cuda', 'cuda:5', ...
		"""
		self.exp_name = exp_name
		self.log_each_step = log
		self.device = device

		# print project name and host name
		self.project_name = Path(__file__).parent.parent.basename()
		m_str = f'┃ {self.project_name}@{Conf.HOSTNAME} ┃'
		u_str = '┏' + '━'*(len(m_str) - 2) + '┓'
		b_str = '┗' + '━'*(len(m_str) - 2) + '┛'
		print(u_str + '\n' + m_str + '\n' + b_str)

		# set random seed
		self.seed = set_seed(seed)  # type: int
		
		# if the configuration file is not specified
		# try to load a configuration file based on the experiment name
		tmp = Path('conf')/(self.exp_name + '.yaml')
		if (conf_file_path is None) and tmp.exists():
			conf_file_path = tmp

		# read the YAML configuation file
		if conf_file_path is None:
			y = {}
		else:
			conf_file = open(conf_file_path, 'r')
			y = yaml.load(conf_file)

		# define output paths
		tmp_path = y.get('OUTPUT_PATH', None)  # type: str
		if tmp_path is not None:
			Conf.OUT_PATH = Path(tmp_path)

		self.project_log_path = Path(Conf.OUT_PATH / 'log' / self.project_name)
		self.exp_log_path = self.project_log_path/exp_name

		# read configuration parameters from YAML file
		# or set their default value
		self.epochs = y.get('EPOCHS', 10)  # type: int
		self.lr = y.get('LR', 0.0001)  # type: float
		self.batch_size = y.get('BATCH_SIZE', 8)  # type: int
		self.n_workers = y.get('N_WORKERS', 0)  # type: int
		if self.device == 'cuda' and y.get('DEVICE', None) is not None:
			self.device = y.get('DEVICE')  # type: str
		self.num_logimgs = y.get('NLOG_IMGS', 10) # type: int
		self.dataset_path = Path(y.get('DATASET', None)) # type: Path

	@property
	def is_cuda(self):
		# type: () -> bool
		"""
		:return: `True` if the required device is 'cuda'; `False` otherwise
		"""
		return 'cuda' in self.device
