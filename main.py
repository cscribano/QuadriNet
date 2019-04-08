# -*- coding: utf-8 -*-
# ---------------------

import matplotlib

matplotlib.use('Agg')

from conf import Conf

import click
import torch.backends.cudnn as cudnn

from trainer import Trainer

cudnn.benchmark = True


@click.command()
@click.option('--exp_name', type=str, default=None)
@click.option('--conf_file_path', type=str, default=None)
@click.option('--seed', type=int, default=None)
def main(exp_name, conf_file_path, seed):
	# type: (str, str, int) -> None

	# if `exp_name` is None,
	# ask the user to enter it
	if exp_name is None:
		exp_name = input('>> experiment name: ')

	# if `exp_name` contains '!',
	# `log_each_step` becomes `False`
	log_each_step = True
	if '!' in exp_name:
		exp_name = exp_name.replace('!', '')
		log_each_step = False

	# if `exp_name` contains a '@' character,
	# the number following '@' is considered as
	# the desired random seed for the experiment
	split = exp_name.split('@')
	if len(split) == 2:
		seed = int(split[1])
		exp_name = split[0]

	cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name, log=log_each_step)

	print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

	trainer = Trainer(cnf=cnf)
	trainer.run()


if __name__ == '__main__':
	main()
