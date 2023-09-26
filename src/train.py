import torch
import time
import numpy as np
import tqdm

from utils.misc import AverageMeter
from utils.torch_utils import reduce_tensor, to_python_float
from losses.losses import Compute_Loss


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, tb_writer):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')

	criterion = Compute_Loss(device=configs.device)
	num_iters_per_epoch = len(train_dataloader)
	# switch to train mode
	model.train()
	start_time = time.time()
	for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
		data_time.update(time.time() - start_time)
		metadatas, imgs, targets = batch_data
		batch_size = imgs.size(0)
		global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1
		for k in targets.keys():
			targets[k] = targets[k].to(configs.device, non_blocking=True)
		imgs = imgs.to(configs.device, non_blocking=True).float()
		outputs = model(imgs)
		total_loss, loss_stats = criterion(outputs, targets)
		# For torch.nn.DataParallel case
		if (not configs.distributed) and (configs.gpu_idx is None):
			total_loss = torch.mean(total_loss)

		# compute gradient and perform backpropagation
		total_loss.backward()
		if global_step % configs.subdivisions == 0:
			optimizer.step()
			# zero the parameter gradients
			optimizer.zero_grad()
			# Adjust learning rate
			if configs.step_lr_in_epoch:
				lr_scheduler.step()
				if tb_writer is not None:
					tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)

		if configs.distributed:
			reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
		else:
			reduced_loss = total_loss.data
		losses.update(to_python_float(reduced_loss), batch_size)
		# measure elapsed time
		# torch.cuda.synchronize()
		batch_time.update(time.time() - start_time)

		if tb_writer is not None:
			if (global_step % configs.tensorboard_freq) == 0:
				loss_stats['avg_loss'] = losses.avg
				tb_writer.add_scalars('Train', loss_stats, global_step)


		start_time = time.time()
