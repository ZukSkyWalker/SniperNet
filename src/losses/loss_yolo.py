import torch.nn as nn
import torch
import torch.nn.functional as F


class Compute_Loss(nn.Module):
	def __init__(self):
		super(Compute_Loss, self).__init__()
		self.weight_pos = 1.
		self.weight_dim = 1.
		self.weight_z   = 1.


	def forward(self, outputs, targets):
		cxy_loss = (outputs['influence'] * (outputs['cxy_offset'] - targets['class_1hot'])**2).sum().item()
		z_loss = (outputs['influence'] * (outputs['z_coor'] - targets['z_coor'])**2).sum().item()
		heading_loss = (outputs['influence'] * torch.cos(outputs['heading'] - targets['heading'])).sum().item()
		dim_loss = (outputs['influence'] * (outputs['dim'] - targets['dimension'])).sum().item()
		influence_loss = ((outputs['influence'] - targets['heat_map'])**2).sum().item()
		classification_loss = ((outputs['class_type'] - targets['heat_map'])**2).sum().item()

		return self.weight_pos * cxy_loss + self.weight_z * z_loss + heading_loss + dim_loss + influence_loss + classification_loss
