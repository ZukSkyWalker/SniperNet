import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
	def __init__(self, inplanes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.stride = stride

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		# Add residual: x itself
		out += x
		out = self.relu(out)

		return out

class ZResNet(nn.Module):
	def __init__(self, block, layers, heads, head_conv):
		self.inplanes = 64
		self.heads = heads

		super(ZResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.conv_up_level1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
		self.conv_up_level2 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
		self.conv_up_level3 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)

		fpn_channels = [256, 128, 64]
		for fpn_idx, fpn_c in enumerate(fpn_channels):
			for head in sorted(self.heads):
				num_output = self.heads[head]
				if head_conv > 0:
					fc = nn.Sequential(
						nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1, bias=True),
						nn.ReLU(inplace=True),
						nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
				else:
					fc = nn.Conv2d(in_channels=fpn_c, out_channels=num_output, kernel_size=1, stride=1, padding=0)

				self.__setattr__(f'fpn{fpn_idx}_{head}', fc)

	def _make_layer(self, block, planes, blocks, stride=1):
		layers = []
		layers.append(block(self.inplanes, planes, stride))
		self.inplanes = planes
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		_, _, input_h, input_w = x.size()

		print("x shape: ", x.size())

		hm_h, hm_w = input_h // 4, input_w // 4
		x = self.conv1(x)
		print("after conv1", x.size())

		x = self.bn1(x)
		print("after bn1", x.size())

		x = self.relu(x)
		print("after relu", x.size())

		x = self.maxpool(x)
		print("after maxpool", x.size())

		out_layer1 = self.layer1(x)
		print("out_layer1 shape:", out_layer1.size())

		out_layer2 = self.layer2(out_layer1)
		print("out_layer2 shape:", out_layer2.size())

		out_layer3 = self.layer3(out_layer2)
		print("out_layer3 shape:", out_layer3.size())

		out_layer4 = self.layer4(out_layer3)
		print("out_layer4 shape:", out_layer4.size())

		# up_level1: torch.Size([b, 512, 14, 14])
		up_level1 = F.interpolate(out_layer4, scale_factor=2, mode='bilinear', align_corners=True)
		print("up_layer1 shape:", up_level1.size(), "; out_layer3", out_layer3.size())

		concat_level1 = torch.cat((up_level1, out_layer3), dim=1)
		# up_level2: torch.Size([b, 256, 28, 28])
		up_level2 = F.interpolate(self.conv_up_level1(concat_level1), scale_factor=2, mode='bilinear',
															align_corners=True)

		concat_level2 = torch.cat((up_level2, out_layer2), dim=1)
		# up_level3: torch.Size([b, 128, 56, 56]),
		up_level3 = F.interpolate(self.conv_up_level2(concat_level2), scale_factor=2, mode='bilinear',
															align_corners=True)
		# up_level4: torch.Size([b, 64, 56, 56])
		up_level4 = self.conv_up_level3(torch.cat((up_level3, out_layer1), dim=1))

		ret = {}
		for head in self.heads:
			temp_outs = []
			for fpn_idx, fdn_input in enumerate([up_level2, up_level3, up_level4]):
				fpn_out = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))(fdn_input)
				_, _, fpn_out_h, fpn_out_w = fpn_out.size()
				# Make sure the added features having same size of heatmap output
				if (fpn_out_w != hm_w) or (fpn_out_h != hm_h):
					fpn_out = F.interpolate(fpn_out, size=(hm_h, hm_w))
				temp_outs.append(fpn_out)
			# Take the softmax in the keypoint feature pyramid network
			final_out = self.apply_kfpn(temp_outs)

			ret[head] = final_out

		return ret

	def apply_kfpn(self, outs):
		outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
		softmax_outs = F.softmax(outs, dim=-1)
		ret_outs = (outs * softmax_outs).sum(dim=-1)
		return ret_outs

def get_model():
	model = ZResNet(BasicBlock, [2, 2, 2, 2],
								 heads={'num_classes': 3, "cxy_offset": 2, "heading": 1, "z_coord": 1, "dim": 3}, head_conv=64)
	
	return model
