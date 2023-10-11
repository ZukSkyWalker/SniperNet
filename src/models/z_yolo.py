import torch.nn as nn

BN_MOMENTUM = 0.1

class Zolo(nn.Module):
	def __init__(self) -> None:
		super(Zolo, self).__init__()
		self.num_classes = 3

		# Define a shared backbone
		self.backbone = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
		)

		# Output heads
		self.get_influence  = nn.Conv2d(64, 1, kernel_size=1)
		self.get_class_type = nn.Conv2d(64, 3, kernel_size=1)  # 3 classes
		self.get_cxy_offset = nn.Conv2d(64, 2, kernel_size=1)
		self.get_heading    = nn.Conv2d(64, 1, kernel_size=1)
		self.get_z_coor     = nn.Conv2d(64, 1, kernel_size=1)
		self.get_dim        = nn.Conv2d(64, 3, kernel_size=1)  # 3 dimensions

	def forward(self, x):
		x = self.backbone(x)

		return {"influence":  self.get_influence(x),
						"class_type": self.get_class_type(x),
						"cxy_offset": self.get_cxy_offset(x),
						"heading":    self.get_heading(x),
						"z_coor":     self.get_z_coor(x),
						"dim":        self.get_dim(x)}
