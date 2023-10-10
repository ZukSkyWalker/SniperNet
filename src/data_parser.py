from optparse import OptionParser
import torch

from data_process import waymo_data_util


if __name__ == "__main__":
	parser = OptionParser()
	parser.add_option("-i", "--in_dir", dest="in_dir",
									 default="/run/user/1000/gvfs/smb-share:server=10.11.0.11,share=share/Zukai/wm_data/tf_1/")
	parser.add_option("-o", "--out_dir", dest="out_dir", default="/home/zukai/dev/waymo_data/pt1/")

	(options, args) = parser.parse_args()

	print(options.in_dir)
	print(options.out_dir)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	waymo_data_util.save_frames(options.in_dir, options.out_dir, device)
