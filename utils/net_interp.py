# Network interpolation
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('path_in1', type=str)
parser.add_argument('path_in2', type=str)
parser.add_argument('--path-out', type=str, default='ckp_interpolated.pkl')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--save-compl', action='store_true', help='save complete state')

args = parser.parse_args()

# Load params
ckp1 = torch.load(args.path_in1)
if 'state_dict' in ckp1:
    ckp1 = ckp1['state_dict']
ckp2 = torch.load(args.path_in2)
if 'state_dict' in ckp2:
    ckp2 = ckp2['state_dict']

# Check consistence
assert all(k in ckp2 and v.shape == ckp2.get(k).shape for k, v in ckp1.items())

# Check alpha
assert 0 < args.alpha < 1

# Interpolation
ckp3 = {k: ckp1.get(k)*args.alpha + ckp2.get(k)*(1-args.alpha) for k in ckp1}

# Store new checkpoint
if args.save_compl:
    ckp3 = {'state_dict': ckp3, 'max_acc': 0.0, 'epoch': 0}

torch.save(ckp3, args.path_out)
