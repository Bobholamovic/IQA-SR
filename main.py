#!/usr/bin/env python3
import argparse
import os
import yaml
import shutil
from time import localtime
from os.path import join, basename, exists
from pdb import set_trace as db

from core.trainer import SRTrainer
from utils.tools import OutPathGetter

## Disturbing warnings from skimage
## Shut them off
import warnings
warnings.filterwarnings('ignore')


def read_config(config_path):
    f = open(config_path, 'r')
    cfg = yaml.load(f.read())
    if cfg is None:
        cfg = {}
    return cfg


def parse_config(cfg_name, cfg):
    # Parse feats
    if cfg.get('feats'):
        feat_names, weights = zip(*(tuple(*f.items()) for f in cfg['feats']))
        del cfg['feats']
        cfg = {**cfg, 'feat_names': feat_names, 'weights': weights}

    # Parse the name of config file
    sp = cfg_name.split('.')[0].split('_')
    if len(sp) >= 2:
        cfg['tag'] = sp[1]
        cfg['suffix'] = '_'.join(sp[1:])
    else:
        cfg['tag'] = cfg['suffix'] = '_'.join([str(k)+'-'+str(v) for k,v in sorted(cfg.items())])

    return cfg


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test', 'val'])
    parser.add_argument('-d', '--data-dir', default='/media/gdf/0005898C000F299F/FDisk/gMAD/pristine_images/')
    parser.add_argument('-l', '--list-dir', default='/media/gdf/0005898C000F299F/FDisk/gMAD/pristine_images/SR/')
    parser.add_argument('-o', '--out-dir', default='./test/')
    parser.add_argument('-p', '--patch-size', type=int, default=256, metavar='P', 
                        help='patch size (default: 64)')
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='NE',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-mode', type=str, default='const')
    parser.add_argument('--weight-decay', default=5e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--exp-config', type=str, default='')
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--save-off', action='store_true')
    parser.add_argument('-s', '--scale', type=int, default=4)
    parser.add_argument('--log-off', action='store_true')
    parser.add_argument('--iqa-patch-size', type=int, default=32)
    parser.add_argument('--criterion', type=str, default='MAE')
    parser.add_argument('--iqa-model-path', type=str, default='/home/gdf/Codes/CNN-FRIQA/models/ckp_n8_p32_d3.pkl')
    parser.add_argument('--trace-freq', type=int, default=50)

    args = parser.parse_args()

    cfg_name = basename(args.exp_config)

    if exists(args.exp_config):
        cfg = read_config(args.exp_config)
        cfg = parse_config(cfg_name, cfg)
        args.__dict__.update(cfg)

    args.global_path = OutPathGetter(
                root=os.path.join(args.out_dir, args.tag), 
                suffix=args.suffix)

    cfg_path = os.path.join(args.global_path.root, cfg_name)
    if exists(args.exp_config) and not exists(cfg_path):
        # Make a copy of the config file
        shutil.copy(args.exp_config, cfg_path)

    return args


def main():
    args = parse_args()

    if args.cmd == 'train':
        trainer = SRTrainer(args)
        trainer.train()
    elif args.cmd == 'val':
        trainer = SRTrainer(args)
        trainer.validate()
    else:
        pass


if __name__ == '__main__':
    main()
