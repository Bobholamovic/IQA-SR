#!/usr/bin/env python3
import argparse
import os
import yaml
import shutil
from time import localtime
from os.path import join, basename, exists
from pdb import set_trace as db

from core.trainer import SRTrainer
from core.predictor import SRPredictor
from utils.misc import OutPathGetter

## Disturbing warnings from skimage
## Shut them off
import warnings
warnings.filterwarnings('ignore')


def read_config(config_path):
    f = open(config_path, 'r')
    cfg = yaml.load(f.read())
    f.close()
    return cfg or {}


def parse_config(cfg_name, cfg):
    # Parse feats
    if cfg.__contains__('feats'):
        # The feature names are stored in sequential along with their weights.
        # What is done here is to separate them into two tuples
        feat_names, weights = zip(*(tuple(*f.items()) for f in cfg['feats']))
        del cfg['feats']
        cfg = {**cfg, 'feat_names': feat_names, 'weights': weights}

    # Parse the name of config file
    sp = cfg_name.split('.')[0].split('_')
    if len(sp) >= 2:
        cfg.setdefault('tag', sp[1])
        cfg.setdefault('suffix', '_'.join(sp[1:]))
    else:
        new_str = '_'.join([str(k)+'-'+str(v) for k,v in sorted(cfg.items())])
        cfg.setdefault('tag', new_str)
        cfg.setdefault('suffix', new_str)
    
    return cfg


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'val', 'test'])
    parser.add_argument('-d', '--data-dir', default='/home/gdf/Datasets/DIV2K/')
    parser.add_argument('-l', '--list-dir', default='/home/gdf/Datasets/DIV2K/lists/X4/')
    parser.add_argument('-o', '--out-dir', default='')
    parser.add_argument('-p', '--patch-size', type=int, default=256, metavar='P', 
                        help='patch size (default: %(default)s)')
    parser.add_argument('--exp-dir', default='./exps/')
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: %(default)s)')
    parser.add_argument('--num-epochs', type=int, default=1000, metavar='NE',
                        help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: %(default)s)')
    parser.add_argument('--lr-mode', type=str, default='const')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: %(default)s)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--exp-config', type=str, default='')
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('-s', '--scale', type=int, default=4)
    parser.add_argument('--save-on', action='store_true')
    parser.add_argument('--log-off', action='store_true')
    parser.add_argument('--suffix-off', action='store_true')
    parser.add_argument('--anew', action='store_true',
                        help='clear history and start from epoch 0 with the checkpoint loaded')
    parser.add_argument('--continu', action='store_true')
    parser.add_argument('--iqa-patch-size', type=int, default=32)
    parser.add_argument('--criterion', type=str, default='MAE')
    parser.add_argument('--iqa-model-path', type=str, default='/home/gdf/Codes/CNN-NRIQA/models/ckp_MA.pkl')
    parser.add_argument('--trace-freq', type=int, default=50)
    parser.add_argument('--reproduce', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0, help='weighting coefficient of pixel loss')

    args = parser.parse_args()

    cfg_name = basename(args.exp_config)

    if exists(args.exp_config):
        cfg = read_config(args.exp_config)
        cfg = parse_config(cfg_name, cfg)
        # Settings from cfg file overwrite those in args
        args.__dict__.update(cfg)

    if args.cmd in ('train', 'val'):
        # Disable global path controller in test phase
        args.global_path = OutPathGetter(
                    root=os.path.join(args.exp_dir, args.tag), 
                    suffix=args.suffix)

        if exists(args.exp_config):
            # Make a copy of the config file
            cfg_path = args.global_path.get_path('root', cfg_name, suffix=False)
            shutil.copy(args.exp_config, cfg_path)

    return args


def main():
    args = parse_args()

    if args.cmd == 'train':
        solver = SRTrainer(args)
        solver.train()
    elif args.cmd == 'val':
        solver = SRTrainer(args)
        solver.validate()
    elif args.cmd == 'test':
        opts = dict(
            scale=args.scale,
            data_dir=args.data_dir,
            ckp_path=args.resume,
            save_lr=args.save_on,
            list_dir=args.list_dir,
            out_dir=args.out_dir
        )
        if not args.log_off:
            opts['log_dir'] = args.out_dir
        solver = SRPredictor(**opts)
        solver.test()
    else:
        pass


if __name__ == '__main__':
    main()
