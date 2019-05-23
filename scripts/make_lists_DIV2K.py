from glob import glob
from tqdm import tqdm
import os
from os.path import join, exists, splitext, basename

PRE = 'DIV2K'
PHASE = ('train', 'valid')
SCALE = ('X2', 'X3', 'X4')

OUT_DIR = './lists'
if not exists(OUT_DIR):
    os.mkdir(OUT_DIR)
    for s in SCALE:
        os.makedirs(join(OUT_DIR, s))

def set_var(name, value):
    globals()[name] = value

def get_var(name):
    return globals().get(name, None)

def gsv(scale, name):
    return get_var('_'.join([scale, name]))

def ssv(scale, name, value):
    return set_var('_'.join([scale, name]), value)

def suf(file_name, suffix):
    sp = splitext(file_name)
    return sp[0]+suffix+sp[1]

def write_line(handler, line):
    return handler.write(line + '\n')

for p in PHASE:
    hr_folder = '_'.join([PRE, p, 'HR'])
    lr_folder_ = '_'.join([PRE, p, 'LR_bicubic'])

    # To suit the code
    if p == 'valid':
        p = 'val'
    for s in SCALE:
        out_dir = join(OUT_DIR, s)
        ssv(s, 'file_lr', open(join(out_dir, '_'.join([p, 'list_lr.txt'])), 'w'))
        ssv(s, 'file', open(join(out_dir, '{}_list.txt'.format(p)), 'w'))
    
    hr_list = sorted(glob(join(hr_folder, '*.png')))
    
    for hr_name in tqdm(hr_list):
        for s in SCALE:
            lr_dir = join(lr_folder_, s)
            # Note that lower-case x2x3x4 in file names while upper cases in dir names
            write_line(gsv(s, 'file_lr'), join(lr_dir, suf(basename(hr_name), s.lower())))
            write_line(gsv(s, 'file'), hr_name)

    for s in SCALE:
        gsv(s, 'file_lr').close()
        gsv(s, 'file').close()
    
	

