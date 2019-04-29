# Global constants

# Sampling related
GAUSSIAN_BLUR = True
_INTERP_METHODS = {
    'NEAREST': 0, 
    'BILINEAR': 1, 
    'BICUBIC': 3
}
INTERP_ORDER = _INTERP_METHODS['BICUBIC']

# Image postfixes to recognize in a given directory
IMAGE_POSTFIXES = ['.jpg', '.jpeg',     '.png', '.bmp', '.tif', '.tiff']

# Network architecture
ARCH = 'EDSR'

# Dataset on training
DATASET = 'DIV2K'
# Expected names of data lists
IMAGE_LIST_PATTERN = "{ph!s}_list.txt"
LR_LIST_PATTERN = "{ph!s}_list_lr.txt"

# Checkpoint patterns
CKP_LATEST = "checkpoint_latest_{}_{}".format(ARCH, DATASET)+"_x{s}.pkl"
CKP_BEST = "model_best_{}_{}".format(ARCH, DATASET)+"_x{s}.pkl"
CKP_COUNTED = "checkpoint_{e:03d}"+"_{}_{}".format(ARCH, DATASET)+"_x{s}.pkl" 

# Phases
# To be added
