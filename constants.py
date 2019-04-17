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
IMAGE_POSTFIXES = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# Network architecture
ARCH = 'SRResNet'
