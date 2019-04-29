from importlib import import_module


_archs = {
    'SRResNet': 'srnet',
    'SRGAN': 'srgan',
    'VDSR': 'vdsr',
    'WDSR': 'wdsr_b',
    'EDSR': 'edsr'
}


def get_model(arch):
    module_name = _archs.get(arch)
    if module_name is not None:
        module = import_module('.'+module_name, package=__package__)
        return hasattr(module, '__MM__') and getattr(module, getattr(module, '__MM__'))
    else:
        return False

        
def build_model(arch, *opts, **kopts):
    model = get_model(arch)
    if not model:
        raise ValueError('{} is not supported'.format(arch))
    return model(*opts, **kopts)
