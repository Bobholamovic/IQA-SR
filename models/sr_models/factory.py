from importlib import import_module


_archs = {
    'SRResNet': 'srnet'
}


def make_model(arch):
    module_name = _archs.get(arch)
    if module_name is not None:
        module = import_module('.'+module_name, package=__package__)
        return hasattr(module, '__MM__') and getattr(module, getattr(module, '__MM__'))
    else:
        return False
