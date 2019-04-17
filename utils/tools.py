import logging
import os
from time import localtime

FORMAT_LONG = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
FORMAT_SHORT = "%(message)s"

class Logger:
    _count = 0

    def __init__(self, scrn=True, log_dir='', phase=''):
        super().__init__()
        self._logger = logging.getLogger('logger_{}'.format(Logger._count))
        self._logger.setLevel(logging.DEBUG)

        if scrn:
            self._scrn_handler = logging.StreamHandler()
            self._scrn_handler.setLevel(logging.INFO)
            self._scrn_handler.setFormatter(logging.Formatter(fmt=FORMAT_SHORT))
            self._logger.addHandler(self._scrn_handler)
            
        if log_dir and phase:
            self.log_path = os.path.join(log_dir,
                    '{}-{:-4d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}.log'.format(
                        phase, *localtime()[:6]
                      ))
            self.show_nl("log into {}\n\n".format(self.log_path))
            self._file_handler = logging.FileHandler(filename=self.log_path)
            self._file_handler.setLevel(logging.DEBUG)
            self._file_handler.setFormatter(logging.Formatter(fmt=FORMAT_LONG))
            self._logger.addHandler(self._file_handler)

    def show(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)

    def show_nl(self, *args, **kwargs):
        self._logger.info("")
        return self.show(*args, **kwargs)

    def dump(self, *args, **kwargs):
        return self._logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self._logger.error(*args, **kwargs)


class OutPathGetter:
    def __init__(self, root='', log='logs', out='outs', weight='weights', suffix=''):
        super().__init__()
        self._root = root
        self._suffix = suffix
        self._sub_dirs = dict(log=log, out=out, weight=weight)

        self.build_dir_tree()

    @property
    def sub_dirs(self):
        return tuple(self._sub_dirs)

    @property
    def root(self):
        return self._root

    def build_dir_tree(self):
        for key, name in self._sub_dirs.items():
            folder = os.path.join(self._root, name)
            self._sub_dirs[key] = folder    # Update
            self.make_dirs(folder)

    @staticmethod
    def make_dirs(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_dir(self, key):
        return self._sub_dirs.get(key, '')

    def get_path(self, key, file, auto_make=False, suffix=True, underline=False):
        folder = self.get_dir(key)
        if suffix:
            path = os.path.join(folder, self.add_suffix(file, underline=underline))
        else:
            path = os.path.join(folder, file)

        if auto_make and path:
            self.make_dirs(os.path.dirname(path))
        return path

    def add_suffix(self, path, underline=False):
        pos = path.rfind('.')
        assert pos > -1
        return path[:pos] + ('_' if underline and self._suffix else '') + self._suffix + path[pos:]

