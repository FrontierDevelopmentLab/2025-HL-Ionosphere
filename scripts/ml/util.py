import sys
import traceback
import random
import time
import numpy as np
import torch


class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def set_random_seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_as_channels(features, image_size=(180,360)):
    if not isinstance(features, list) and not isinstance(features, tuple):
        raise ValueError('Expecting a list or tuple of features')
    c = []
    for f in features:
        if not isinstance(f, torch.Tensor):
            f = torch.tensor(f, dtype=torch.float32)
        if f.ndim == 0:
            f = f.expand(image_size)
            f = f.unsqueeze(0)
        elif f.ndim == 1:
            f = f.view(-1, 1, 1)
            f = f.expand((-1,) + image_size)
        elif f.shape == image_size:
            f = f.unsqueeze(0)  # add channel dimension
        else:
            raise ValueError('Expecting 0d or 1d features, or 2d features with shape equal to image_size')
        c.append(f)
    c = torch.cat(c, dim=0)
    return c
