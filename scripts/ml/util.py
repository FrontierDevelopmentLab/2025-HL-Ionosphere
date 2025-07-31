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


# Takes a list of scalars and returns stack of channels with shape (num_scalars, height, width) 
# where each scalar is expanded to a 2D image of the specified size.
def channelize(scalars, image_size=(180, 360)):
    if not torch.is_tensor(scalars):
        scalars = torch.tensor(scalars)
    c = scalars.view(-1, 1, 1)
    c = c.expand((-1,) + image_size)
    return c

