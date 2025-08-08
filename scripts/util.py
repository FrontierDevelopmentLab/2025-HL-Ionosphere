import sys
import traceback
import random
import time
import numpy as np
import torch
import hashlib


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


# Yeo-Johnson transformation
# Based on https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7eacfaff061cf68e09bcd48aa93d4d6b/sklearn/preprocessing/_data.py#L3480
def yeojohnson(X, lambdas):
    if X.shape != lambdas.shape:
        raise ValueError("X and lambdas must have the same shape.")
    if X.ndim != 1:
        raise ValueError("X must be a 1D tensor.")
    
    # Ensure that no lambdas are 0 or 2 to avoid division by zero
    if torch.isclose(lambdas, torch.zeros_like(lambdas), atol=1e-8).any() or torch.isclose(lambdas, torch.tensor(2.0, dtype=lambdas.dtype, device=lambdas.device), atol=1e-8).any():
        raise ValueError("Lambdas must not contain 0 or 2 to avoid division by zero.")

    out = torch.zeros_like(X)
    pos = X >= 0  # binary mask

    # CAUTION: this assumes a lambda will never be 0 or 2
    out[pos] = (torch.pow(X[pos] + 1, lambdas[pos]) - 1) / lambdas[pos]
    out[~pos] = -(torch.pow(-X[~pos] + 1, 2 - lambdas[~pos]) - 1) / (2 - lambdas[~pos])
    return out


# Yeo-Johnson inverse transformation
# Based on https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7eacfaff061cf68e09bcd48aa93d4d6b/sklearn/preprocessing/_data.py#L3424C1-L3431C41
def yeojhonson_inverse(X, lambdas):
    if X.shape != lambdas.shape:
        raise ValueError("X and lambdas must have the same shape.")
    if X.ndim != 1:
        raise ValueError("X must be a 1D tensor.")
    X_original = torch.zeros_like(X)
    pos = X >= 0

    # Ensure that no lambdas are 0 or 2 to avoid division by zero
    if torch.isclose(lambdas, torch.zeros_like(lambdas), atol=1e-8).any() or torch.isclose(lambdas, torch.tensor(2.0, dtype=lambdas.dtype, device=lambdas.device), atol=1e-8).any():
        raise ValueError("Lambdas must not contain 0 or 2 to avoid division by zero.")


    # CAUTION: this assumes a lambda will never be 0 or 2
    X_original[pos] = (X[pos] * lambdas[pos] + 1) ** (1 / lambdas[pos]) - 1
    X_original[~pos] = 1 - (-(2 - lambdas[~pos]) * X[~pos] + 1) ** (1 / (2 - lambdas[~pos]))

    return X_original


def md5_hash_str(input_str):
    m = hashlib.md5()
    m.update(input_str.encode('utf-8'))
    hash_str = m.hexdigest()
    return hash_str


def format_bytes(num_bytes):
    """Format a number of bytes as a human-readable string (MiB, GiB, TiB, etc)."""
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0