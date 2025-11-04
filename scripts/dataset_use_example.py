import torch
import datetime

from dataset_jpld import JPLD
from dataset_sequences import Sequences
from dataset_union import Union
from dataset_sunmoongeometry import SunMoonGeometry
from dataset_quasidipole import QuasiDipole
from dataset_celestrak import CelesTrak
from dataset_omniweb import OMNIWeb, omniweb_all_columns
from dataset_set import SET, set_all_columns
from dataloader_cached import CachedDataLoader
from events import EventCatalog, validation_events_1, validation_events_2, validation_events_3, validation_events_4, validation_events_5

# Utility function to stack features as channels for input to a model
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

'''
Example of setting up a dataset for training and validation using the JPLD and OMNIWeb datasets, excluding certain events for validation.
'''

# Initialize variables
event_catalog = EventCatalog(events_csv_file_name='events.csv')
valid_event_id = ["G0H3-201804202100"]
image_size = (180, 360)
context_window = 2 # Number of time steps of context used in model
prediction_window = 1 # Number of time steps to predict
training_sequence_length = context_window + prediction_window
delta_minutes = 15 # 15-minute cadence 
date_dilation = 16 # Use 16x dilation for 15-min data to get 4-hour context

date_exclusions = []
datasets_omniweb_valid = []
datasets_jpld_valid = []
datasets_sunmoon_valid = []

dataset_omniweb_dir = '/path/to/omniweb/data/'
dataset_jpld_dir = '/path/to/jpld/data/'

# Process validation events
for event_id in valid_event_id:
    print('Excluding event ID: {}'.format(event_id))

    if event_id not in event_catalog:
        raise ValueError('Event ID {} not found in EventCatalog'.format(event_id))
    # EventCatalog[event_id] is a dict with keys:
    # 'date_start': date_start,
    # 'date_end': date_end,
    # 'duration': duration,
    # 'max_kp': max_kp,
    # 'time_steps': time_steps
    event = event_catalog[event_id]
    exclusion_start = datetime.datetime.fromisoformat(event['date_start']) - datetime.timedelta(minutes=context_window * delta_minutes)
    exclusion_end = datetime.datetime.fromisoformat(event['date_end'])
    date_exclusions.append((exclusion_start, exclusion_end))
    print('Exclusion start: {}, end: {}'.format(exclusion_start, exclusion_end))

    datasets_omniweb_valid.append(OMNIWeb(dataset_omniweb_dir, date_start=exclusion_start, date_end=exclusion_end, return_as_image_size=image_size))
    datasets_jpld_valid.append(JPLD(dataset_jpld_dir, date_start=exclusion_start, date_end=exclusion_end))
    datasets_sunmoon_valid.append(SunMoonGeometry(date_start=exclusion_start, date_end=exclusion_end))

# Create validation dataset
dataset_jpld_valid = Union(datasets=datasets_jpld_valid)
dataset_omniweb_valid = Union(datasets=datasets_omniweb_valid)

# Create training dataset excluding validation events
date_start = datetime.datetime(2015, 1, 1)
date_end = datetime.datetime(2020, 12, 31)
dataset_jpld_train = JPLD(dataset_jpld_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions)
dataset_omniweb_train = OMNIWeb(dataset_omniweb_dir, date_start=date_start, date_end=date_end, date_exclusions=date_exclusions, return_as_image_size=image_size)

# Create Sequence datasets
dataset_valid = Sequences(datasets=[dataset_jpld_valid, dataset_omniweb_valid], sequence_length=training_sequence_length, dilation=date_dilation, delta_minutes=delta_minutes)
dataset_train = Sequences(datasets=[dataset_jpld_train, dataset_omniweb_train], sequence_length=training_sequence_length, dilation=date_dilation, delta_minutes=delta_minutes)

