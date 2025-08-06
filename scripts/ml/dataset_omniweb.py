import torch
import os
import pandas as pd
import numpy as np

from dataset_pandasdataset import PandasDataset

# Consider the following scaling, not currently implemented:
# omniweb__ae_index__[nT] clamp(0, inf) -> log1p -> z-score
# omniweb__al_index__[nT] -= 11 -> clamp(-inf, 0) -> neg -> log1p -> z-score (or maybe discard)
# omniweb__au_index__[nT] += 9 -> clamp(0, inf) -> log1p -> z-score (or maybe discard)
# omniweb__sym_d__[nT] z-score / yeo 
# omniweb__sym_h__[nT] z-score x -> sign(x) * log1p(abs(x))
# omniweb__asy_d__[nT] z-score
# omniweb__bx_gse__[nT] z-score OK
# omniweb__by_gse__[nT] z-score OK
# omniweb__bz_gse__[nT] z-score OK
# omniweb__speed__[km/s] z-score
# omniweb__vx_velocity__[km/s] z-score
# omniweb__vy_velocity__[km/s] z-score
# omniweb__vz_velocity__[km/s] z-score

# mean and std for omni indices after yeo-johnson transformation
# omniweb__ae_index__[nT]: mean=3.382117810171, std=0.599638789174
# omniweb__al_index__[nT]: mean=-10.758787729440, std=7.776873220187
# omniweb__au_index__[nT]: mean=21.202255971757, std=17.939704773432
# omniweb__sym_d__[nT]: mean=-0.181702064224, std=2.998393072357
# omniweb__sym_h__[nT]: mean=-6.992693259217, std=12.396448647053
# omniweb__asy_d__[nT]: mean=2.628395773277, std=0.500936940952
# omniweb__speed__[km/s]: mean=0.962903675585, std=0.000397192623
# omniweb__vx_velocity__[km/s]: mean=-0.959153291778, std=0.000387756752
# omniweb__vy_velocity__[km/s]: mean=-1.591600332192, std=24.061675738624
# omniweb__vz_velocity__[km/s]: mean=-3.172566847907, std=21.392025871686
# omniweb__bx_gse__[nT]: mean=0.009877579877, std=3.477797581682
# omniweb__by_gse__[nT]: mean=-0.086797729712, std=4.004904143231
# omniweb__bz_gse__[nT]: mean=0.009738823405, std=3.251240498296

omniweb_all_columns = [
    'omniweb__ae_index__[nT]',
    'omniweb__al_index__[nT]',
    'omniweb__au_index__[nT]',
    'omniweb__sym_d__[nT]',
    'omniweb__sym_h__[nT]',
    'omniweb__asy_d__[nT]',
    'omniweb__bx_gse__[nT]',
    'omniweb__by_gse__[nT]',
    'omniweb__bz_gse__[nT]',
    'omniweb__speed__[km/s]',
    'omniweb__vx_velocity__[km/s]',
    'omniweb__vy_velocity__[km/s]',
    'omniweb__vz_velocity__[km/s]'
]

# lambda values for yeo-johnson transformation computed by Giacomo Acciarini, 5 Aug 2025
omniweb_yeojohnson_lambdas = {
    'omniweb__ae_index__[nT]': -0.127596533271,
    'omniweb__al_index__[nT]': 1.593553014366,
    'omniweb__au_index__[nT]': 0.694548772242,
    'omniweb__sym_d__[nT]': 1.023942223073,
    'omniweb__sym_h__[nT]': 1.150157823027,
    'omniweb__asy_d__[nT]': -0.044273161179,
    'omniweb__bx_gse__[nT]': 0.995147537010,
    'omniweb__by_gse__[nT]': 0.990598907069,
    'omniweb__bz_gse__[nT]': 1.007983055426,
    'omniweb__speed__[km/s]': -1.036437447980,
    'omniweb__vx_velocity__[km/s]': 3.040535004580,
    'omniweb__vy_velocity__[km/s]': 0.965800969635,
    'omniweb__vz_velocity__[km/s]': 0.998312073289
}

omniweb_mean_after_yeojohnson = {'omniweb__ae_index__[nT]': 3.382117810171,
                       'omniweb__al_index__[nT]': -10.758787729440,
                       'omniweb__au_index__[nT]': 21.202255971757,
                       'omniweb__sym_d__[nT]': -0.181702064224,
                       'omniweb__sym_h__[nT]': -6.992693259217,
                       'omniweb__asy_d__[nT]': 2.628395773277,
                       'omniweb__bx_gse__[nT]': 0.009877579877,
                       'omniweb__by_gse__[nT]': -0.086797729712,
                       'omniweb__bz_gse__[nT]': 0.009738823405,
                       'omniweb__speed__[km/s]': 0.962903675585,
                       'omniweb__vx_velocity__[km/s]': -0.959153291778,
                       'omniweb__vy_velocity__[km/s]': -1.591600332192,
                       'omniweb__vz_velocity__[km/s]': -3.172566847907}
omniweb_std_after_yeojohnson = {'omniweb__ae_index__[nT]': 0.599638789174,
                       'omniweb__al_index__[nT]': 7.776873220187,
                       'omniweb__au_index__[nT]': 17.939704773432,
                       'omniweb__sym_d__[nT]': 2.998393072357,
                       'omniweb__sym_h__[nT]': 12.396448647053,
                       'omniweb__asy_d__[nT]': 0.500936940952,
                       'omniweb__bx_gse__[nT]': 3.477797581682,
                       'omniweb__by_gse__[nT]': 4.004904143231,
                       'omniweb__bz_gse__[nT]': 3.251240498296,
                       'omniweb__speed__[km/s]': 0.000397192623,
                       'omniweb__vx_velocity__[km/s]': 0.000387756752,
                       'omniweb__vy_velocity__[km/s]': 24.061675738624,
                       'omniweb__vz_velocity__[km/s]': 21.392025871686}

omniweb_all_columns_yeojohnson_lambdas = torch.tensor([omniweb_yeojohnson_lambdas[col] for col in omniweb_all_columns], dtype=torch.float32)
omniweb_all_columns_mean_of_yeojohnson = torch.tensor([omniweb_mean_after_yeojohnson[col] for col in omniweb_yeojohnson_lambdas.keys()], dtype=torch.float32)
omniweb_all_columns_std_of_yeojohnson = torch.tensor([omniweb_std_after_yeojohnson[col] for col in omniweb_yeojohnson_lambdas.keys()], dtype=torch.float32)

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


# ionosphere-data/omniweb_karman_2025
class OMNIWeb(PandasDataset):
    def __init__(self, data_dir, date_start=None, date_end=None, normalize=True, rewind_minutes=50, date_exclusions=None, column=omniweb_all_columns, delta_minutes=15): # 50 minutes rewind defualt
        file_name_indices = os.path.join(data_dir, 'omniweb_indices_15min.csv')
        file_name_magnetic_field = os.path.join(data_dir, 'omniweb_magnetic_field_15min.csv')
        file_name_solar_wind = os.path.join(data_dir, 'omniweb_solar_wind_15min.csv')

        print('\nOMNIWeb')
        print('File indices           : {}'.format(file_name_indices))
        print('File magnetic field    : {}'.format(file_name_magnetic_field))
        print('File solar wind        : {}'.format(file_name_solar_wind))
        # delta_minutes = 1
        # delta_minutes = 15 # unclear what this is immediately, i think its supposed to match cadence but something to check out tmo

        data_indices = pd.read_csv(file_name_indices)
        data_magnetic_field = pd.read_csv(file_name_magnetic_field)
        data_solar_wind = pd.read_csv(file_name_solar_wind)

        #concat all columns to a single dataframe
        data = pd.concat([data_indices, data_magnetic_field, data_solar_wind], axis=1)

        # Remove duplicate columns (keep first occurrence)
        data = data.loc[:, ~data.columns.duplicated()]

        print('Data shape             : {}'.format(data.shape))
        print('Data columns           : {}'.format(data.columns.tolist()))
        
        # rename all__dates_datetime__ to Datetime
        data.rename(columns={'all__dates_datetime__': 'Datetime'}, inplace=True)
        # convert Datetime to datetime
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.copy()

        self.column = column

        super().__init__('OMNIWeb', data, self.column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    def normalize_data(self, data): 
        if self.column == omniweb_all_columns:
            data = yeojohnson(data, omniweb_all_columns_yeojohnson_lambdas)
            data = (data - omniweb_all_columns_mean_of_yeojohnson) / omniweb_all_columns_std_of_yeojohnson
        else:
            lambdas = torch.tensor([omniweb_yeojohnson_lambdas[col] for col in self.column], dtype=torch.float32)
            means = torch.tensor([omniweb_mean_after_yeojohnson[col] for col in self.column], dtype=torch.float32)
            stds = torch.tensor([omniweb_std_after_yeojohnson[col] for col in self.column], dtype=torch.float32)
            data = yeojohnson(data, lambdas)
            data = (data - means) / stds
        return data

    def unnormalize_data(self, data):
        if self.column == omniweb_all_columns:
            data = data * omniweb_all_columns_std_of_yeojohnson + omniweb_all_columns_mean_of_yeojohnson
            data = yeojhonson_inverse(data, omniweb_all_columns_yeojohnson_lambdas)
        else:
            lambdas = torch.tensor([omniweb_yeojohnson_lambdas[col] for col in self.column], dtype=torch.float32)
            means = torch.tensor([omniweb_mean_after_yeojohnson[col] for col in self.column], dtype=torch.float32)
            stds = torch.tensor([omniweb_std_after_yeojohnson[col] for col in self.column], dtype=torch.float32)
            data = data * stds + means
            data = yeojhonson_inverse(data, lambdas)
        return data