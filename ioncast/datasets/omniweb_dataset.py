'''
Solar indices pytorch dataset

Reads in data from the OMNI and Celestrak datasets.

Full dataset information:
    OMNIWeb
        # 'Year', 'Day', 'Hour', 'Minute',
        # 'IMF_ID', 'SW_ID', 'Npts_IMF', 'Npts_SW', 'Pct_interp',
        # 'Time_shift', 'RMS_shift', 'RMS_PFN', 'DBOT1',
        # 'B_mag', 'Bx_GSE', 'By_GSE', 'Bz_GSE', 'By_GSM', 'Bz_GSM',
        # 'RMS_B_scalar', 'RMS_B_vector',
        # 'V_flow', 'Vx', 'Vy', 'Vz',
        # 'Density', 'Temp', 'P_dyn', 'E_field', 'Beta', 'Mach_Alfven',
        # 'X_GSE', 'Y_GSE', 'Z_GSE',
        # 'BSN_X', 'BSN_Y', 'BSN_Z',
        # 'AE', 'AL', 'AU', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',
        # 'PC_N', 'Mach_sonic',
        # 'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV
        # NaN values fill data timestamp where data is not present


We only use the following columns (for initial research):
OMNIWeb
    'Year', 'Day', 'Hour', 'Minute',

    # Solar
    'B_mag', 'Bx_GSE', 'By_GSM', 'Bz_GSM', #IMF interplanetary magnetic field
    'RMS_B_scalar', 'RMS_B_vector',

    # Solar
    'V_flow', 'Vx', 'Vy', 'Vz',
    'Density', 'Temp', 

    # Solar derivations (tbd on including)
    #'P_dyn', 'E_field', 'Beta', 'Mach_Alfven',

    # Geomagnetic
    'AE', 'AL', 'AU', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',

    # Solar particle flux
    'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV'
    
'''
# TODO: make TimeSeriesDataset / CSVDataset base class
# TODO: should be made more robust to certain conditions (what if omni_rows empty) Also what if other parameters have major nan issues like the goes rows, this is more of a preprocessing problem
# TODO: not great to load the csvs each time, 
# 

from .base_datasets import PandasDataset
import torch
import os
import pandas as pd



# file_dir = "/mnt/ionosphere-data/omniweb/cleaned/"
class OMNIDataset(PandasDataset):
    def __init__(self, file_dir, date_start=None, date_end=None, normalize=True, rewind_minutes=50, date_exclusions=None, column=None, delta_minutes=15): # 50 minutes rewind defualt
        file_name = os.path.join(file_dir, "omni_5min_full_cleaned.csv")
        stats_dir = os.path.join(file_dir, "omniweb_stats.csv")

        print('\nOMNIWeb dataset')
        print('File                 : {}'.format(file_name))
        # delta_minutes = 1
        # delta_minutes = 15 # unclear what this is immediately, i think its supposed to match cadence but something to check out tmo


        self.stats_df = pd.read_csv(stats_dir)

        data = pd.read_csv(file_name)
        # print("1", data.head())
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        # print("2", data.head())
        # data = data.sort_values(by='Datetime')
        print('Rows                 : {:,}'.format(len(data)))

        if column is None:
            self.column = [
                # Solar
                'B_mag', 'Bx_GSE', 'By_GSM', 'Bz_GSM', #IMF interplanetary magnetic field
                'RMS_B_scalar', 'RMS_B_vector',
                # Solar
                'V_flow', 'Vx', 'Vy', 'Vz',
                'Density', 'Temp', 

                # Solar derivations (tbd on including) # NOTE: what was the consensus on  including these by default?
                'P_dyn', 'E_field', 'Beta', 'Mach_Alfven', 

                # Geomagnetic
                # 'AE', 'AL', 'AU', #  (Missing in 2019-2020)
                'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H',

                # # Solar particle flux # Unusable in Omni dataset
                # 'GOES_flux_10MeV', 'GOES_flux_30MeV', 'GOES_flux_60MeV' # (missing 2020 onwards)
            ]
        else:
            self.column = column
        # Remove outliers based on quantiles, # NOTE: is this something we care about? we have already removed the 999... so missing data based outliers shouldnt be an issue unless if some were missed (there was one atleast missed but was for an unused column)
        # q_low = data[self.column].quantile(0.001)
        # q_hi  = data[self.column].quantile(0.999) # Quantile stuff broken
        # print("3", data.head())

        # data = data[(data[self.column] < q_hi) & (data[self.column] > q_low)] # Quantile stuff broken this leads to NaNs
        # print("4", data.head())
        data = data[['Datetime']+ self.column] # filter out unused data early (as certain cols lead to removal of clean data as unusd cols have nans which are removed with dropna)
        # print("5", data.head())
        super().__init__('OMNIWeb dataset', data, self.column, delta_minutes, date_start, date_end, normalize, rewind_minutes, date_exclusions)

    # NOTE: what is the reason these methods were kept as instance methods but also passing the data in as an argument in the RSTNRadio dataset (or the other dattasets as well)?
    def normalize_data(self, data): 
        stats = torch.tensor(self.stats_df[self.column].values, dtype=torch.float32)
        omni_means = stats[0]
        omni_stds = stats[1]
        return (data - omni_means) / omni_stds

    def unnormalize_data(self, data):
        stats = torch.tensor(self.stats_df[self.column].values, dtype=torch.float32)
        omni_means = stats[0]
        omni_stds = stats[1]
        return data * omni_stds + omni_means

    # def unnormalize_data(self, data):
    #     data = data * data * data
    #     mean_cuberoot_data = 2.264420986175537
    #     std_cuberoot_data = 0.9795352816581726
    #     data = data * std_cuberoot_data
    #     data = data + mean_cuberoot_data
    #     return data

