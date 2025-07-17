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
        # each file is actuall a csv.gzip file

    Celestrak
        DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,
        AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OB

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


Celestrak
    
'''

import torch
import os
import numpy as np
import datetime

class OMNIDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        omni_dir,
        date_start=None, 
        date_end=None, 
        normalize=True,
        transform=None
        ):
        super().__init__()

        print('OMNI Dataset')
        self.data_dir = data_dir
        self.normalize = normalize
        dates_avialable = self.find_date_range(data_dir)
        if dates_avialable is None:
            raise ValueError("No data found in the specified directory.")
        date_start_on_disk, date_end_on_disk = dates_avialable

        self.date_start = date_start_on_disk if date_start is None else date_start
        self.date_end = date_end_on_disk if date_end is None else date_end

        if self.date_start > self.date_end:
            raise ValueError("Start date cannot be after end date.")
        if self.date_start < date_start_on_disk or self.date_end > date_end_on_disk:
            raise ValueError("Specified date range is outside the available data range.")

        self.num_days = (self.date_end - self.date_start).days + 1
        self.cadence = 15 # minutes (actual cadence is 5 minutes but sample at 15 minutes for consistency)
        self.num_samples = int(self.num_days * (24 * 60 / self.cadence))

        print('Number of days in dataset   : {:,}'.format(self.num_days))
        print('Number of samples in dataset: {:,}'.format(self.num_samples))

        # size on disk
        size_on_disk = sum(os.path.getsize(f) for f in glob(f"{data_dir}/*.csv"))
        print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))
    
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
    
    @staticmethod
    def find_date_range(directory):
        # print("Checking date range of data in directory: {}".format(directory))
        files = sorted(glob(f"{directory}/*.csv"))
        if len(days) == 0:
            return None

        years = [y.replace(directory, '') for y in years]
        date_start_str = pd.csv_open(files[0], compression="gzip")["Datetime"].iloc[0] # '2006-01-01 00:00:00' example
        date_end_str = pd.csv_open(files[-1], compression="gzip")["Datetime"].iloc[-1] # '2006-01-01 00:00:00' example
        date_start = datetime.strptime(date_start_str, "%Y-%m-%d %H:%M:%S")
        date_end = datetime.strptime(date_end_str, "%Y-%m-%d %H:%M:%S")

        print("Directory  : {}".format(directory))
        print("Start date : {}".format(date_start.strftime('%Y-%m-%d')))
        print("End date   : {}".format(date_end.strftime('%Y-%m-%d')))

        return date_start, date_end




class JPLDGIMDataset(Dataset):
    def __init__(self, data_dir, date_start=None, date_end=None, normalize=True):
        print('JPLD GIM Dataset')
        self.data_dir = data_dir
        self.normalize = normalize
        dates_avialable = self.find_date_range(data_dir)
        if dates_avialable is None:
            raise ValueError("No data found in the specified directory.")
        date_start_on_disk, date_end_on_disk = dates_avialable

        self.date_start = date_start_on_disk if date_start is None else date_start
        self.date_end = date_end_on_disk if date_end is None else date_end
        if self.date_start > self.date_end:
            raise ValueError("Start date cannot be after end date.")
        if self.date_start < date_start_on_disk or self.date_end > date_end_on_disk:
            raise ValueError("Specified date range is outside the available data range.")

        self.num_days = (self.date_end - self.date_start).days + 1
        self.cadence = 15 # minutes
        self.num_samples = int(self.num_days * (24 * 60 / cadence))
        print('Number of days in dataset   : {:,}'.format(self.num_days))
        print('Number of samples in dataset: {:,}'.format(self.num_samples))
        # size on disk
        size_on_disk = sum(os.path.getsize(f) for f in glob(f"{data_dir}/*/*.nc.gz"))
        print('Size on disk                : {:.2f} GB'.format(size_on_disk / (1024 ** 3)))

    @staticmethod
    def normalize(data):
        return (data - JPLDGIM_mean) / JPLDGIM_std
        # return torch.log1p(data)

    @staticmethod
    def unnormalize(data):
        return data * JPLDGIM_std + JPLDGIM_mean

    def __len__(self):
        return self.num_samples
    
    @lru_cache(maxsize=1024) # number of days to cache in memory, roughly 3 MiB per day
    def _get_day_data(self, date):
        file_name = f"jpld{date:%j}0.{date:%y}i.nc.gz"
        file_path = os.path.join(self.data_dir, f"{date:%Y}", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with gzip.open(file_path, 'rb') as f:
            ds = xr.open_dataset(f, engine='h5netcdf')
            
            # Assuming 'tecmap' is the variable of interest
            data = ds['tecmap'].values
            # data_tensor shape torch.Size([96, 180, 360]) where 96 is nepochs, 180 is nlats, and 360 is nlons
            data_tensor = torch.tensor(data, dtype=torch.float32)
            if self.normalize:
                data_tensor = JPLDGIMDataset.normalize(data_tensor)

            return data_tensor

    def __getitem__(self, index):
        samples_per_day = 24 * 60 // self.cadence  
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, int):
            if index < 0 or index >= self.num_samples:
                raise IndexError("Index out of range for the dataset.")
            days = index // samples_per_day
            minutes = (index % samples_per_day) * 15
            date = self.date_start + datetime.timedelta(days=days, minutes=minutes)
        else:
            raise TypeError("Index must be an integer or a datetime object.")

        data = self._get_day_data(date)
        time_index = (index % samples_per_day)  # Get the time index within the
        data = data[time_index, :, :]  # Select the specific time slice

        return data
