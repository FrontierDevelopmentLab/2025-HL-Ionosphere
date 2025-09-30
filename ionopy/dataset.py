import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

from tqdm import tqdm

TEC_MEAN_LOG1P = 2.34
TEC_STD_LOG1P = 0.82

DTEC_MEDIAN = 0.9271219968795776

class MadrigalDatasetTimeSeries(Dataset):
    def __init__(self, 
                config, 
                torch_type=torch.float32,
                min_date=pd.to_datetime("2010-06-13 00:00:00"),
                max_date=pd.to_datetime("2024-07-31 23:45:00"),
                features_to_exclude_timed=None
                ):
        """
        Initializes the MadrigalDatasetTimeSeries dataset.
        Parameters:
        -----------
            - config (`dict`): Configuration dictionary containing paths and parameters for the dataset.
            - torch_type (`torch.dtype`): The data type to use for the tensors. Default 
                                            is `torch.float32`.
            - min_date (`pd.Timestamp`): The minimum date for filtering the dataset. Default is "2010-06-13 00:00:00".
            - max_date (`pd.Timestamp`): The maximum date for filtering the dataset. Default is
                "2024-07-31 23:45:00".
        """
        self.min_date = min_date
        self.max_date = max_date
        print("\nMadrigalDatasetTimeSeries initialized with min_date:", self.min_date, "and max_date:", self.max_date)
        self.time_series_data = {}
        self.config = config
        self.torch_type = torch_type
        self.features = []
        # self.min_date = min_date
        # self.max_date = max_date

        print("Loading Madrigal dataset with config:\n", config)
        # Ensure all dataframes are sorted for merge_asof
        self.data=pd.read_csv(config['madrigal_path'])
        print("Madrigal data loaded with shape:", self.data.shape)
        self.data = self.data[(pd.to_datetime(self.data['all__dates_datetime__'])>=self.min_date) & (pd.to_datetime(self.data['all__dates_datetime__'])<=self.max_date)]
        #let's discard TEC values below 0.1:
        print("Discarding Madrigal TEC values < 0.1")
        mask = self.data['madrigal__tec__[TECU]'].values > 0.1
        self.data = self.data[mask]
        self.data.reset_index(drop=True, inplace=True)
        dates=pd.DatetimeIndex(self.data['all__dates_datetime__'].values)
        self.data['all__dates_datetime__']=dates
        self.data = self.data.sort_values('all__dates_datetime__').reset_index(drop=True)
        print("Now removing 0 TEC values...")
        self.data=self.data[self.data['madrigal__tec__[TECU]'].values>0]
        self.data.reset_index(drop=True, inplace=True)
        print("Madrigal data processed, final shape:", self.data.shape)

        # Final sort and cleanup
        self.data.sort_values('all__dates_datetime__', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        ##### Normalization and Features Extraction #####
        # Extract date and features
        self.input_features=[]
        self.input_features_names = []
        dt=self.data['all__dates_datetime__'].dt
        #day of the year:
        self.doy=torch.tensor(dt.dayofyear,dtype=self.torch_type)
        feature_as_radian = (2* np.pi * (self.doy-1.)/ (366. - 1.))
        self.input_features.append(feature_as_radian.sin())
        self.input_features.append(feature_as_radian.cos())
        self.input_features_names += ['doy_sin', 'doy_cos']
        #seconds in day:
        self.sid = torch.tensor(dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6,dtype=self.torch_type)
        feature_as_radian = (2.* np.pi * (self.sid-0.)/ (86400. - 0.))
        self.input_features.append(feature_as_radian.sin())
        self.input_features.append(feature_as_radian.cos())
        self.input_features_names += ['sid_sin', 'sid_cos']
        #now the latitude:
        self.latitudes = torch.tensor(np.deg2rad(self.data['all__latitudes__[deg]'].values), dtype=self.torch_type)
        #let's create indices that highlight -> low latitude zone (<30 deg), mid-latitude zone (30-60 deg), and high-lat zone (>60 deg)
        latitude_bins_classification={0:'Low latitude (<30 deg)', 
                                    1:'Mid latitude (30-60 deg)', 
                                    2:'High latitude (>60 deg)'}
        latitude_bins = np.deg2rad(np.array([30, 60, 91]))
        latitude_classification = np.digitize(abs(self.latitudes.numpy()), latitude_bins)
        latitude_classification = [latitude_bins_classification[v] for v in latitude_classification]    
        self.latitude_classification = latitude_classification
        self.input_features.append( 2 * (self.latitudes - (-np.pi/2)) / (np.pi/2 - (-np.pi/2)) - 1)
        self.input_features_names += ['latitudes_normalized']
        #self.latitudes_normalized = 2 * (self.latitudes - (-np.pi/2)) / (np.pi/2 - (-np.pi/2)) - 1
        #now the longitude using sine/cosine:
        self.longitudes = torch.tensor(np.deg2rad(self.data['all__longitudes__[deg]'].values), dtype=self.torch_type)
        self.input_features.append(self.longitudes.sin())
        self.input_features.append(self.longitudes.cos())
        self.input_features_names += ['longitudes_sin', 'longitudes_cos']

        self.min_date = self.data['all__dates_datetime__'].min()
        self.max_date = self.data['all__dates_datetime__'].max()

        self.dates = self.data['all__dates_datetime__'].values
        self.dates_str=pd.to_datetime(self.dates).strftime('%Y-%m-%d %H:%M:%S.%f')

        self.input_features = torch.stack(self.input_features).T
        print("Input features shape:", self.input_features.shape)
        self.features_to_exclude_timed = features_to_exclude_timed
        tec = self.data['madrigal__tec__[TECU]'].values
        self.tec = torch.tensor((np.log1p(tec)-TEC_MEAN_LOG1P)/TEC_STD_LOG1P, dtype=self.torch_type)
        #now the std of the TEC:
        dtec=self.data['madrigal__dtec__[TECU]'].values
        #we replace to the NaN of the DTEC the median (e.g. if the std is not known, the median std over the whole thing is used)
        dtec[np.isnan(dtec)] = DTEC_MEDIAN #we replace NaN with the median
        #let's also replace the ones above 10 with 10.:
        dtec = np.clip(dtec, 0., 10.) #we clip the values to be between 0 and 10
        self.dtec = torch.tensor(np.log1p(dtec), dtype=self.torch_type)
        #let's now normalize the dtec min max from -1 to 1 assuming the min is 0 and max is 1.609
        self.dtec = 2*(self.dtec - 0.)/(np.log1p(10.)-0.)-1.
        print("Before normalization:")
        print("Min TEC:", self.data['madrigal__tec__[TECU]'].min(), "Max TEC:", self.data['madrigal__tec__[TECU]'].max())
        print("Min dTEC:", self.data['madrigal__dtec__[TECU]'].min(), "Max dTEC:", self.data['madrigal__dtec__[TECU]'].max())
        print("After normalization:")
        print("Min TEC:", self.tec.min().item(), "Max TEC:", self.tec.max().item())
        print("Min dTEC:", self.dtec.min().item(), "Max dTEC:", self.dtec.max().item())


        # Add time series data here.            
        if config["use_timed"] is True:
            print("\nLoading TIMED SEE Level 3 dataset.")
            if self.features_to_exclude_timed is not None:
                self.features_to_exclude_timed+=["all__dates_datetime__"]
            else:
                self.features_to_exclude_timed=["all__dates_datetime__"]
            self._add_time_series_data(
                "timed",
                config["timed_path"],
                config['lag_days_timed'],
                config['timed_resolution'],
                self.features_to_exclude_timed,
            )
        if config["use_jpld"] is True:
            print("\nLoading JPLD dataset.")
            self._add_time_series_data(
                "jpld",
                config["jpld_path"],
                config['lag_minutes_jpld'],
                config['jpld_resolution'],
                ["all__dates_datetime__"],
            )
        if config["use_omni_indices"] is True:
            print("\nLoading Omni indices.")
            self._add_time_series_data(
                "omni_indices",
                config["omni_indices_path"],
                config['lag_minutes_omni'],
                config['omni_resolution'],
                ["all__dates_datetime__", "source__gaps_flag__", "omniweb__ae_index__[nT]", "omniweb__al_index__[nT]", "omniweb__au_index__[nT]"],
            )
        if config['use_omni_solar_wind'] is True:
            print("\nLoading Omni Solar Wind.")
            self._add_time_series_data(
                "omni_solar_wind",
                config["omni_solar_wind_path"],
                config['lag_minutes_omni'],
                config['omni_resolution'],
                ["all__dates_datetime__", "source__gaps_flag__"],
            )
        if config['use_omni_magnetic_field'] is True:
            print("\nLoading Omni Magnetic Field.")
            self._add_time_series_data(
                "omni_magnetic_field",
                config["omni_magnetic_field_path"],
                config['lag_minutes_omni'],
                config['omni_resolution'],
                ["all__dates_datetime__", "source__gaps_flag__"],
            )

        if config["use_set_sw"] is True:
            print("\nLoading SET Solar Wind data.")
            self._add_time_series_data(
                "set_sw",
                config["set_sw_path"],
                config['lag_days_proxies'],
                config['proxies_resolution'],
                ["all__dates_datetime__"],
            )
        
        if config["use_celestrack"] is True:
            print("\nLoading Celestrack data.")
            self._add_time_series_data(
                "celestrack",
                config["celestrack_path"],
                config['lag_days_proxies'],
                config['proxies_resolution'],
                ["all__dates_datetime__"],
            )
        
    def _add_time_series_data(self, 
                              data_name, 
                              data_path, 
                              lag, 
                              resolution, 
                              excluded_features):
        # Data loading:
        self.time_series_data[data_name] = {}
        if data_name in ["omni_indices", "omni_solar_wind", "omni_magnetic_field", "jpld"]:
            print("Loading time series data for", data_name)
            self.time_series_data[data_name]["data"] = pd.read_csv(data_path)
            self.time_series_data[data_name]["data"]['all__dates_datetime__'] = pd.DatetimeIndex(self.time_series_data[data_name]["data"]['all__dates_datetime__'])
            # we now index the data by the datetime column, and sort it by the index. The reason is that it is then easier to resample

            # We exclude the columns that are not needed for the model.
            print("Normalizing time series data for", data_name)
            # This is to remove significant outliers, such as the fism2 flare data which has 10^45 photons at one point. Regardless
            # of whther this is true or not, it severely affects the distribution.
            # We replace NaNs and +/-inf by interpolating them away.
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].replace([np.inf, -np.inf], None)
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].dropna()
            self.time_series_data[data_name]["data"].index = pd.to_datetime(self.time_series_data[data_name]["data"]["all__dates_datetime__"])
            self.time_series_data[data_name]["data"].sort_index(inplace=True)
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].drop(columns=excluded_features, axis=1)

            self.time_series_data[data_name]["data"] = (self.time_series_data[data_name]["data"].resample(f"{resolution}min").ffill())
            # self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"][
            #                                                                                     (self.time_series_data[data_name]["data"].index >= (self.min_date - pd.DateOffset(minutes=lag))) &
            #                                                                                     (self.time_series_data[data_name]["data"].index <= self.max_date)
            #                                                                                 ]
            # for column in self.time_series_data[data_name]["data"].columns:
            #     quantile = self.time_series_data[data_name]["data"][column].quantile(0.998)
            #     more_than = self.time_series_data[data_name]["data"][column] >= quantile
            #     self.time_series_data[data_name]["data"].loc[more_than, column] = None

            self.time_series_data[data_name]["date_start"] = min(self.time_series_data[data_name]["data"].index)
            self.time_series_data[data_name]["column_names"] = self.time_series_data[data_name]["data"].columns
            #let's apply the Yeo-Johnson transformation:
            self.time_series_data[data_name]["features_names"] = [f'{column}_yeojohnson_zscore' for column in self.time_series_data[data_name]["data"].columns]

            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            vals = self.time_series_data[data_name]["data"].values
            vals = scaler.fit_transform(vals)
            vals = torch.tensor(vals, dtype=self.torch_type)
            self.time_series_data[data_name]["data"] = vals

            self.time_series_data[data_name]["lag"] = lag

            self.time_series_data[data_name]["scaler"] = scaler
            self.time_series_data[data_name]["resolution"] = resolution
        elif data_name.startswith("set"):
            #now SET F10.7, M10.7, S10.7, Y10.7:
            tmp=pd.read_csv(data_path)
            self.time_series_data[data_name]["data"] = tmp
            self.time_series_data[data_name]["data"]["all__dates_datetime__"]=pd.DatetimeIndex(tmp['all__dates_datetime__'])
            self.time_series_data[data_name]["data"]['all__dates_datetime__'] = pd.DatetimeIndex(self.time_series_data[data_name]["data"]['all__dates_datetime__'])

            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].dropna()
            self.time_series_data[data_name]["data"].index = pd.to_datetime(self.time_series_data[data_name]["data"]["all__dates_datetime__"])
            self.time_series_data[data_name]["data"].sort_index(inplace=True)

            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].drop(columns=excluded_features, axis=1)
            self.time_series_data[data_name]["data"] = (self.time_series_data[data_name]["data"].resample(f"{resolution}D").ffill())
            # self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"][
            #                                                                                     (self.time_series_data[data_name]["data"].index >= (
            #                                                                                         self.min_date - pd.DateOffset(days=lag)
            #                                                                                     )) &
            #                                                                                     (self.time_series_data[data_name]["data"].index <= self.max_date)
            #                                                                                 ]
            f107_obs=self.time_series_data[data_name]["data"]['space_environment_technologies__f107_obs__'].values
            print("Normalizing time series data for", data_name)
            vals=[]
            for column in self.time_series_data[data_name]["data"].columns:
                vals.append(torch.tensor((np.log1p(self.time_series_data[data_name]["data"][column].values)-4.71)/0.4, dtype=self.torch_type))
            #let's apply the Yeo-Johnson transformation:
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            self.time_series_data[data_name]["column_names"] = self.time_series_data[data_name]["data"].columns
            #self.input_features_names += [f'{column}_log1p_normalized']

            self.time_series_data[data_name]["date_start"] = min(self.time_series_data[data_name]["data"].index)
            self.time_series_data[data_name]["column_names"] = self.time_series_data[data_name]["data"].columns
            #self.time_series_data[data_name]["data_matrix"] = self.time_series_data[data_name]["data"].values
            self.time_series_data[data_name]["features_names"] = [f'{column}_yeo' for column in self.time_series_data[data_name]["data"].columns]

            vals = self.time_series_data[data_name]["data"].values
            vals = scaler.fit_transform(vals)
            vals = torch.tensor(vals, dtype=self.torch_type)

            self.time_series_data[data_name]["data"]=vals
            self.time_series_data[data_name]["lag"] = lag
            self.time_series_data[data_name]["resolution"] = resolution
            self.time_series_data[data_name]["scaler"] = scaler

            #classify solar activity levels according to F10.7 values:
            solar_activity_bins_classification={0:'F10.7: 0-70 (low)',
                                                1:'F10.7: 70-150 (moderate)',
                                                2:'F10.7: 150-200 (moderate-high)',
                                                3:'F10.7: 200 (high)'}
            solar_activity_bins = np.array([0, 70, 150, 200, 1000])
            solar_activity_classification = np.digitize(f107_obs, solar_activity_bins)-1
            solar_activity_classification = [solar_activity_bins_classification[v] for v in solar_activity_classification]
            self.time_series_data[data_name]["solar_activity_classification"] = solar_activity_classification

        elif data_name in ["celestrack", "timed"]:
            #ap average
            tmp=pd.read_csv(data_path)
            self.time_series_data[data_name]["data"] = tmp
            self.time_series_data[data_name]["data"]["all__dates_datetime__"]=pd.DatetimeIndex(tmp['all__dates_datetime__'])
            
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].dropna()
            self.time_series_data[data_name]["data"].index = pd.to_datetime(self.time_series_data[data_name]["data"]["all__dates_datetime__"])
            self.time_series_data[data_name]["data"].sort_index(inplace=True)
            self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].drop(columns=excluded_features, axis=1)

            self.time_series_data[data_name]["data"] = (self.time_series_data[data_name]["data"].resample(f"{resolution}D").ffill())
            #let's remove the NaN:
            #self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"].replace([np.inf, -np.inf], None)
            # self.time_series_data[data_name]["data"] = self.time_series_data[data_name]["data"][
            #                                                                                     (self.time_series_data[data_name]["data"].index >= (
            #                                                                                         self.min_date - pd.DateOffset(days=lag)
            #                                                                                     )) &
            #                                                                                     (self.time_series_data[data_name]["data"].index <= self.max_date)
            #                                                                                 ]

            #let's now 

            print("Normalizing time series data for", data_name)
            # vals=[]
            # for column in self.time_series_data[data_name]["data"].columns:
            #     vals.append(torch.tensor((np.log1p(self.time_series_data[data_name]["data"][column].values)-2.03)/0.7, dtype=self.torch_type))
            #let's apply the Yeo-Johnson transformation:
            self.time_series_data[data_name]["date_start"] = min(self.time_series_data[data_name]["data"].index)
            self.time_series_data[data_name]["column_names"] = self.time_series_data[data_name]["data"].columns
            self.time_series_data[data_name]["features_names"] = [f'{column}_yeojohnson_zscore' for column in self.time_series_data[data_name]["data"].columns]
            if data_name.startswith("celestrack"):
                self.ap_average=self.time_series_data[data_name]["data"]['celestrack__ap_average__'].values

            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            vals = self.time_series_data[data_name]["data"].values
            vals = scaler.fit_transform(vals)
            vals = torch.tensor(vals, dtype=self.torch_type)

            self.time_series_data[data_name]["scaler"] = scaler
            self.time_series_data[data_name]["data"]=vals
            #self.input_features_names += [f'{column}_log1p_normalized']
            self.time_series_data[data_name]["lag"] = lag
            self.time_series_data[data_name]["resolution"] = resolution

            #self.time_series_data[data_name]["data_matrix"] = self.time_series_data[data_name]["data"].values

            if data_name.startswith("celestrack"):
                ap_values_bins = np.array([0, 39, 67, 111, 179, 300, 4001])
                ap_values_classification={0:'G0', 
                                            1:'G1', 
                                            2:'G2', 
                                            3:'G3', 
                                            4:'G4', 
                                            5:'G5'}
                storm_classification = np.digitize(self.ap_average, ap_values_bins)-1
                storm_classification = [ap_values_classification[v] for v in storm_classification]
                ap_average = torch.tensor(self.ap_average, dtype=self.torch_type)
                self.time_series_data[data_name]["storm_classification"] = storm_classification
                self.time_series_data[data_name]["ap_average"] = ap_average
    
    def _set_indices(self, test_month_idx, validation_month_idx, custom=None):
        """
        Works out which indices are in the training, validation and test sets.
        Previously, a file with indices was stored in the cloud. However, this way
        simply works it out based on dates, which I feel is much safer. It takes
        only a couple of minutes.

        Parameters:
        ------------
            - test_month_idx (`list`): The indices of the months to use for the test set.
            - validation_month_idx (`list`): The indices of the months to use for the validation set.
            - custom (`dict`): The dictionary with the custom validation and test months to use for each year (example: {2020: {"validation": 2, "test":0}, 2024: {...}}).
        """
        years = list(range(self.min_date.year, self.max_date.year + 1))
        months = np.array(range(1, 13))

        # Take the remaining indices (0-11 inclusive) as train months.
        train_month_idx = sorted(
            list(
                set(list(range(0, 12)))
                - set(validation_month_idx)
                - set(test_month_idx)
            )
        )

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        date_column = self.data["all__dates_datetime__"]
        print("Creating training, validation and test sets.")
        for idx_year, year in enumerate(tqdm(years, desc=f"{len(years)} years to iterate through.")):
            #custom events:
            if custom is not None and year in custom:
                year_months_val = [custom[year]['validation']]
                year_months_test = [custom[year]['test']]
                year_months_train=sorted(list(set(list(range(0, 12)))- set(validation_month_idx)- set(test_month_idx)))
            else:  
                year_months_train = list(np.roll(months, idx_year)[train_month_idx])
                year_months_val = list(np.roll(months, idx_year)[validation_month_idx])
                year_months_test = list(np.roll(months, idx_year)[test_month_idx])
            correct_year = date_column.dt.year.astype(int) == int(year)
            date_as_month = date_column.dt.month
            self.train_indices += list(
                self.data[
                    (correct_year & date_as_month.isin(year_months_train))
                ].index
            )
            self.val_indices += list(
                self.data[
                    (correct_year & date_as_month.isin(year_months_val))
                ].index
            )
            self.test_indices += list(
                self.data[
                    (correct_year & date_as_month.isin(year_months_test))
                ].index
            )
        self.train_indices = sorted(self.train_indices)
        self.val_indices = sorted(self.val_indices)
        self.test_indices = sorted(self.test_indices)

        print("Train size:", len(self.train_indices))
        print("Validation size:", len(self.val_indices))
        print("Test size:", len(self.test_indices))

    def date_to_index(self, date, date_start, resolution_seconds):
        delta = date - date_start
        index = int(delta.total_seconds() / resolution_seconds)
        return index
    def __len__(self):
        return len(self.dates_str)

    # These three methods return a subset of the dataset based on the calculated indices.
    def train_dataset(self):
        return Subset(self, self.train_indices)

    def validation_dataset(self):
        return Subset(self, self.val_indices)

    def test_dataset(self):
        return Subset(self, self.test_indices)

    def __getitem__(self, idx):
        sample={}
        #sample["ap_average"] = self.ap_average[idx]
        sample["date"] = self.dates_str[idx]
        sample["inputs"] = self.input_features[idx]
        sample["tec"] = self.tec[idx]
        sample["dtec"] = self.dtec[idx]
        sample["latitude_classification"] = self.latitude_classification[idx]

        for data_name, data in self.time_series_data.items():
            ####IMPORTANT -> first row, is old data, last row is new data.
            #So sample[data_name][0,:] will contain the data at lagged_index time (older in time), while sample[data_name][-1,:] will contain the data at now_index time (newer in time).
            is_omni_or_jpl = "omni" in data_name or "jpld" in data_name

            resolution_seconds = data["resolution"] * 60 if is_omni_or_jpl else data["resolution"] * 24 * 3600
            lag = data["lag"]
            lag_timedelta = pd.Timedelta(minutes=lag) if is_omni_or_jpl else pd.Timedelta(days=lag)

            now_index = self.date_to_index(
                date=pd.to_datetime(sample['date']),
                date_start=data["date_start"],
                resolution_seconds=resolution_seconds
            )

            lagged_index = self.date_to_index(
                date=pd.to_datetime(sample['date']) - lag_timedelta,
                date_start=data["date_start"],
                resolution_seconds=resolution_seconds
            )

            sample[data_name] = data["data"][lagged_index : (now_index + 1), :]
            #if it's celestrack or SET, also return the storm classification and/or the solar activity classification:
            if data_name.startswith("celestrack"):
                sample["storm_classification"] = data["storm_classification"][now_index]
            if data_name.startswith("set"):
                sample["solar_activity_classification"] = data["solar_activity_classification"][now_index]
        return sample