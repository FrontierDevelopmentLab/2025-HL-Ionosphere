"""
Solar Position Dataset

Computes solar position parameters for ionosphere modeling.
Provides subsolar point latitude and longitude for any given datetime.
"""

import torch
import math
import numpy as np
import datetime
from datetime import timezone
from torch.utils.data import Dataset

class SolarPositionDataset(Dataset):
    def __init__(self, date_start=None, date_end=None, normalize=True, sampled_cadence=15):
        print('\nSolar Position dataset')
        
        self.normalize = normalize
        self.sampled_cadence = sampled_cadence  # in minutes
        
        # Constants
        self.DEG2RAD = math.pi / 180
        self.RAD2DEG = 180 / math.pi
        
        # Set default date range if not provided
        if date_start is None:
            self.date_start = datetime.datetime(2000, 1, 1, tzinfo=timezone.utc)
        else:
            if isinstance(date_start, str):
                date_start = datetime.datetime.fromisoformat(date_start)
            self.date_start = date_start.replace(tzinfo=timezone.utc) if date_start.tzinfo is None else date_start
            
        if date_end is None:
            self.date_end = datetime.datetime(2030, 12, 31, tzinfo=timezone.utc)
        else:
            if isinstance(date_end, str):
                date_end = datetime.datetime.fromisoformat(date_end)
            self.date_end = date_end.replace(tzinfo=timezone.utc) if date_end.tzinfo is None else date_end
        
        # Calculate normalization parameters if needed
        if normalize:
            # For normalization: subsolar latitude ranges from -23.44 to 23.44 degrees
            # subsolar longitude ranges from -180 to 180 degrees
            self.lat_mean = 0.0
            self.lat_std = 23.44 / 2.0  # approximate std for declination
            self.lon_mean = 0.0
            self.lon_std = 180.0 / 2.0  # approximate std for longitude
        
        print(f'Date range           : {self.date_start} to {self.date_end}')
        print(f'Sampled cadence      : {self.sampled_cadence} minutes')
        print(f'Normalize            : {self.normalize}')
        
    def day_of_year(self, date):
        """Calculate day of year for given date"""
        return date.timetuple().tm_yday
    
    def solar_declination(self, n):
        """Calculate solar declination angle in degrees"""
        return 23.44 * math.sin(self.DEG2RAD * (360 / 365.0 * (n - 81)))
    
    def greenwich_mean_sidereal_time(self, utc_datetime):
        """Calculate Greenwich Mean Sidereal Time in degrees"""
        days = (utc_datetime - datetime.datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
        GMST = 280.46061837 + 360.98564736629 * days
        return GMST % 360
    
    def compute_subsolar_point(self, utc_datetime):
        """Compute subsolar point (latitude, longitude) for given UTC datetime"""
        # Ensure datetime has UTC timezone
        if utc_datetime.tzinfo is None:
            utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
        elif utc_datetime.tzinfo != timezone.utc:
            utc_datetime = utc_datetime.astimezone(timezone.utc)
            
        # Calculate day of year and solar declination
        n = self.day_of_year(utc_datetime)
        decl = self.solar_declination(n)
        
        # Calculate Greenwich Hour Angle (GHA) of the Sun
        gmst = self.greenwich_mean_sidereal_time(utc_datetime)
        gha = gmst  # Approximation: GHA = GMST
        
        # Subsolar point is at (declination, -GHA)
        # Longitude needs to be wrapped to [-180, 180]
        subsolar_lat = decl
        subsolar_lon = -gha
        while subsolar_lon > 180:
            subsolar_lon -= 360
        while subsolar_lon < -180:
            subsolar_lon += 360
            
        return subsolar_lat, subsolar_lon
    
    def get_date_range(self):
        """Get the date range of this dataset"""
        return self.date_start, self.date_end
    
    def set_date_range(self, date_start, date_end):
        """Set the date range for this dataset"""
        if isinstance(date_start, str):
            date_start = datetime.datetime.fromisoformat(date_start)
        if isinstance(date_end, str):
            date_end = datetime.datetime.fromisoformat(date_end)
            
        self.date_start = date_start.replace(tzinfo=timezone.utc) if date_start.tzinfo is None else date_start
        self.date_end = date_end.replace(tzinfo=timezone.utc) if date_end.tzinfo is None else date_end
        
    def normalize_data(self, data):
        """Normalize solar position data"""
        if not self.normalize:
            return data
        
        # data is expected to be [latitude, longitude]
        normalized = torch.zeros_like(data)
        normalized[0] = (data[0] - self.lat_mean) / self.lat_std  # latitude
        normalized[1] = (data[1] - self.lon_mean) / self.lon_std  # longitude
        return normalized
    
    def unnormalize_data(self, data):
        """Unnormalize solar position data"""
        if not self.normalize:
            return data
            
        # data is expected to be [latitude, longitude]
        unnormalized = torch.zeros_like(data)
        unnormalized[0] = data[0] * self.lat_std + self.lat_mean  # latitude
        unnormalized[1] = data[1] * self.lon_std + self.lon_mean  # longitude
        return unnormalized
    
    def __len__(self):
        """Calculate dataset length based on date range and cadence"""
        total_minutes = (self.date_end - self.date_start).total_seconds() / 60
        return int(total_minutes / self.sampled_cadence)
    
    def __getitem__(self, index):
        """Get solar position data for given index or datetime"""
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        elif isinstance(index, int):
            # Calculate datetime from index
            minutes_offset = index * self.sampled_cadence
            date = self.date_start + datetime.timedelta(minutes=minutes_offset)
        else:
            raise TypeError("Index must be either a datetime, string, or integer.")
        
        # Ensure date is within range
        if date < self.date_start or date > self.date_end:
            return None
            
        # Compute subsolar point
        subsolar_lat, subsolar_lon = self.compute_subsolar_point(date)
        
        # Create tensor with [latitude, longitude]
        data = torch.tensor([subsolar_lat, subsolar_lon], dtype=torch.float32)
        
        # Apply normalization if enabled
        if self.normalize:
            data = self.normalize_data(data)
            
        return data