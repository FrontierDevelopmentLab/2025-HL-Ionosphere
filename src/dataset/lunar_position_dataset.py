"""
Lunar Position Dataset

Computes lunar position parameters for ionosphere modeling.
Provides sublunary point latitude and longitude for any given datetime.
"""

import torch
import math
import datetime
from datetime import timezone
from torch.utils.data import Dataset


class LunarPositionDataset(Dataset):
    def __init__(self, date_start=None, date_end=None, normalize=True, 
                 sampled_cadence=15):
        print('\nLunar Position dataset')
        
        self.normalize = normalize
        self.sampled_cadence = sampled_cadence  # in minutes
        
        # Constants
        self.DEG2RAD = math.pi / 180
        self.RAD2DEG = 180 / math.pi
        
        # Set default date range if not provided
        if date_start is None:
            self.date_start = datetime.datetime(2000, 1, 1, 
                                                tzinfo=timezone.utc)
        else:
            if isinstance(date_start, str):
                date_start = datetime.datetime.fromisoformat(date_start)
            self.date_start = (date_start.replace(tzinfo=timezone.utc) 
                             if date_start.tzinfo is None else date_start)
            
        if date_end is None:
            self.date_end = datetime.datetime(2030, 12, 31, 
                                              tzinfo=timezone.utc)
        else:
            if isinstance(date_end, str):
                date_end = datetime.datetime.fromisoformat(date_end)
            self.date_end = (date_end.replace(tzinfo=timezone.utc) 
                           if date_end.tzinfo is None else date_end)
        
        # Calculate normalization parameters if needed
        if normalize:
            # For normalization: sublunary latitude ranges from ~-28.6 to 28.6°
            # sublunary longitude ranges from -180 to 180 degrees
            self.lat_mean = 0.0
            self.lat_std = 28.6 / 2.0  # approximate std for lunar declination
            self.lon_mean = 0.0
            self.lon_std = 180.0 / 2.0  # approximate std for longitude
        
        print(f'Date range           : {self.date_start} to {self.date_end}')
        print(f'Sampled cadence      : {self.sampled_cadence} minutes')
        print(f'Normalize            : {self.normalize}')
        
    def moon_equatorial_coords(self, dt):
        """Approximate the Moon's RA, Dec (degrees) for a given UTC datetime."""
        # Convert datetime to Julian Day
        Y = dt.year
        M = dt.month
        D = dt.day + (dt.hour + dt.minute/60 + dt.second/3600)/24
        if M <= 2:
            Y -= 1
            M += 12
        A = int(Y / 100)
        B = 2 - A + int(A / 4)
        JD = (int(365.25 * (Y + 4716)) + int(30.6001 * (M + 1)) + 
              D + B - 1524.5)

        # Days since J2000.0
        D = JD - 2451545.0

        # Mean longitude, mean anomaly, argument of latitude (all in degrees)
        L = (218.316 + 13.176396 * D) % 360      # Mean longitude of Moon
        M_moon = (134.963 + 13.064993 * D) % 360  # Moon's mean anomaly
        M_sun = (357.529 + 0.98560028 * D) % 360  # Sun's mean anomaly
        D_moon = (297.850 + 12.190749 * D) % 360  # Moon's mean elongation
        F = (93.272 + 13.229350 * D) % 360        # Argument of latitude

        # Convert to radians
        L_rad = math.radians(L)
        M_moon_rad = math.radians(M_moon)
        M_sun_rad = math.radians(M_sun)
        D_moon_rad = math.radians(D_moon)
        F_rad = math.radians(F)

        # Ecliptic longitude (lambda) and latitude (beta) in degrees
        # Use simplified series for demonstration (omit many small terms)
        lon = (L + 6.289 * math.sin(M_moon_rad) + 
               1.274 * math.sin(2*D_moon_rad - M_moon_rad) + 
               0.658 * math.sin(2*D_moon_rad) + 
               0.214 * math.sin(2*M_moon_rad) - 
               0.186 * math.sin(M_sun_rad))
        lat = (5.128 * math.sin(F_rad) + 
               0.280 * math.sin(M_moon_rad + F_rad) + 
               0.277 * math.sin(M_moon_rad - F_rad) + 
               0.173 * math.sin(2*D_moon_rad - F_rad) + 
               0.055 * math.sin(2*D_moon_rad + F_rad))

        # Convert to radians
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)

        # Obliquity of ecliptic
        eps = 23.439 - 0.0000004 * D
        eps_rad = math.radians(eps)

        # Convert to equatorial coordinates (RA, Dec)
        x = math.cos(lon_rad) * math.cos(lat_rad)
        y = (math.sin(lon_rad) * math.cos(lat_rad) * math.cos(eps_rad) - 
             math.sin(lat_rad) * math.sin(eps_rad))
        z = (math.sin(lon_rad) * math.cos(lat_rad) * math.sin(eps_rad) + 
             math.sin(lat_rad) * math.cos(eps_rad))
        r = math.sqrt(x*x + y*y + z*z)
        dec = math.asin(z/r)
        ra = math.atan2(y, x)

        # Convert to degrees, RA in [0,360)
        ra_deg = (math.degrees(ra) + 360) % 360
        dec_deg = math.degrees(dec)

        return ra_deg, dec_deg

    def greenwich_mean_sidereal_time(self, utc_datetime):
        """Calculate Greenwich Mean Sidereal Time in degrees"""
        days = ((utc_datetime - 
                datetime.datetime(2000, 1, 1, 12, tzinfo=timezone.utc))
                .total_seconds() / 86400.0)
        GMST = 280.46061837 + 360.98564736629 * days
        return GMST % 360
    
    def compute_sublunary_point(self, utc_datetime):
        """Compute sublunary point (latitude, longitude) for given UTC datetime"""
        # Ensure datetime has UTC timezone
        if utc_datetime.tzinfo is None:
            utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
        elif utc_datetime.tzinfo != timezone.utc:
            utc_datetime = utc_datetime.astimezone(timezone.utc)
            
        # Calculate Moon's equatorial coordinates
        ra_moon, dec_moon = self.moon_equatorial_coords(utc_datetime)
        
        # Calculate Greenwich Hour Angle (GHA) of the Moon
        gmst = self.greenwich_mean_sidereal_time(utc_datetime)
        gha_moon = (gmst - ra_moon) % 360
        
        # Sublunary point is at (declination, -GHA)
        # Longitude needs to be wrapped to [-180, 180]
        sublunary_lat = dec_moon
        sublunary_lon = -gha_moon
        while sublunary_lon > 180:
            sublunary_lon -= 360
        while sublunary_lon < -180:
            sublunary_lon += 360
            
        return sublunary_lat, sublunary_lon
    
    def get_date_range(self):
        """Get the date range of this dataset"""
        return self.date_start, self.date_end
    
    def set_date_range(self, date_start, date_end):
        """Set the date range for this dataset"""
        if isinstance(date_start, str):
            date_start = datetime.datetime.fromisoformat(date_start)
        if isinstance(date_end, str):
            date_end = datetime.datetime.fromisoformat(date_end)
            
        self.date_start = (date_start.replace(tzinfo=timezone.utc) 
                          if date_start.tzinfo is None else date_start)
        self.date_end = (date_end.replace(tzinfo=timezone.utc) 
                        if date_end.tzinfo is None else date_end)
        
    def normalize_data(self, data):
        """Normalize lunar position data"""
        if not self.normalize:
            return data
        
        # data is expected to be [latitude, longitude]
        normalized = torch.zeros_like(data)
        normalized[0] = (data[0] - self.lat_mean) / self.lat_std  # latitude
        normalized[1] = (data[1] - self.lon_mean) / self.lon_std  # longitude
        return normalized
    
    def unnormalize_data(self, data):
        """Unnormalize lunar position data"""
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
        """Get lunar position data for given index or datetime"""
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
            
        # Compute sublunary point
        sublunary_lat, sublunary_lon = self.compute_sublunary_point(date)
        
        # Create tensor with [latitude, longitude]
        data = torch.tensor([sublunary_lat, sublunary_lon], dtype=torch.float32)
        
        # Apply normalization if enabled
        if self.normalize:
            data = self.normalize_data(data)
            
        return data