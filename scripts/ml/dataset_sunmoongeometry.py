import torch
from torch.utils.data import Dataset
import datetime
import numpy as np
import skyfield.api


class SunMoonGeometry(Dataset):
    def __init__(self, date_start=None, date_end=None, delta_minutes=15, image_size=(180, 360), normalize=True):
        self.date_start = date_start
        self.date_end = date_end
        self.delta_minutes = delta_minutes
        self.image_size = image_size
        self.normalize = normalize

        if self.date_start is None:
            self.date_start = datetime.datetime(2010, 5, 13, 0, 0, 0)
        if self.date_end is None:
            self.date_end = datetime.datetime(2024, 8, 1, 0, 0, 0)

        current_date = self.date_start
        self.dates = []
        while current_date <= self.date_end:
            self.dates.append(current_date)
            current_date += datetime.timedelta(minutes=self.delta_minutes)

        self.dates_set = set(self.dates)
        self.name = 'SunMoonGeometry'

        print('\nSun and Moon Geometry')
        print('Start date              : {}'.format(self.date_start))
        print('End date                : {}'.format(self.date_end))
        print('Delta                   : {} minutes'.format(self.delta_minutes))
        print('Image size              : {}'.format(self.image_size))

        # Don't initialize Skyfield objects here
        self._ts = None
        self._eph = None
        self._earth_body = None
        self._sun_body = None
        self._moon_body = None

    def _init_skyfield_objects(self):
        """Initialize Skyfield objects once per worker process."""
        if self._ts is None:
            self._ts = skyfield.api.load.timescale()
            self._eph = skyfield.api.load('de421.bsp')
            self._earth_body = self._eph['earth']
            self._sun_body = self._eph['sun']
            self._moon_body = self._eph['moon']

    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, index):
        if isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        elif isinstance(index, int):
            if index < 0 or index >= len(self.dates):
                raise IndexError("Index out of range for the dataset.")
            date = self.dates[index]
        else:
            raise ValueError('Expecting index to be datetime.datetime or str (in the format of 2022-11-01T00:01:00), but got {}'.format(type(index)))

        if date not in self.dates_set:
            raise ValueError('Date {} not found in the dataset'.format(date))

        if date.tzinfo is None:
            date = date.replace(tzinfo=datetime.timezone.utc)

        # get the sun data and the moon data and concat everything in the channel dimension
        sun_data = self.generate_solar_data(date, normalized=self.normalize)
        moon_data = self.generate_lunar_data(date, normalized=self.normalize)

        sun_zenith_angle_map, sun_subsolar_coords, sun_antipode_coords, sun_distance = sun_data
        moon_zenith_angle_map, moon_sublunar_coords, moon_antipode_coords, moon_distance = moon_data

        sun_zenith_angle_map = torch.tensor(sun_zenith_angle_map, dtype=torch.float32)
        sun_data = torch.tensor(sun_subsolar_coords + sun_antipode_coords + (sun_distance,), dtype=torch.float32).view(-1, 1, 1).expand(-1, self.image_size[0], self.image_size[1])
        moon_zenith_angle_map = torch.tensor(moon_zenith_angle_map, dtype=torch.float32)
        moon_data = torch.tensor(moon_sublunar_coords + moon_antipode_coords + (moon_distance,), dtype=torch.float32).view(-1, 1, 1).expand(-1, self.image_size[0], self.image_size[1])

        # Concatenate the maps and data along the channel dimension
        combined_data = torch.cat((sun_zenith_angle_map.unsqueeze(0), sun_data, 
                                   moon_zenith_angle_map.unsqueeze(0), moon_data), dim=0)
        
        return combined_data, date.isoformat()

    def _normalize_coords(self, lat, lon):
        """Normalizes geographic coordinates into a 3D vector suitable for ML.

        This transforms longitude into a cyclical representation using sin and cos,
        and scales latitude to the [-1, 1] range.

        Args:
            lat (float): Latitude in degrees (-90 to 90).
            lon (float): Longitude in degrees (-180 to 180).

        Returns:
            tuple: A 3-element tuple (normalized_latitude, lon_x, lon_y).
        """
        lat_norm = lat / 90.0
        lon_rad = np.radians(lon)
        lon_x = np.cos(lon_rad)
        lon_y = np.sin(lon_rad)
        return (lat_norm, lon_x, lon_y)

    def generate_solar_data(self, utc_dt, normalized=True):
        """Generates the Sun's zenith angle map and related geometric data.

        Args:
            utc_dt (datetime): The timezone-aware datetime for the calculation (must be UTC).
            normalized (bool): If True (default), all outputs are normalized for ML.
                If False, outputs are in degrees and kilometers.

        Returns:
            tuple: A tuple containing four items: (map_array, subsolar_coords,
                antipode_coords, distance). The format of the coordinates and
                distance depends on the 'normalized' flag.
                - If normalized, coords are a 3D vector and distance is in AU.
                - If not, coords are (lat, lon) degrees and distance is in km.
        """
        return self._generate_map_data(utc_dt, 'sun', normalized=normalized)

    def generate_lunar_data(self, utc_dt, normalized=True):
        """Generates the Moon's zenith angle map and related geometric data.

        Args:
            utc_dt (datetime): The timezone-aware datetime for the calculation (must be UTC).
            normalized (bool): If True (default), all outputs are normalized for ML.
                If False, outputs are in degrees and kilometers.

        Returns:
            tuple: A tuple containing four items: (map_array, sublunar_coords,
                antipode_coords, distance). The format of the coordinates and
                distance depends on the 'normalized' flag.
                - If normalized, coords are a 3D vector and distance is in LD.
                - If not, coords are (lat, lon) degrees and distance is in km.
        """
        return self._generate_map_data(utc_dt, 'moon', normalized=normalized)

    def _generate_map_data(self, utc_dt, body_name, normalized):
        """
        Helper function to generate a zenith angle map and data for a celestial body.

        Args:
            utc_dt (datetime): The time for the calculation.
            body_name (str): The name of the celestial body ('sun' or 'moon').
            normalized (bool): Controls the output units for the map and distance.

        Returns:
            tuple: A tuple containing the map, sub-point coords, antipode coords, and distance.
        """
        AVG_LUNAR_DISTANCE_KM = 384400.0  # 1 Lunar Distance (LD)

        # Initialize objects if needed (once per worker)
        self._init_skyfield_objects()
        
        celestial_body = self._sun_body if body_name == 'sun' else self._moon_body
        t = self._ts.from_datetime(utc_dt)

        astrometric = self._earth_body.at(t).observe(celestial_body)
        subpoint = skyfield.api.wgs84.subpoint_of(astrometric)

        sub_lat = subpoint.latitude.degrees
        sub_lon = subpoint.longitude.degrees
        
        antipode_lat = -sub_lat
        antipode_lon = sub_lon + 180
        if antipode_lon > 180:
            antipode_lon -= 360
        
        lat = np.linspace(89.5, -89.5, self.image_size[0])
        lon = np.linspace(-179.5, 179.5, self.image_size[1])
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lat_rad = np.radians(lat_grid)
        lon_rad = np.radians(lon_grid)
        sub_lat_rad = np.radians(sub_lat)
        sub_lon_rad = np.radians(sub_lon)
        
        hour_angle_rad = lon_rad - sub_lon_rad
        cos_z = (np.sin(lat_rad) * np.sin(sub_lat_rad) +
                np.cos(lat_rad) * np.cos(sub_lat_rad) * np.cos(hour_angle_rad))
        cos_z = np.clip(cos_z, -1.0, 1.0)

        if normalized:
            distance = astrometric.distance().au if body_name == 'sun' else astrometric.distance().km / AVG_LUNAR_DISTANCE_KM
            sub_coords = self._normalize_coords(sub_lat, sub_lon)
            antipode_coords = self._normalize_coords(antipode_lat, antipode_lon)
            return cos_z, sub_coords, antipode_coords, distance
        else:
            distance = astrometric.distance().km
            sub_coords = (sub_lat, sub_lon)
            antipode_coords = (antipode_lat, antipode_lon)
            zenith_angle_deg = np.degrees(np.arccos(cos_z))
            return zenith_angle_deg, sub_coords, antipode_coords, distance