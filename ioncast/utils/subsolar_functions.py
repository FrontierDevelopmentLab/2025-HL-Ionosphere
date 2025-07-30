import torch
import math
import numpy as np
import datetime
from datetime import timezone

DEG2RAD = math.pi / 180
RAD2DEG = 180 / math.pi

# Function that takes in time of day and returns subsolar point lat, lon
def day_of_year(date):
    """Calculate day of year for given date"""
    return date.timetuple().tm_yday

def solar_declination(n):
    """Calculate solar declination angle in degrees"""
    return 23.44 * math.sin(DEG2RAD * (360 / 365.0 * (n - 81)))

def greenwich_mean_sidereal_time(utc_datetime):
    """Calculate Greenwich Mean Sidereal Time in degrees"""
    days = (utc_datetime - datetime.datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0
    GMST = 280.46061837 + 360.98564736629 * days
    return GMST % 360

def compute_subsolar_point(utc_datetime):
    """Compute subsolar point (latitude, longitude) for given UTC datetime"""
    # Ensure datetime has UTC timezone
    if utc_datetime.tzinfo is None:
        utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
    elif utc_datetime.tzinfo != timezone.utc:
        utc_datetime = utc_datetime.astimezone(timezone.utc)
        
    # Calculate day of year and solar declination
    n = day_of_year(utc_datetime)
    decl = solar_declination(n)
    
    # Calculate Greenwich Hour Angle (GHA) of the Sun
    gmst = greenwich_mean_sidereal_time(utc_datetime)
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

# Function that takes in time of day and returns sublunar point lat, lon
def moon_equatorial_coords(dt):
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

def greenwich_mean_sidereal_time(utc_datetime):
    """Calculate Greenwich Mean Sidereal Time in degrees"""
    days = ((utc_datetime - 
            datetime.datetime(2000, 1, 1, 12, tzinfo=timezone.utc))
            .total_seconds() / 86400.0)
    GMST = 280.46061837 + 360.98564736629 * days
    return GMST % 360

def compute_sublunar_point(utc_datetime):
    """Compute sublunar point (latitude, longitude) for given UTC datetime"""
    # Ensure datetime has UTC timezone
    if utc_datetime.tzinfo is None:
        utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
    elif utc_datetime.tzinfo != timezone.utc:
        utc_datetime = utc_datetime.astimezone(timezone.utc)
        
    # Calculate Moon's equatorial coordinates
    ra_moon, dec_moon = moon_equatorial_coords(utc_datetime)
    
    # Calculate Greenwich Hour Angle (GHA) of the Moon
    gmst = greenwich_mean_sidereal_time(utc_datetime)
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