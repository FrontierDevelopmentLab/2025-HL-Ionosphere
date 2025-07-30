from src.utils.subsolar_functions import compute_sublunary_point
from src.utils.subsolar_functions import compute_subsolar_point
import datetime

datetime_str = "2025-01-01 00:00:00"
utc_datetime = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
utc_datetime = utc_datetime.replace(tzinfo=datetime.timezone.utc)

# Compute subsolar point
subsolar_lat, subsolar_lon = compute_subsolar_point(utc_datetime)
print(f"Subsolar Point: Latitude = {subsolar_lat}, Longitude = {subsolar_lon}")

# Compute sublunar point
sublunar_lat, sublunar_lon = compute_sublunary_point(utc_datetime)
print(f"Sublunar Point: Latitude = {sublunar_lat}, Longitude = {sublunar_lon}")