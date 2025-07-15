'''
This file downloads daily TEC global ionosphere maps (GIMs) from CDDIS: https://cddis.nasa.gov/archive/gnss/products/ionex/ 

File format: NNNNDDD0.YYi
NNNN: name of the organization
DDD: day of the year (001-365)
0: flag
YY: year (2000 + YY)
i: some index, unsure
'''

import requests
import netrc
import os
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import subprocess

# ==== Configuration ====
start_year = 2023
end_year = 2024
base_url = "https://cddis.nasa.gov/archive/gnss/products/ionex"
save_dir = "/home/simone/Desktop/PhD/fdl/datasets/vtec_data"
desired_prefix = "uqrg"  # or "upcg" if you want the CODE-like GIM
# ======================

os.makedirs(save_dir, exist_ok=True)

try: 
    auth_data = netrc.netrc()
    username, _, password = auth_data.authenticators("urs.earthdata.nasa.gov")
except FileNotFoundError:
    raise RuntimeError("No .netrc file found. Please create one with your Earthdata credentials.")
except TypeError:
    raise RuntimeError("Please set up your .netrc file with Earthdata credentials.")


# Start session - ATTENTION: COMMENT PRINT STATEMENT AFTER TESTING
with requests.Session() as session:
    session.auth = (username, password)
    session.headers.update({'User-Agent': 'simone-ionex-script/1.0'})

    for year in range(start_year, end_year + 1):
        year_short = str(year)[-2:]

        for doy in range(1,367):
            doy_str = f"{doy:03d}"  # Format day of year as three digits
            file_name = f"{desired_prefix}{doy_str}0.{year_short}i.Z"
            folder_url = f"{base_url}/{year}/{doy_str}/"
            file_url = folder_url + file_name
            
            print(f"Trying to download {file_name} from {file_url}")

            #response = session.get(folder_url)
            response = session.get(file_url)

            if response.status_code == 200:
                local_path = os.path.join(save_dir, file_name)
                with open(local_path, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {file_name} to {local_path}")


                try:
                    subprocess.run(['uncompress', local_path], check=True)
                    print(f"Uncompressed {file_name} successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to uncompress {file_name}: {e}")
                except FileNotFoundError:
                    print("Uncompress command not found. Please install 'uncompress' utility.")
                        

            else:
                print(f"File {file_name} not found at {file_url}. Status code: {response.status_code}")
                continue
            