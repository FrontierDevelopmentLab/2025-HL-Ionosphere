#The following script connects to the Madrigal database, retrieves GNSS data from 2010 to July 2025, and downloads the files to a specified directory.
#These maps are not continous and consist of daily lat long grids with a resolution of 1 degree.

# Requires: pip install madrigalWeb

from madrigalWeb.madrigalWeb import MadrigalData
import os
import time 

# 1. Connect to a Madrigal server
murl = "https://cedar.openmadrigal.org"
md = MadrigalData(murl)

# 2. List instruments and find GNSS network
instrs = md.getAllInstruments()
for inst in instrs:
    if "GNSS" in inst.name or "GPS" in inst.name:
        print(inst.code, inst.name)

# Suppose it prints: 8000 World-wide GNSS Receiver Network

# 3. Search for experiments from 2010 to July 2025
inst_code = 8000
exps = md.getExperiments(inst_code,
                         2010, 1, 1, 0, 0, 0,
                         2025, 7, 23, 23, 59, 59)
print(f"Found {len(exps)} experiments.")

# 4. Download files for the first few experiments
user = {
    "fullname": "Your Name",
    "email": "you@example.com",
    "affil": "Your Institution"
}

# Destination folder (you can change this)
output_folder = "/mnt/ionosphere-data/madrigal_data"
# Create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for exp in exps:  # just first 3 for demo
    try:
        #print(f"Processing experiment: {exp.name} ({exp.id})")
        files = md.getExperimentFiles(exp.id)
    except Exception as e:
        print(f"Error retrieving files for experiment {exp.id}: {e}")
        continue

    for f in files:
        filename = os.path.basename(f.name)

        if not filename.startswith("gps"):
            continue # Skip non-GNSS files

        dest_path = os.path.join(output_folder, filename)

        if not os.path.exists(dest_path):  # skip if already downloaded
            print(f"Downloading {filename} ...")
            md.downloadFile(f.name, dest_path,
                            user["fullname"], user["email"],
                            user["affil"], "hdf5")
        else:
            print(f"Already downloaded: {filename}")

        time.sleep(2)  # be nice to the server, avoid rate limits
