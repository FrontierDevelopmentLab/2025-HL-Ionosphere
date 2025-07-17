'''
File to process space environment technology Indices_F10.txt from https://login.spacenvironment.net/login
'''

#Separating file 

import pandas as pd
from datetime import datetime, timedelta

def parse_jb2008_solar_file(infile: str, outfile: str):
    """
    Parse JB2008 daily solar flux file and write clean CSV.

    Parameters
    ----------
    infile : str
        Path to JB2008-like formatted file with F10/S10/M10/Y10 data.
    outfile : str
        Path to write output CSV with datetime and solar indices.
    """
    data = []
    with open(infile, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            year = int(parts[0])
            doy = int(parts[1])
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            row = {
                "Datetime": date.strftime("%Y-%m-%d"),
                "F10": float(parts[3]),
                "F81c": float(parts[4]),
                "S10": float(parts[5]),
                "S81c": float(parts[6]),
                "M10": float(parts[7]),
                "M81c": float(parts[8]),
                "Y10": float(parts[9]),
                "Y81c": float(parts[10]),
            }
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(outfile, index=False)
    print(f"Saved clean solar flux file to {outfile}")


#example usage
parse_jb2008_solar_file(
    infile="/home/simone/Desktop/PhD/fdl/datasets/Indices_F10.txt",         # ← your raw input file
    outfile="/home/simone/Desktop/PhD/fdl/datasets/Indices_F10.csv"   # → clean output CSV
)
