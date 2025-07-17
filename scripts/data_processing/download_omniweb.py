import os
import requests
import pandas as pd
import numpy as np

def download_omni_5min(year: int, output_dir='data'):
    """
    Download OMNI 5-min resolution ASCII file for a given year.
    """
    url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{year}.asc"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"omni_5min{year}.asc")

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return file_path

    print(f"Downloading: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Download complete: {file_path}")
        return file_path
    else:
        raise Exception(f"[✗] Download failed: HTTP {response.status_code}")

def load_omni_5min_range(start_year: int,
                         end_year: int,
                         output_dir: str = "data",
                         concat: bool = True,
                         *,
                         save_csv: bool = False,
                         csv_dir: str = "csv",
                         csv_compress=None):
    """
    Download and parse OMNI 5‑min data for a range of years.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range.
    output_dir : str
        Where .asc files are saved.
    concat : bool
        If True return one big DataFrame, else dict(year -> df).
    save_csv : bool
        If True, write each year's DataFrame to <csv_dir>/omni_5minYYYY.csv.
    csv_dir : str
        Folder for CSVs (created if it doesn't exist).
    csv_compress : bool or str
        Compression for CSVs.  True → "gzip".  Or pass "bz2", "zip", etc.

    Returns
    -------
    pandas.DataFrame or dict[int, pandas.DataFrame]
    """
    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)
        if csv_compress is True:
            csv_compress = "gzip"

    all_dfs = {}

    for year in range(start_year, end_year + 1):
        print(f"\nProcessing {year} …")
        try:
            asc_path = download_omni_5min(year, output_dir)
            df = parse_omni_5min(asc_path)
            all_dfs[year] = df

            # ── save CSV ─────────────────────────────────────────────
            if save_csv:
                csv_path = os.path.join(csv_dir, f"omni_5min{year}.csv")
                df.to_csv(csv_path, compression=csv_compress)
                print(f"    ↳ saved {csv_path}")

        except Exception as e:
            print(f"[✗] {year}: {e}")

    return (pd.concat(all_dfs.values()).sort_index()
            if concat else all_dfs)

if __name__ == "__main__":
    # Download–parse 2022‑2025, keep DataFrames separate,
    # and save compressed CSVs into folder "omni_csv":
    dfs_by_year = load_omni_5min_range(
        1981, 2025,
        concat=False,
        save_csv=True,
        csv_dir="/mnt/ionosphere-data/omniweb",
        csv_compress="gzip"   # or True
    )

    # Access a single year:
    print(dfs_by_year[2023].head())

    # The gzip files are now at omni_csv/omni_5min2022.csv.gz, etc.

