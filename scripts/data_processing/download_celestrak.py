import pandas as pd
from pathlib import Path
import numpy as np


def swall_to_timeseries(infile: str, outfile: str,
                        *, kp_div10=True, drop_na=True):
    """
    Convert CelesTrak SW‑All table to 3‑hourly Kp/Ap time series.

    Parameters
    ----------
    infile  : str | Path
        Path to SW‑All.csv (downloaded from CelesTrak).
    outfile : str | Path
        Destination CSV with columns: Datetime, Kp, Ap
    kp_div10 : bool
        Divide Kp by 10 (CelesTrak stores 53 → 5.3).  Leave False if
        your file already has decimal Kp.
    drop_na : bool
        Remove rows where either Kp or Ap is missing.
    """

    infile = Path(infile)
    outfile = Path(outfile)

    df = pd.read_csv(infile, parse_dates=["DATE"])

    # names of the 3‑hourly columns
    kp_cols = [f"KP{i}" for i in range(1, 9)]
    ap_cols = [f"AP{i}" for i in range(1, 9)]

    if kp_div10:
        df[kp_cols] = df[kp_cols]      # 53 → 5.3

    # ── build long format ──────────────────────────────────────
    records = []
    for idx, row in df.iterrows():
        base_date = row["DATE"]
        for k in range(8):                    # 0–7
            ts = base_date + pd.Timedelta(hours=3 * k)
            kp_val = row[kp_cols[k]]
            ap_val = row[ap_cols[k]]
            records.append((ts, kp_val, ap_val))

    ts_df = pd.DataFrame(records, columns=["Datetime", "Kp", "Ap"]) \
             .set_index("Datetime") \
             .sort_index()

    if drop_na:
        ts_df = ts_df.dropna(subset=["Kp", "Ap"])

    ts_df.to_csv(outfile, index_label="Datetime")
    print(f"[✓] written {outfile}")


# ---------------- example -------------------
if __name__ == "__main__":
    swall_to_timeseries(
        infile="/home/simone/Downloads/celestrak_SW-All.csv",        # ← your original file PLEASE CHANGE
        outfile="kp_ap_timeseries.csv",
        kp_div10=True                         # set False if Kp already decimal
    )

# import pandas as pd
# import matplotlib.pyplot as plt

# ts = pd.read_csv("kp_ap_timeseries.csv",
#                  parse_dates=["Datetime"],
#                  index_col="Datetime")

# # Example: yearly Kp trend (first 00‑03 UT bin)
# plt.plot(ts["Kp"].resample("365D").mean())
# plt.ylabel("Kp (mean, yearly)")
# plt.show()
