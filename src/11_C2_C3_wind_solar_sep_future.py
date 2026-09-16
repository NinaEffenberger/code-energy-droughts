import xarray as xr
import numpy as np
import glob
import os
import gc
import utils
import importlib

importlib.reload(utils)

# Folder containing your files
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/benchmarks/corrdiff/"
file_pattern = "*id-011*2090-2099*.nc"

files = glob.glob(os.path.join(data_folder, file_pattern))
print(f"Found {len(files)} files.")
time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/"

for file_path in files:
    print(f"Processing {file_path}...")
    data_pred = xr.open_dataset(file_path, group="truth")
    season_labels = utils.get_season_labels(file_path, time_folder)

    # 1. Spatial mean for each ensemble & time
    rsds_mean = data_pred["rsds"].mean(dim=("y", "x"))
    wind_mean = data_pred["sfcWind"].mean(dim=("y", "x"))

    # 1. Spatial mean for each ensemble & time
    rsds_mean = data_pred["rsds"].mean(dim=("y", "x"))
    wind_mean = data_pred["sfcWind"].mean(dim=("y", "x"))

    # 2. Compute 20th percentile threshold per ensemble
    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    # 3. Binary time series: 1 if below threshold, else 0
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # 4. Assign season coordinates
    rsds_bin = rsds_bin.assign_coords(season=("time", season_labels))
    wind_bin = wind_bin.assign_coords(season=("time", season_labels))

    # 5. Group by season and sum over time
    rsds_season_counts = rsds_bin.groupby("season").sum(dim="time")
    wind_season_counts = wind_bin.groupby("season").sum(dim="time")

    # 6. Convert to numpy
    rsds_counts_np = rsds_season_counts.compute().values
    wind_counts_np = wind_season_counts.compute().values

    # 7. Save separately
    filename = file_path.split("/")[-1].rsplit(".", 1)[0]
    np.save(f"../plotting_data/orig_RCM_solar_id011/{filename}.npy", rsds_counts_np)
    np.save(f"../plotting_data/orig_RCM_wind_id011/{filename}.npy", wind_counts_np)

    gc.collect()
