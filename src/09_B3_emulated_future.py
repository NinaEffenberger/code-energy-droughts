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
file_pattern = "*id-012*2090-2099*.nc"

files = glob.glob(os.path.join(data_folder, file_pattern))
time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/cmip6/test/extrapolation"

print(f"Found {len(files)} files.")

for file_path in files:
    print(f"Processing {file_path}...")
    data_pred = xr.open_dataset(
        file_path,
        group="prediction",
        chunks={"ensemble": 2, "time": 365, "y": 128, "x": 128},
    )
    season_labels = utils.get_season_labels(file_path, time_folder)

    # 1. Spatial mean for each ensemble & time
    rsds_mean = data_pred["rsds"].mean(dim=("y", "x"))
    wind_mean = data_pred["sfcWind"].mean(dim=("y", "x"))

    # 2. Compute 20th percentile threshold per ensemble
    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    # 3. Binary time series: 1 if below threshold, else 0
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # 4. Combine 
    combined = (rsds_bin & wind_bin).astype("int8")
    # 5. Stack into one array (ensemble, time, variable)
    result = xr.concat([rsds_bin, wind_bin, combined], dim="variable")
    result = result.assign_coords(variable=["rsds", "wind", "combined"])

    combined = combined.assign_coords(
        season=("time", season_labels[0 : combined.shape[1]])
    )

    # Group by season and sum over time
    season_counts = combined.groupby("season").sum(dim="time")

    filename = file_path.split("/")[-1].rsplit(".", 1)[0]
    season_counts_np = (
        season_counts.compute().values
    )  # shape: (ensemble, number_of_seasons)
    np.save(
        "/cluster/home/neffenberger/energy-droughts-emulator/data/CMIP6/corrdiff/future/"
        + filename
        + ".npy",
        season_counts_np,
    )
    gc.collect()
