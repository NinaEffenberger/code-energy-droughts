import xarray as xr
import numpy as np
import glob
import os
import gc
import utils
import importlib

importlib.reload(utils)


# Folder containing your files
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"
solar_files = glob.glob(os.path.join(data_folder, "*rsds*EUROPE*.nc"))

print(f"Found {len(solar_files)} files.")


grid_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"
orig_data = xr.open_dataset(grid_path)


target_lat = orig_data.lat.values[50, 50]
target_lon = orig_data.lon.values[50, 50]
time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"


for solar_file in solar_files:
    print(f"Processing {solar_file}...")
    base_name = os.path.basename(solar_file)
    parts = base_name.split("_")
    model_name = parts[2]
    key_word = "*sfcWind*" + model_name + "*EUROPE*.nc"
    wind_file = glob.glob(os.path.join(data_folder, key_word))
    solar_data = xr.open_dataset(solar_file)
    wind_data = xr.open_dataset(wind_file[0])
    months = solar_data.time.dt.month

    seasons = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }

    # Convert to pandas Series to map
    season_labels = months.to_series().map(seasons).values

    # 1. Extract data for the target location

    solar_point = solar_data.sel(lat=target_lat, lon=target_lon, method="nearest")[
        "rsds"
    ]
    wind_point = wind_data.sel(lat=target_lat, lon=target_lon, method="nearest")[
        "sfcWind"
    ]
    # 2. Compute 20th percentile threshold per ensemble
    rsds_thresh = solar_point.quantile(0.2, dim="time")
    wind_thresh = wind_point.quantile(0.2, dim="time")

    # 3. Binary time series: 1 if below threshold, else 0
    rsds_bin = (solar_point <= rsds_thresh).astype("int8")
    wind_bin = (wind_point <= wind_thresh).astype("int8")

    # 4. Combine (e.g., require both low → 1, else 0)
    combined = (rsds_bin & wind_bin).astype("int8")
    # 5. Stack into one array (ensemble, time, variable)
    combined = combined.assign_coords(season=("time", season_labels))

    # Group by season and sum over time
    season_counts = combined.groupby("season").sum(dim="time")

    season_counts_np = (
        season_counts.compute().values
    )  # shape: (ensemble, number_of_seasons)
    np.save(
        "../plotting_data/single_location/GCM/" + model_name + ".npy", season_counts_np
    )
    gc.collect()

# Folder containing your files
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/benchmarks/corrdiff/"
file_pattern = "*id-011*2090-2099*.nc"


files = glob.glob(os.path.join(data_folder, file_pattern))
time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"

print(f"Found {len(files)} files.")

for file_path in files[1:]:
    print(f"Processing {file_path}...")
    data_pred = xr.open_dataset(
        file_path,
        group="prediction",
        chunks={"ensemble": 2, "time": 365, "y": 1, "x": 1},
    )
    # ys are upside down, see 0047_locations
    data_pred = data_pred.isel(y=slice(None, None, -1))
    season_labels = utils.get_season_labels(file_path, time_folder)

    # 1. Spatial mean for each ensemble & time
    rsds_mean = data_pred.sel(x=data_pred.x[50], y=data_pred.y[50])["rsds"]
    wind_mean = data_pred.sel(x=data_pred.x[50], y=data_pred.y[50])["sfcWind"]

    # 2. Compute 20th percentile threshold per ensemble
    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    # 3. Binary time series: 1 if below threshold, else 0
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # 4. Combine (e.g., require both low → 1, else 0)
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
        "../plotting_data/single_location/emulated/" + filename + ".npy",
        season_counts_np,
    )
    gc.collect()


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
    rsds_mean = data_pred.sel(x=data_pred.x[50], y=data_pred.y[50])["rsds"]
    wind_mean = data_pred.sel(x=data_pred.x[50], y=data_pred.y[50])["sfcWind"]

    # 2. Compute 20th percentile threshold per ensemble
    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    # 3. Binary time series: 1 if below threshold, else 0
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # 4. Combine (e.g., require both low → 1, else 0)
    combined = (rsds_bin & wind_bin).astype("int8")
    # 5. Stack into one array (ensemble, time, variable)
    result = xr.concat([rsds_bin, wind_bin, combined], dim="variable")
    result = result.assign_coords(variable=["rsds", "wind", "combined"])

    combined = combined.assign_coords(season=("time", season_labels[0 : len(combined)]))

    # Group by season and sum over time
    season_counts = combined.groupby("season").sum(dim="time")

    filename = file_path.split("/")[-1].rsplit(".", 1)[0]
    season_counts_np = (
        season_counts.compute().values
    )  # shape: (ensemble, number_of_seasons)
    np.save(
        "../plotting_data/single_location/RCM/" + filename + ".npy", season_counts_np
    )
    gc.collect()
