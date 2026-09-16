import xarray as xr
import numpy as np
import glob
import os
import gc

# Folder containing your files
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/cmip6/test/extrapolation"
solar_files = glob.glob(os.path.join(data_folder, "*rsds*EUROPE*.nc"))

print(f"Found {len(solar_files)} files.")


grid_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"
orig_data = xr.open_dataset(grid_path)


time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/cmip6/test/extrapolation"


# Get RCM domain in lat/lon
lat_min = orig_data["lat"].min().item()
lat_max = orig_data["lat"].max().item()
lon_min = orig_data["lon"].min().item()
lon_max = orig_data["lon"].max().item()

print(lat_min, lat_max, lon_min, lon_max)


for solar_file in solar_files:
    print(f"Processing {solar_file}...")
    base_name = os.path.basename(solar_file)
    parts = base_name.split("_")
    model_name = parts[2]
    key_word = "*sfcWind*" + model_name + "*EUROPE*.nc"
    wind_file = glob.glob(os.path.join(data_folder, key_word))
    solar_data = xr.open_dataset(solar_file).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
    wind_data = xr.open_dataset(wind_file[0]).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
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

    # 1. Spatial mean for each ensemble & time
    rsds_mean = solar_data["rsds"].mean(dim=("lon", "lat"))
    wind_mean = wind_data["sfcWind"].mean(dim=("lon", "lat"))

    # 2. Compute 20th percentile threshold per ensemble
    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    # 3. Binary time series: 1 if below threshold, else 0
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # 4. Combine
    combined = (rsds_bin & wind_bin).astype("int8")
    # 5. Stack into one array (ensemble, time, variable)

    combined = combined.assign_coords(season=("time", season_labels))

    # Group by season and sum over time
    season_counts = combined.groupby("season").sum(dim="time")

    season_counts_np = (
        season_counts.compute().values
    )  # shape: (ensemble, number_of_seasons)
    np.save(
        "../plotting_data/CMIP6/GCM/future/" + model_name + ".npy", season_counts_np
    )
    gc.collect()
