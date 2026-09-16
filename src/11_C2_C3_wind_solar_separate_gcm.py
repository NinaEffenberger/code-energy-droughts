import xarray as xr
import numpy as np
import glob
import os
import gc

# Folder containing your files
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"
solar_files = glob.glob(os.path.join(data_folder, "*rsds*EUROPE*.nc"))

print(f"Found {len(solar_files)} files.")


grid_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"
orig_data = xr.open_dataset(grid_path)
lons = orig_data.lon.values
lats = orig_data.lat.values


time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"

# Get RCM domain in lat/lon
lat_min = orig_data["lat"].min().item()
lat_max = orig_data["lat"].max().item()
lon_min = orig_data["lon"].min().item()
lon_max = orig_data["lon"].max().item()

for solar_file in solar_files:
    print(f"Processing {solar_file}...")
    base_name = os.path.basename(solar_file)
    parts = base_name.split("_")
    model_name = parts[2]

    # Match corresponding wind file
    key_word = "*sfcWind*" + model_name + "*EUROPE*.nc"
    wind_file = glob.glob(os.path.join(data_folder, key_word))

    # Load and subset data
    solar_data = xr.open_dataset(solar_file).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
    wind_data = xr.open_dataset(wind_file[0]).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )

    # Map months to seasons
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
    season_labels = months.to_series().map(seasons).values

    # Spatial mean
    rsds_mean = solar_data["rsds"].mean(dim=("lon", "lat"))
    wind_mean = wind_data["sfcWind"].mean(dim=("lon", "lat"))

    # 20th percentile threshold
    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    # Binary low-value indicator
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # Assign season coordinates
    rsds_bin = rsds_bin.assign_coords(season=("time", season_labels))
    wind_bin = wind_bin.assign_coords(season=("time", season_labels))

    # Group by season and sum over time
    rsds_season_counts = rsds_bin.groupby("season").sum(dim="time").compute().values
    wind_season_counts = wind_bin.groupby("season").sum(dim="time").compute().values

    # Save separately
    np.save(f"../plotting_data/GCM/90-99/solar/{model_name}.npy", rsds_season_counts)
    np.save(f"../plotting_data/GCM/90-99/wind/{model_name}.npy", wind_season_counts)

    gc.collect()
