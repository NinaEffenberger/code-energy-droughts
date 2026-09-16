import gc
import glob
import os
import numpy as np
import xarray as xr

# 1. Setup Input / Output Paths for 2030-2039 (Interpolation Folder)
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/interpolation"
output_dir = "../plotting_data/land_only/GCM/30-39/"
os.makedirs(output_dir, exist_ok=True)

# Find solar files matching 2030-2039 pattern (falls back to general pattern if decade isn't in name)
solar_files = sorted(glob.glob(os.path.join(data_folder, "*rsds*EUROPE*2030-2039*.nc")))
if not solar_files:
    solar_files = sorted(glob.glob(os.path.join(data_folder, "*rsds*EUROPE*.nc")))

# Exclude CanESM2 file
solar_files = [f for f in solar_files if "CanESM2" not in f]

print(f"Found {len(solar_files)} files to process (excluding CanESM2).")

# Grid reference path for domain bounds & land mask
grid_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"

# Open reference grid 
with xr.open_dataset(grid_path) as orig_data:
    lat_min = float(orig_data["lat"].min())
    lat_max = float(orig_data["lat"].max())
    lon_min = float(orig_data["lon"].min())
    lon_max = float(orig_data["lon"].max())

    if "sfcLSM" in orig_data:
        land_mask = (orig_data["sfcLSM"] > 0.5).compute()
    elif "landmask" in orig_data:
        land_mask = (orig_data["landmask"] > 0.5).compute()
    else:
        land_mask = orig_data["tas"].isel(time=0).notnull().compute()

seasons_map = {
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


def process_variable_file(
    file_path, var_name, lat_min, lat_max, lon_min, lon_max, land_mask
):
    """Open netCDF with Dask lazy loading, crop, apply land mask, and compute spatial mean."""
    ds = xr.open_dataset(file_path, chunks={"time": 365})
    da = ds[var_name]

    # Crop to bounding box
    if da.lat.ndim == 1 and da.lon.ndim == 1:
        da_cropped = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    else:
        mask_bbox = (
            (da.lat >= lat_min)
            & (da.lat <= lat_max)
            & (da.lon >= lon_min)
            & (da.lon <= lon_max)
        )
        da_cropped = da.where(mask_bbox, drop=True)

    # Reindex mask to fit the cropped dataset
    mask_aligned = land_mask.reindex_like(da_cropped, method="nearest")

    # Mask ocean points
    da_land = da_cropped.where(mask_aligned)

    # Detect spatial dimensions 
    spatial_dims = [
        d for d in da_land.dims if d in ["lat", "lon", "rlat", "rlon", "x", "y"]
    ]

    # Compute 1D time series mean immediately to save memory
    spatial_mean = da_land.mean(dim=spatial_dims, skipna=True).compute()

    ds.close()
    return spatial_mean


for solar_file in solar_files:
    print(f"Processing {os.path.basename(solar_file)}...")
    base_name = os.path.basename(solar_file)
    parts = base_name.split("_")
    model_name = parts[2]

    wind_pattern = os.path.join(data_folder, f"*sfcWind*{model_name}*EUROPE*.nc")
    wind_files = glob.glob(wind_pattern)

    if not wind_files:
        print(f"Warning: No matching wind file found for {model_name}. Skipping...")
        continue

    # 1. Spatial mean over LAND ONLY
    rsds_mean = process_variable_file(
        solar_file, "rsds", lat_min, lat_max, lon_min, lon_max, land_mask
    )
    wind_mean = process_variable_file(
        wind_files[0], "sfcWind", lat_min, lat_max, lon_min, lon_max, land_mask
    )

    # 2. Compute 20th percentile threshold on 1D series
    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    # 3. Binary time series
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # 4. Combine
    combined = (rsds_bin & wind_bin).astype("int8")

    # Map seasons
    season_labels = [seasons_map[m] for m in combined.time.dt.month.values]
    combined = combined.assign_coords(season=("time", season_labels))

    # 5. Group and sum over seasons
    season_counts = combined.groupby("season").sum(dim="time")
    season_counts_np = season_counts.values

    print(f"Results for {model_name} (30-39 Interpolation): {season_counts_np}")

    # Save output
    save_path = os.path.join(output_dir, f"{model_name}.npy")
    np.save(save_path, season_counts_np)

    # Clean up memory
    del rsds_mean, wind_mean, rsds_bin, wind_bin, combined, season_counts
    gc.collect()

print("Processing complete for 30-39 interpolation data.")
