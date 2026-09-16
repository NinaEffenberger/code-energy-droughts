import gc
import glob
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops as sops
import shapely.vectorized
import xarray as xr

# 1. Setup Input / Output Paths
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"
output_dir = "../plotting_data/land_only/GCM/90-99/"
os.makedirs(output_dir, exist_ok=True)

# Find all solar files
solar_files = sorted(glob.glob(os.path.join(data_folder, "*rsds*EUROPE*.nc")))

# EXCLUDE CanESM2 file
solar_files = [f for f in solar_files if "CanESM2" not in f]

print(f"Found {len(solar_files)} files to process (excluding CanESM2).")

grid_path = os.path.join(
    data_folder,
    "tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc",
)

# Open reference grid to establish bounding box
with xr.open_dataset(grid_path) as orig_data:
    lat_min = float(orig_data["lat"].min())
    lat_max = float(orig_data["lat"].max())
    lon_min = float(orig_data["lon"].min())
    lon_max = float(orig_data["lon"].max())

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


def get_cartopy_land_mask(da_cropped):
    """Generates a boolean 2D DataArray where True = Land using Cartopy shapefiles."""
    if da_cropped.lat.ndim == 1 and da_cropped.lon.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(da_cropped.lon.values, da_cropped.lat.values)
    else:
        lon_grid, lat_grid = da_cropped.lon.values, da_cropped.lat.values

    try:
        land_feature = cfeature.LAND.with_scale("10m")
        combined_land_shape = sops.unary_union(list(land_feature.geometries()))
    except Exception as e:
        print(
            f"Warning: Could not fetch Cartopy shapefiles online ({e}). Using coarse offline land features..."
        )
        land_feature = cfeature.GND_100
        combined_land_shape = sops.unary_union(list(land_feature.geometries()))

    is_land = shapely.vectorized.contains(combined_land_shape, lon_grid, lat_grid)

    spatial_coords = {
        k: v
        for k, v in da_cropped.coords.items()
        if k in da_cropped.dims and k != "time"
    }
    spatial_dims = [
        d for d in da_cropped.dims if d in ["lat", "lon", "rlat", "rlon", "x", "y"]
    ]

    return xr.DataArray(is_land, coords=spatial_coords, dims=spatial_dims)


def plot_sanity_check(
    da_uncropped, da_masked, mask, save_path="land_mask_sanity_check.png"
):
    """Plots original field, generated mask, and masked field side-by-side."""
    print(f"Saving sanity check plot to {save_path}...")
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # 1. Original Unmasked Data
    ax = axes[0]
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    da_uncropped.plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"shrink": 0.7}
    )
    ax.set_title("1. Original Raw Field (Full Domain)")

    # 2. Binary Land Mask
    ax = axes[1]
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    mask.astype(int).plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap="Blues", cbar_kwargs={"shrink": 0.7}
    )
    ax.set_title("2. Cartopy Land Mask (1=Land, 0=Ocean)")

    # 3. Masked Field (Land Only)
    ax = axes[2]
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    da_masked.plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap="viridis", cbar_kwargs={"shrink": 0.7}
    )
    ax.set_title("3. Land-Only Masked Field")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def process_variable_file(
    file_path, var_name, lat_min, lat_max, lon_min, lon_max, make_plot=False
):
    """Open netCDF, crop, apply Cartopy land mask, compute spatial mean, and optionally plot a sanity check."""
    ds = xr.open_dataset(file_path, chunks={"time": 365})
    da = ds[var_name]

    # Crop to bounding box lazily
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

    # Build geographic land mask using Cartopy
    cartopy_land_mask = get_cartopy_land_mask(da_cropped)

    # Mask out ocean points
    da_land = da_cropped.where(cartopy_land_mask)

    # --- SANITY CHECK PLOT ---
    if make_plot:
        # Extract 2D field at t=0 for plotting
        sample_raw = da_cropped.isel(time=0).compute()
        sample_masked = da_land.isel(time=0).compute()
        plot_sanity_check(sample_raw, sample_masked, cartopy_land_mask)

    # Detect spatial dimensions
    spatial_dims = [
        d for d in da_land.dims if d in ["lat", "lon", "rlat", "rlon", "x", "y"]
    ]

    # Compute 1D time series mean immediately over LAND ONLY
    spatial_mean = da_land.mean(dim=spatial_dims, skipna=True).compute()

    ds.close()
    return spatial_mean


# Run loop
for i, solar_file in enumerate(solar_files):
    print(f"Processing {os.path.basename(solar_file)}...")
    base_name = os.path.basename(solar_file)
    parts = base_name.split("_")
    model_name = parts[2]

    wind_pattern = os.path.join(data_folder, f"*sfcWind*{model_name}*EUROPE*.nc")
    wind_files = glob.glob(wind_pattern)

    if not wind_files:
        print(f"Warning: No matching wind file found for {model_name}. Skipping...")
        continue

    # Plot sanity check only on the first file (i == 0)
    plot_first_sample = i == 0

    rsds_mean = process_variable_file(
        solar_file,
        "rsds",
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        make_plot=plot_first_sample,
    )
    wind_mean = process_variable_file(
        wind_files[0], "sfcWind", lat_min, lat_max, lon_min, lon_max
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

    print(f"Results for {model_name}: {season_counts_np}")

    # Save output
    save_path = os.path.join(output_dir, f"{model_name}.npy")
    np.save(save_path, season_counts_np)

    del rsds_mean, wind_mean, rsds_bin, wind_bin, combined, season_counts
    gc.collect()

print("Processing complete.")
