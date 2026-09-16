import concurrent.futures
import gc
import glob
import importlib
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops as sops
import shapely.vectorized
import xarray as xr
import utils

importlib.reload(utils)

# Paths
data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/benchmarks/corrdiff/"
file_pattern = "*id-011_sample-10*2090-2099*.nc"
time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/interpolation"
output_dir = "../plotting_data/land_only/corrdiff/90-99"
os.makedirs(output_dir, exist_ok=True)

# Reference Grid Coordinates (Original / Panel 1)
grid_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"
orig_data = xr.open_dataset(grid_path)
lons = orig_data.lon.values
lats = orig_data.lat.values

files = sorted(glob.glob(os.path.join(data_folder, file_pattern)))
print(f"Found {len(files)} files.")


def sanitize_dataset(ds):
    """Strips object/string coordinates that trigger pyarrow indexer bugs during .isel()."""
    drop_coords = [
        c
        for c in ds.coords
        if c not in ds.dims and np.issubdtype(ds[c].dtype, np.dtype("O"))
    ]
    return ds.drop_vars(drop_coords) if drop_coords else ds


def build_cartopy_land_mask_from_coords(lats, lons, target_da):
    """Generates a boolean DataArray (True = Land) using Cartopy and explicit coordinate arrays."""
    if lats.ndim == 1 and lons.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lon_grid, lat_grid = lons, lats

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
        k: v for k, v in target_da.coords.items() if k in target_da.dims and k != "time"
    }
    spatial_dims = [
        d for d in target_da.dims if d in ["y", "x", "lat", "lon", "rlat", "rlon"]
    ]

    return xr.DataArray(is_land, coords=spatial_coords, dims=spatial_dims)


def plot_sanity_check(
    da_uncropped,
    da_masked,
    mask,
    lats,
    lons,
    save_path="corrdiff_land_mask_sanity_check.png",
):
    """Plots original field, generated mask, and masked field side-by-side."""
    print(f"Saving sanity check plot to {save_path}...")
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    plot_da = da_uncropped.copy()
    plot_masked = da_masked.copy()
    plot_mask = mask.copy()

    if "lat" not in plot_da.coords:
        plot_da = plot_da.assign_coords(lat=(("y", "x"), lats), lon=(("y", "x"), lons))
        plot_masked = plot_masked.assign_coords(
            lat=(("y", "x"), lats), lon=(("y", "x"), lons)
        )
        plot_mask = plot_mask.assign_coords(
            lat=(("y", "x"), lats), lon=(("y", "x"), lons)
        )

    plot_kwargs = {"x": "lon", "y": "lat"}

    # 1. Original (Vertically Flipped Array Values)
    ax = axes[0]
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    plot_da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs={"shrink": 0.7},
        **plot_kwargs,
    )
    ax.set_title("1. Vertically Flipped Field (data[::-1, :])")

    # 2. Mask
    ax = axes[1]
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    plot_mask.astype(int).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Blues",
        cbar_kwargs={"shrink": 0.7},
        **plot_kwargs,
    )
    ax.set_title("2. Cartopy Land Mask")

    # 3. Masked Field
    ax = axes[2]
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    plot_masked.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs={"shrink": 0.7},
        **plot_kwargs,
    )
    ax.set_title("3. Land-Only Masked Field")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# Pre-compute the land mask once using the first file sample
first_ds = sanitize_dataset(xr.open_dataset(files[0], group="prediction"))
first_ds_flipped = first_ds.isel(y=slice(None, None, -1))
first_file_sample = first_ds_flipped["rsds"]
cartopy_land_mask = build_cartopy_land_mask_from_coords(lats, lons, first_file_sample)

# Perform sanity check plot on the first file (sequential before parallel loop)
sample_data_pred = sanitize_dataset(xr.open_dataset(files[0], group="prediction")).isel(
    y=slice(None, None, -1)
)
sample_solar_land = sample_data_pred["rsds"].where(cartopy_land_mask)
sample_raw = sample_data_pred["rsds"].isel(time=0).compute()
if "ensemble" in sample_raw.dims:
    sample_raw = sample_raw.isel(ensemble=0)
sample_masked = sample_solar_land.isel(time=0).compute()
if "ensemble" in sample_masked.dims:
    sample_masked = sample_masked.isel(ensemble=0)
plot_sanity_check(sample_raw, sample_masked, cartopy_land_mask, lats, lons)


def process_single_file(file_path):
    """Worker function to process a single file independently."""
    try:
        print(f"Processing {file_path}...")
        data_pred = sanitize_dataset(
            xr.open_dataset(
                file_path,
                group="prediction",
                chunks={"ensemble": 2, "time": 365, "y": 128, "x": 128},
            )
        )

        # --- TRUE VERTICAL FLIP (PANEL 1 / A) ---
        data_pred = data_pred.isel(y=slice(None, None, -1))

        season_labels = np.array(
            utils.get_season_labels(file_path, time_folder), dtype=str
        )

        # Apply land mask
        solar_land = data_pred["rsds"].where(cartopy_land_mask)
        wind_land = data_pred["sfcWind"].where(cartopy_land_mask)

        # 1. Spatial mean over LAND ONLY
        spatial_dims = [
            d for d in solar_land.dims if d in ["y", "x", "lat", "lon", "rlat", "rlon"]
        ]
        rsds_mean = solar_land.mean(dim=spatial_dims, skipna=True).reset_coords(
            drop=True
        )
        wind_mean = wind_land.mean(dim=spatial_dims, skipna=True).reset_coords(
            drop=True
        )

        # 2. Compute 20th percentile threshold per ensemble
        rsds_thresh = rsds_mean.quantile(0.2, dim="time")
        wind_thresh = wind_mean.quantile(0.2, dim="time")

        # 3. Binary time series: 1 if below threshold, else 0
        rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
        wind_bin = (wind_mean <= wind_thresh).astype("int8")

        # 4. Combine (both low → 1, else 0)
        combined = (rsds_bin & wind_bin).astype("int8")

        combined = combined.assign_coords(
            season=("time", season_labels[0 : combined.shape[1]])
        )

        # Group by season and sum over time
        season_counts = combined.groupby("season").sum(dim="time")

        filename = file_path.split("/")[-1].rsplit(".", 1)[0]
        season_counts_np = season_counts.compute().values

        print(f"Results for {filename}: {season_counts_np}")
        np.save(os.path.join(output_dir, filename + ".npy"), season_counts_np)
        gc.collect()
        return filename, "Success"
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, str(e)


# Run parallel execution using 8 worker processes
max_workers = 8
print(f"Starting parallel processing with {max_workers} workers...")

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_single_file, file_path) for file_path in files]
    for future in concurrent.futures.as_completed(futures):
        res_file, status = future.result()
        print(f"Finished {res_file}: {status}")

print("Processing complete for prediction data.")
