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

data_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/benchmarks/corrdiff/"
file_pattern = "*id-011*2090-2099*.nc"

grid_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"
orig_data = xr.open_dataset(grid_path)

lons = orig_data.lon.values
lats = orig_data.lat.values

files = glob.glob(os.path.join(data_folder, file_pattern))
print(f"Found {len(files)} files.")
time_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/"

output_dir = "../plotting_data/land_only/RCM/90-99/"
os.makedirs(output_dir, exist_ok=True)


def sanitize_dataset(ds):
    """Strips object/string coordinates that trigger pyarrow indexer bugs during .isel()."""
    drop_coords = [
        c
        for c in ds.coords
        if c not in ds.dims and np.issubdtype(ds[c].dtype, np.dtype("O"))
    ]
    return ds.drop_vars(drop_coords) if drop_coords else ds


try:
    land_shape = sops.unary_union(list(cfeature.LAND.with_scale("10m").geometries()))
except Exception:
    land_shape = sops.unary_union(list(cfeature.GND_100.geometries()))

is_land = shapely.vectorized.contains(land_shape, lons, lats)

first_file_sample = sanitize_dataset(xr.open_dataset(files[0], group="truth"))["rsds"]
spatial_coords = {
    k: v
    for k, v in first_file_sample.coords.items()
    if k in first_file_sample.dims and k != "time"
}
spatial_dims = [
    d for d in first_file_sample.dims if d in ["y", "x", "lat", "lon", "rlat", "rlon"]
]

cartopy_land_mask = xr.DataArray(is_land, coords=spatial_coords, dims=spatial_dims)


def plot_sanity_check(
    da_uncropped,
    da_masked,
    mask,
    lats,
    lons,
    save_path="rcm_land_mask_sanity_check.png",
):
    """Plots original field, generated mask, and masked field side-by-side."""
    print(f"Saving sanity check plot to {save_path}...")
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    plot_da = da_uncropped.assign_coords(lat=(("y", "x"), lats), lon=(("y", "x"), lons))
    plot_masked = da_masked.assign_coords(
        lat=(("y", "x"), lats), lon=(("y", "x"), lons)
    )
    plot_mask = mask.assign_coords(lat=(("y", "x"), lats), lon=(("y", "x"), lons))

    plot_kwargs = {"x": "lon", "y": "lat"}

    # 1. Original
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
    ax.set_title("1. Original Raw Field")

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

    # 3. Masked
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


for i, file_path in enumerate(files):
    print(f"Processing {file_path}...")
    data_pred = sanitize_dataset(xr.open_dataset(file_path, group="truth"))
    season_labels = np.array(utils.get_season_labels(file_path, time_folder), dtype=str)

    solar_land = data_pred["rsds"].where(cartopy_land_mask)
    wind_land = data_pred["sfcWind"].where(cartopy_land_mask)

    # --- SANITY CHECK PLOT (First file only) ---
    if i == 0:
        sample_raw = data_pred["rsds"].isel(time=0).compute()
        if "ensemble" in sample_raw.dims:
            sample_raw = sample_raw.isel(ensemble=0)
        sample_masked = solar_land.isel(time=0).compute()
        if "ensemble" in sample_masked.dims:
            sample_masked = sample_masked.isel(ensemble=0)
        plot_sanity_check(sample_raw, sample_masked, cartopy_land_mask, lats, lons)

    spatial_dims = [
        d for d in solar_land.dims if d in ["y", "x", "lat", "lon", "rlat", "rlon"]
    ]
    rsds_mean = solar_land.mean(dim=spatial_dims, skipna=True).reset_coords(drop=True)
    wind_mean = wind_land.mean(dim=spatial_dims, skipna=True).reset_coords(drop=True)

    rsds_thresh = rsds_mean.quantile(0.2, dim="time")
    wind_thresh = wind_mean.quantile(0.2, dim="time")

    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    combined = (rsds_bin & wind_bin).astype("int8")
    combined = combined.assign_coords(season=("time", season_labels[0 : len(combined)]))

    season_counts = combined.groupby("season").sum(dim="time")

    filename = file_path.split("/")[-1].rsplit(".", 1)[0]
    season_counts_np = season_counts.compute().values

    print(f"Results for {filename}: {season_counts_np}")
    np.save(os.path.join(output_dir, filename + ".npy"), season_counts_np)
    gc.collect()

print("Processing complete.")
