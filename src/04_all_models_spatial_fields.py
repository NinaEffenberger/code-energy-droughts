import xarray as xr
import glob
import os
import gc


# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
data_folder_corrdiff = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/benchmarks/corrdiff/"
file_pattern = "*id-011*2090-2099*.nc"

extrap_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"
output_folder = "../plotting_data/results_matrices/"
os.makedirs(output_folder, exist_ok=True)

grid_path = os.path.join(
    extrap_folder,
    "tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc",
)
orig_data = xr.open_dataset(grid_path)
lons = orig_data.lon.values
lats = orig_data.lat.values

lat_min, lat_max = float(orig_data.lat.min()), float(orig_data.lat.max())
lon_min, lon_max = float(orig_data.lon.min()), float(orig_data.lon.max())

files = glob.glob(os.path.join(data_folder_corrdiff, file_pattern))
solar_files = glob.glob(os.path.join(extrap_folder, "*rsds*EUROPE*.nc"))


# --------------------------------------------------------------------
# PROCESS LOOP
# --------------------------------------------------------------------
def compute_metrics(solar, wind, time_dim):
    """Compute thresholds, combined drought frequency, and below-threshold means."""
    rsds_thresh = solar["rsds"].quantile(0.2, dim=time_dim)
    wind_thresh = wind["sfcWind"].quantile(0.2, dim=time_dim)
    combined = (
        (solar["rsds"] <= rsds_thresh) & (wind["sfcWind"] <= wind_thresh)
    ).astype("float32")
    return {
        "combined_mean": combined.mean(dim=time_dim),
        "solar_below_mean": solar["rsds"]
        .where(solar["rsds"] <= rsds_thresh)
        .mean(dim=time_dim),
        "wind_below_mean": wind["sfcWind"]
        .where(wind["sfcWind"] <= wind_thresh)
        .mean(dim=time_dim),
    }


for file_path in files:
    base = os.path.basename(file_path).replace(".nc", "")
    print(f"Processing {base}")

    # --- Identify model and matching files ---
    model_name = base.split("_")[4]
    print(model_name)
    solar_file = [f for f in solar_files if model_name in f][0]
    wind_file = glob.glob(
        os.path.join(extrap_folder, f"*sfcWind*{model_name}*EUROPE*.nc")
    )[0]

    # --- Load GCM data ---
    solar_data = xr.open_dataset(solar_file).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
    wind_data = xr.open_dataset(wind_file).sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )

    # --- Load RCM data ---
    data_pred = xr.open_dataset(
        file_path,
        group="prediction",
        chunks={"ensemble": 2, "time": 365, "y": 128, "x": 128},
    )
    data_truth = xr.open_dataset(file_path, group="truth")

    # Flip y dimension to match grid
    data_pred = data_pred.isel(y=slice(None, None, -1))
    data_truth = data_truth.isel(y=slice(None, None, -1))

    # --- Compute metrics ---
    gcm_metrics = compute_metrics(solar_data, wind_data, time_dim="time")
    emul_metrics = compute_metrics(data_pred, data_pred, time_dim=("ensemble", "time"))
    truth_metrics = compute_metrics(data_truth, data_truth, time_dim="time")

    # --- Combine and save ---
    ds_out = xr.Dataset(
        {
            # GCM
            "combined_GCM": gcm_metrics["combined_mean"],
            "solar_GCM": gcm_metrics["solar_below_mean"],
            "wind_GCM": gcm_metrics["wind_below_mean"],
            # RCM emulated
            "combined_RCM_emulated": emul_metrics["combined_mean"],
            "solar_RCM_emulated": emul_metrics["solar_below_mean"],
            "wind_RCM_emulated": emul_metrics["wind_below_mean"],
            # RCM truth
            "combined_RCM_truth": truth_metrics["combined_mean"],
            "solar_RCM_truth": truth_metrics["solar_below_mean"],
            "wind_RCM_truth": truth_metrics["wind_below_mean"],
        }
    )

    out_path = os.path.join(output_folder, f"{base}_drought_metrics.nc")
    ds_out.to_netcdf(out_path)
    ds_out.close()
    solar_data.close()
    wind_data.close()
    data_pred.close()
    data_truth.close()
    gc.collect()

    print(f"Saved: {out_path}")

print("All models processed and saved.")
