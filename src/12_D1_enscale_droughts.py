import xarray as xr
import glob
import os
import gc
import torch

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
data_folder_corrdiff = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/samples_multivariate/maybritt_nicolai_zeros-constant_dec-1e-3_onehot/"
file_pattern = "idx*_inter.pt"

extrap_folder = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation"
output_folder = "../plotting_data/results_matrices/enscale/"
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
    base = os.path.basename(file_path).replace(".pt", "")
    print(f"Processing {base}")

    # --- Identify model and matching files ---
    data = torch.load(file_path)
    gen_data = {}
    variables = ["sfcWind", "rsds"]
    var_order = ["tas", "pr", "sfcWind", "rsds"]
    for var in variables:
        data_var = data[:, var_order.index(var), :, :]
        data_var = torch.flip(data_var, dims=[1])
        gen_data[var] = data_var.view(-1, 128, 128, 9).numpy()
    dims = ("time", "lat", "lon", "ensemble")
    gen_data = {
        "rsds": xr.DataArray(gen_data["rsds"], dims=dims),
        "sfcWind": xr.DataArray(gen_data["sfcWind"], dims=dims),
    }
    emul_metrics = compute_metrics(gen_data, gen_data, time_dim=("ensemble", "time"))

    # --- Combine and save ---
    ds_out = xr.Dataset(
        {
            # RCM emulated
            "combined_RCM_emulated": emul_metrics["combined_mean"],
            "solar_RCM_emulated": emul_metrics["solar_below_mean"],
            "wind_RCM_emulated": emul_metrics["wind_below_mean"],
        }
    )

    out_path = os.path.join(output_folder, f"{base}_drought_metrics.nc")
    ds_out.to_netcdf(out_path)
    ds_out.close()
    gc.collect()

    print(f"Saved: {out_path}")

print("All models processed and saved.")
