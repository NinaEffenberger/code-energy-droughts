import numpy as np
import xarray as xr
import torch
import os

start_index = 0
end_index = 7

for index in range(start_index, end_index):
    file_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/interpolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2030-2039.nc"
    pt_file_path = f"/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/samples_multivariate/maybritt_nicolai_zeros-constant_dec-1e-3_onehot_cmip6/idx{index}_inter.pt"

    # Check if the .pt file exists before processing
    if not os.path.exists(pt_file_path):
        print(f"Skipping index {index}: File not found.")
        continue

    data = torch.load(pt_file_path)

    gen_data = {}
    variables = ["sfcWind", "rsds"]
    var_order = ["tas", "pr", "sfcWind", "rsds"]
    for var in variables:
        data_var = data[:, var_order.index(var), :, :]
        gen_data[var] = data_var.view(-1, 128, 128, 9).numpy()

    # 1. Spatial mean for each ensemble & time
    rsds_mean = gen_data["rsds"].mean(axis=(1, 2))
    wind_mean = gen_data["sfcWind"].mean(axis=(1, 2))

    # 2. Compute 20th percentile threshold per ensemble
    rsds_thresh = np.quantile(rsds_mean, 0.2, axis=0)
    wind_thresh = np.quantile(wind_mean, 0.2, axis=0)

    # 3. Binary time series: 1 if below threshold, else 0
    rsds_bin = (rsds_mean <= rsds_thresh).astype("int8")
    wind_bin = (wind_mean <= wind_thresh).astype("int8")

    # 4. Combine 
    combined = (rsds_bin & wind_bin).astype("int8")

    orig_data = xr.open_dataset(file_path, decode_times=True)
    months = orig_data.time.dt.month

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
    season_labels = np.array(season_labels)
    unique_seasons = np.unique(season_labels)
    n_samples = combined.shape[1]
    n_seasons = len(unique_seasons)

    season_counts_np = np.zeros((n_samples, n_seasons), dtype=np.int32)

    for i, season in enumerate(unique_seasons):
        mask = season_labels == season
        mask = np.asarray(mask).ravel()
        season_counts_np[:, i] = combined[mask, :].sum(axis=0)

    # Save the output for the current index
    save_path = f"..plotting_data/enscale/interp/cmip6/{index}.npy"
    np.save(save_path, season_counts_np)
    print(f"Successfully saved index {index} to {save_path}")

print("Loop finished.")
