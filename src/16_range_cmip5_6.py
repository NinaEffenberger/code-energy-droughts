import gc
import glob
import os
import pandas as pd
import xarray as xr

# --- Domain setup (from your script) ---
grid_path = "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation/tas_day_EUR-11_ALADIN63_CNRM-CM5_r1i1p1_rcp85_ALPS_cordexgrid_2090-2099.nc"
orig_data = xr.open_dataset(grid_path)
lat_min = orig_data["lat"].min().item()
lat_max = orig_data["lat"].max().item()
lon_min = orig_data["lon"].min().item()
lon_max = orig_data["lon"].max().item()

# Folders to compare
datasets = {
    "CMIP5": [
        "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/interpolation",
        "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/test/extrapolation",
    ],
    "CMIP6": [
        "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/cmip6/test/interpolation",
        "/cluster/work/math/climate-downscaling/cordex-data/cordex-ALPS-allyear/cmip6/test/extrapolation",
    ],
}

results = []

print("=" * 80)
print("PROCESSING FILES AND PRINTING INDIVIDUAL RANGES")
print("=" * 80)

for cmip_label, folder_list in datasets.items():
    for data_folder in folder_list:
        # Find GCM solar files 
        solar_files = glob.glob(os.path.join(data_folder, "rsds_day_*EUROPE*.nc"))

        for solar_file in solar_files:
            solar_filename = os.path.basename(solar_file)
            parts = solar_filename.split("_")
            model_name = parts[2]

            # EXCLUDE CanESM2
            if "CanESM2" in model_name:
                print(f"[{cmip_label}] Skipping excluded model: {model_name}")
                continue

            # Strict file pairing: replace variable name 
            wind_file = solar_file.replace("rsds_day_", "sfcWind_day_")

            # Fallback if filename structure differs 
            if not os.path.exists(wind_file):
                pattern = os.path.join(
                    data_folder, f"sfcWind_day_{model_name}_*EUROPE*.nc"
                )
                matches = glob.glob(pattern)
                if matches:
                    wind_file = matches[0]
                else:
                    print(f"Skipping {solar_filename}: Matching wind file not found.")
                    continue

            wind_filename = os.path.basename(wind_file)

            # Load and crop using domain bounds
            solar_ds = xr.open_dataset(solar_file).sel(
                lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
            )
            wind_ds = xr.open_dataset(wind_file).sel(
                lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
            )

            r_min = float(solar_ds["rsds"].min().compute())
            r_max = float(solar_ds["rsds"].max().compute())
            w_min = float(wind_ds["sfcWind"].min().compute())
            w_max = float(wind_ds["sfcWind"].max().compute())

            print(f"\n[{cmip_label}] Model: {model_name}")
            print(f"  Solar File: {solar_filename}")
            print(f"    -> rsds min: {r_min:.4f}, max: {r_max:.4f}")
            print(f"  Wind File:  {wind_filename}")
            print(f"    -> sfcWind min: {w_min:.4f}, max: {w_max:.4f}")

            results.append(
                {
                    "CMIP": cmip_label,
                    "Model": model_name,
                    "Subfolder": os.path.basename(data_folder),
                    "Solar_File": solar_filename,
                    "Wind_File": wind_filename,
                    "rsds_min": r_min,
                    "rsds_max": r_max,
                    "wind_min": w_min,
                    "wind_max": w_max,
                }
            )

            gc.collect()

# Convert to DataFrame
df_ranges = pd.DataFrame(results)

print("\n" + "=" * 80)
print("PER-FILE DETAILED TABLE (CanESM2 Excluded)")
print("=" * 80)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)
print(df_ranges.to_string(index=False))

print("\n" + "=" * 80)
print("AGGREGATED CMIP5 VS CMIP6 SUMMARY")
print("=" * 80)
summary = (
    df_ranges.groupby("CMIP")
    .agg(
        rsds_min=("rsds_min", "min"),
        rsds_max=("rsds_max", "max"),
        wind_min=("wind_min", "min"),
        wind_max=("wind_max", "max"),
        file_pairs_count=("Solar_File", "count"),
    )
    .reset_index()
)

print(summary.to_string(index=False))
