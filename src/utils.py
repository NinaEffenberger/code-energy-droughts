import os 
import xarray as xr
import numpy as np


def get_season_labels(file_path, folder):
    parts = file_path.split('_')
    model = '_'.join(parts[4:5])  # 4th part
    found_file = None
    for f in os.listdir(folder):
        if model in f:
            found_file = os.path.join(folder, f)
            orig_data = xr.open_dataset(found_file, decode_times=True)
            # Convert to pandas index
            months = orig_data.time.dt.month

            seasons = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Autumn", 10: "Autumn", 11: "Autumn"
            }
            
            # Convert to pandas Series to map
            season_labels = months.to_series().map(seasons).values
            return season_labels
            break