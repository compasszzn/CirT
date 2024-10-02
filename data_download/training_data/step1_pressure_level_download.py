import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import argparse
import config
# 基本文件路径
def main(args):

    a = xr.open_zarr('gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr')
 
    #Beginning and Ending day
    start_date = datetime(1979, 1, 1)
    end_date = datetime(2018, 12, 31)


    date_range = pd.date_range(start_date, end_date)


    output_dir = f"{config.DATA_DIR}/pressure_level_1.5"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #process 0.25 to 1.5
    lat_idx = np.arange(0, len(a.latitude), 6)
    lon_idx = np.arange(0, len(a.longitude), 6)
    for single_date in date_range:

        date_str = single_date.strftime('%Y-%m-%d')


        daily_data = a.sel(time=date_str)[config.ERA5_PRESSURE_LEVEL]
        yy, mm, dd = daily_data.time.dt.strftime('%Y-%m-%d').item().split('-')
        pressure_levels_indices = np.where(np.isin(a.level.values, config.PRESSURE_LEVELS))[0]
        daily_data = daily_data.isel(level=pressure_levels_indices)
        daily_data=daily_data.isel(latitude=lat_idx, longitude=lon_idx)

        output_daily_file = f"{output_dir}/era5_pressure_full_1.5deg_{yy}{mm}{dd}.zarr"
        daily_data = daily_data.fillna(0)
 
        daily_data.to_zarr(output_daily_file)

        print(f"Saved data for {date_str} to {output_daily_file}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process and save annual SST data.')
    args = parser.parse_args()
    main(args)
