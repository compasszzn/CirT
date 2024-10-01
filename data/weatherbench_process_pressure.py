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
 
    # 定义开始日期和结束日期
    start_date = datetime(2019, 1, 1)
    # end_date = datetime(2022, 12, 31)
    end_date = datetime(2022, 12, 31)

    # 生成日期范围
    date_range = pd.date_range(start_date, end_date)

    # 选择保存数据的输出目录
    output_dir = "/data/zzn/S2S/pressure_level"  # 替换为实际输出目录
        # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lat_idx = np.arange(0, len(a.latitude), 2)
    lon_idx = np.arange(0, len(a.longitude), 2)
    for single_date in date_range:
        # 将日期转换为字符串格式，适用于 xarray 的时间选择器
        date_str = single_date.strftime('%Y-%m-%d')

        # 根据日期选择数据
        daily_data = a.sel(time=date_str)[config.ERA5_PRESSURE_LEVEL]
        yy, mm, dd = daily_data.time.dt.strftime('%Y-%m-%d').item().split('-')
        pressure_levels_indices = np.where(np.isin(a.level.values, config.PRESSURE_LEVELS))[0]
        daily_data = daily_data.isel(level=pressure_levels_indices)
        daily_data=daily_data.isel(latitude=lat_idx, longitude=lon_idx)
        # 定义输出文件路径
        output_daily_file = f"{output_dir}/era5_pressure_full_0.5deg_{yy}{mm}{dd}.zarr"
        daily_data = daily_data.fillna(0)
        # 将一天的数据保存为 Zarr 文件
        daily_data.to_zarr(output_daily_file)

        print(f"Saved data for {date_str} to {output_daily_file}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process and save annual SST data.')
    parser.add_argument('--name', type=str, help='Name of the variable to process')
    args = parser.parse_args()
    main(args)
