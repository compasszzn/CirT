import os
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import argparse
import config
# 基本文件路径
def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    a = xr.open_zarr(input_dir+f'/era5_{args.data_type}_full_0.5deg_19790101.zarr')

    # 定义开始日期和结束日期
    start_date = datetime(1979, 1, 1)
    # end_date = datetime(2022, 12, 31)
    end_date = datetime(2022, 12, 31)

    # 生成日期范围
    date_range = pd.date_range(start_date, end_date)

    # 选择保存数据的输出目录


        # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lat_idx = np.arange(0, len(a.latitude), args.sample)
    lon_idx = np.arange(0, len(a.longitude), args.sample)
    for single_date in date_range:
        # 将日期转换为字符串格式，适用于 xarray 的时间选择器
        date_str = single_date.strftime('%Y-%m-%d')

        # 根据日期选择数
        yy, mm, dd = date_str.split('-')

        # daily_data=daily_data.isel(latitude=lat_idx, longitude=lon_idx)
        # # 定义输出文件路径
        input_daily_file = f"{input_dir}/era5_{args.data_type}_full_0.5deg_{yy}{mm}{dd}.zarr"
        output_daily_file = f"{output_dir}/era5_{args.data_type}_full_1.5deg_{yy}{mm}{dd}.zarr"
        daily_data = xr.open_zarr(input_daily_file)
        daily_data=daily_data.isel(latitude=lat_idx, longitude=lon_idx)
        # 将一天的数据保存为 Zarr 文件
        daily_data.to_zarr(output_daily_file)

        print(f"Saved data for {date_str} to {output_daily_file}")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process and save annual SST data.')
    parser.add_argument('--input_dir', type=str, default='/data/zzn/S2S/single_level_merge')
    parser.add_argument('--output_dir', type=str, default='/data/zzn/S2S/single_level_1.5')
    parser.add_argument('--data_type', type=str, default='single')
    parser.add_argument('--sample', type=int, default=3)
    args = parser.parse_args()
    main(args)
