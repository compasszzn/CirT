import os
import xarray as xr
import pandas as pd
from datetime import datetime

# 生成日期范围
start_date = datetime(1979, 1, 1)
end_date = datetime(2022, 12, 31)
date_range = pd.date_range(start_date, end_date)

# 指定数据文件夹
data_dir_level = "/data/zzn/S2S/single_level"
data_dir_other = "/data/zzn/S2S/single_level_other"
output_dir_merge = "/data/zzn/S2S/single_level_merge"

# 确保输出目录存在
os.makedirs(output_dir_merge, exist_ok=True)

# 遍历所有日期并检查文件是否存在
missing_files = []
for single_date in date_range:
    date_str = single_date.strftime('%Y%m%d')
    file_level = os.path.join(data_dir_level, f"era5_single_full_0.5deg_{date_str}.zarr")
    file_other = os.path.join(data_dir_other, f"era5_single_full_0.5deg_{date_str}.zarr")
    file_output = os.path.join(output_dir_merge, f"era5_single_full_0.5deg_{date_str}.zarr")

    if os.path.exists(file_level) and os.path.exists(file_other):
        # 打开两个 Zarr 文件
        ds_level = xr.open_zarr(file_level)
        ds_other = xr.open_zarr(file_other)

        # 使用 xarray 的 merge 功能将两个数据集合并
        merged_ds = xr.merge([ds_level, ds_other])

        # 保存合并后的数据集为 Zarr 文件
        merged_ds.to_zarr(file_output)
    else:
        missing_files.append(date_str)

# 打印缺失的文件
if missing_files:
    print("Missing files for the following dates:")
    for missing in missing_files:
        print(missing)
else:
    print("All files processed successfully.")
