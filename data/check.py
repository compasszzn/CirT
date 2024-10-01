import os
from datetime import datetime, timedelta
import pandas as pd

def main():
    # 定义开始日期和结束日期
    start_date = datetime(1979, 1, 1)
    end_date = datetime(2022, 12, 31)

    # 生成日期范围
    date_range = pd.date_range(start_date, end_date)

    # 指定数据文件夹
    data_dir = "/data/zzn/S2S/single_level_merge"

    # 遍历所有日期并检查文件是否存在
    missing_files = []
    for single_date in date_range:
        yy = single_date.year
        mm = single_date.month
        dd = single_date.day

        # 定义文件名
        file_name = f"era5_single_full_0.5deg_{yy:04d}{mm:02d}{dd:02d}.zarr"

        # 检查文件是否存在
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    # 输出缺失文件
    if missing_files:
        print("以下文件缺失:")
        for file in missing_files:
            print(file)
    else:
        print("所有文件齐全.")

if __name__ == "__main__":
    main()