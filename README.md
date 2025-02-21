# CirT: Global Subseaonal-To-Seasonal Forecasting with Geometry-inspired Transformer
This is the official Pytorch implementation of the paper: CirT: Global Subseaonal-to-Seasonal Forecasting with Geometry-inspired Transformer in ICLR 2025.
## Quickstart
Create environment and install dependencies
```
conda create -n cirt python==3.9
conda activate cirt
pip install -r requirements.txt
```
## Download and Process Data
The traning data is downloaded from [WeatherBench2](https://console.cloud.google.com/storage/browser/weatherbench2;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)
```
python data_download/training_data/step1_pressure_level_download.py

python data_download/training_data/step2_single_level_download.py

python data_download/training_data/step3_compute_climatology.py --dataset_name pressure_level_1.5

python data_download/training_data/step3_compute_climatology.py --dataset_name single_level_1.5
```

## Train CirT
```
Update `CIRT/configs/CirT.yaml` field: data_dir: <YOUR_DATA_DIR>
python train.py
```

## Citation
If you find any of the code useful, feel free to cite these works.
```
@inproceedings{
liu2025cirt,
title={CirT: Global Subseasonal-to-Seasonal Forecasting with Geometry-inspired Transformer},
author={Yang Liu and Zinan Zheng and Jiashun Cheng and Fugee Tsung and Deli Zhao and Yu Rong and Jia Li},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```
## Acknowledgement
We use the code from the repository [ChaosBench](https://github.com/leap-stc/ChaosBench)
