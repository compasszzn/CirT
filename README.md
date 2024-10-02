# create environment
```
conda create -n chaos python==3.9
conda activate chaos
pip install -r requirements.txt
```
# Download and Process Data
The traning data is downloaded from [WeatherBench2](https://console.cloud.google.com/storage/browser/weatherbench2;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)
```
python data_download/training_data/step1_pressure_level_download.py

python data_download/training_data/step2_single_level_download.py

python data_download/training_data/step3_compute_climatology.py --dataset_name pressure_level_1.5

python data_download/training_data/step3_compute_climatology.py --dataset_name single_level_1.5
```

# Train CirT
```
python train.py
```