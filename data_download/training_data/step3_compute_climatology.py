import xarray as xr
from tqdm import tqdm
from pathlib import Path
import argparse
import numpy as np
import config
import re

import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")
    

def main(args):
    """
    Main driver to compute climatology of individual param/level pair
    Usage example: 
        (1) Scalar climatology for training  : `python compute_climatology.py --dataset_name era5 --is_spatial 0`
        (2) Proper DOY-spatial climatology   : `python compute_climatology.py --dataset_name era5 --is_spatial 1`
    """
    if args.dataset_name=='pressure_level_1.5':
        data_dir = Path(config.DATA_DIR) / 'pressure_level_1.5'
    elif args.dataset_name=='single_level_1.5':
        data_dir = Path(config.DATA_DIR) / 'single_level_1.5'
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 'climatology_1.5'
    output_dir.mkdir(parents=True, exist_ok=True)
    

    print('Computing climatology including spatial domain...')
    
    output_file = output_dir / f'climatology_{args.dataset_name}_new.zarr'
    
    ## Collect all files
    dataset_files = list(data_dir.glob('*.zarr'))
    dataset_files.sort()
    # Collect values
    
    climatology_mean = list()
    climatology_sigma = list()

    if args.dataset_name== 'single_level_1.5':
        PARAMS = config.ERA5_SINGLE_LEVEL
    elif args.dataset_name== 'pressure_level_1.5':
        PARAMS = config.ERA5_PRESSURE_LEVEL
    for params in PARAMS:
        all_vars = list()
        for dataset_file in tqdm(dataset_files):
            
            ds = xr.open_dataset(dataset_file, engine='zarr')
            
            # Handle datasets without pressure levels (eg. ocean + land reanalysis)

            ds = ds[[params]]

                
            ds = ds.to_array().values#(6, 10, 121, 240)shape
            all_vars.append(ds.reshape(-1, ds.shape[-2], ds.shape[-1]))
        
    
        all_vars = np.array(all_vars) # Shape (w/ levels): (time, param*level, lat, lon); (surface): (time, param, lat, lon)
        
        # Aggregation
        climatology_mean.extend(np.nanmean(all_vars, axis=(0,2,3)))
        climatology_sigma.extend(np.nanstd(all_vars, axis=(0,2,3)))
        
    if args.dataset_name == 'single_level_1.5':
        param_string = PARAMS
    elif args.dataset_name == 'pressure_level_1.5':
        param_string = [f"{param}-{level}" for param in config.ERA5_PRESSURE_LEVEL for level in config.PRESSURE_LEVELS]

    climatology_mean = np.array(climatology_mean)
    climatology_sigma = np.array(climatology_sigma)
    # Climatology dataset construction
    ds = xr.Dataset(
        {
            'mean': (('param'), climatology_mean),
            'sigma': (('param'), climatology_sigma),
        },

        coords = {
            'param': param_string,
        }
    )

    # Save climatology
    ds.to_zarr(output_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',default='pressure_level_1.5', choices=['single_level_1.5','pressure_level_1.5'])

    args = parser.parse_args()
    main(args)
