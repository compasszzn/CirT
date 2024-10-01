import xarray as xr
from pathlib import Path
import config

import logging
logging.basicConfig(level=logging.INFO)

import cdsapi

def main():
    """
    Main driver to download ERA5 data based on individual variable
    """
    # Initialize CDS API
    c = cdsapi.Client()
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 'era5_pressure'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the corresponding data based on year/month
    for year in config.ERA5_YEARS:
        
        for month in config.MONTHS:

            for day in config.DAYS:
            
                logging.info(f'Downloading {year}/{month}/{day}...')
                
                output_file = output_dir / f'era5_pressure_full_0.5deg_{year}{month}{day}.nc'
                    
                try:
                    c.retrieve(
                        'reanalysis-era5-pressure-levels',
                        {
                            'product_type': 'reanalysis',
                            'variable': config.ERA5_PRESSURE_LIST,
                            'pressure_level': config.PRESSURE_LEVELS,
                            'year': year,
                            'month': month,
                            'day': day,
                            'time': config.TIMES,
                            'grid': ['0.5', '0.5'],
                            'format': 'netcdf',
                        },
                        output_file)
                except Exception as e:
                    break
                
                # Break down into daily .zarr (cloud-optimized)
                ds = xr.open_dataset(output_file)
                ds['z'] = ds['z'] / config.G_CONSTANT ## Convert to gpm

                ds_mean = ds.mean(dim='time')
                yy, mm, dd = ds.time[0].dt.strftime('%Y-%m-%d').item().split('-')
                output_daily_file = output_dir / f'era5_pressure_full_0.5deg_{yy}{mm}{dd}.zarr'
                ds_mean.to_zarr(output_daily_file)

                output_file.unlink()

if __name__ == "__main__":
    main()

