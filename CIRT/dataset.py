import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import glob
import xarray as xr
import numpy as np
from datetime import datetime
import re
from tqdm import tqdm
from CIRT import config
from torch_geometric.data import Data

class S2SDataset(Dataset):
    """
    Dataset object to handle input reanalysis.
    
    Params:
        years <List[int]>      : list of years to load and process,
        n_step <int>           : number of contiguous timesteps included in the data (default: 1)
        lead_time <int>        : delta_t ahead in time, useful for direct prediction (default: 1)
        single_vars <List[str]>  : list of land variables to include (default: empty)`
        ocean_vars <List[str]> : list of sea/ice variables to include (default: empty)`
        is_normalized <bool>   : flag to indicate whether we should perform normalization or not (default: True)
    """
    
    def __init__(
        self, 
        data_dir: str,
        years: List[int],
        n_step: int = 1,
        lead_time: int = 1,
        kernel_size: int = 4,
        single_vars: List[str] = [],
        pred_single_vars: List[str] = [],
        pred_pressure_vars: List[str] = [],
        is_normalized: bool = True,
    ) -> None:
        self.data_dir = [
            Path(data_dir) / 'pressure_level_1.5',
            Path(data_dir) / 'single_level_1.5',
            # Path(config.DATA_DIR) / 'oras5'
        ]
        self.normalization_file = [
            Path(data_dir) / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr',
            Path(data_dir) / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr',
            # Path(config.DATA_DIR) / 'climatology' / 'climatology_oras5.zarr'
        ]
        
        self.years = [str(year) for year in years]
        self.n_step = n_step
        self.lead_time = lead_time
        self.single_vars = single_vars
        self.pred_single_vars = pred_single_vars
        self.pred_pressure_vars = pred_pressure_vars
        self.is_normalized = is_normalized
        
        # Subset files that match with patterns (eg. years specified)
        # pressure_level_files, single_level_merge_files, oras5_files = list(), list(), list()
        pressure_level_files, single_level_merge_files = list(), list()
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            
            curr_files = [
                list(self.data_dir[0].glob(f'*{year}*.zarr')),
                list(self.data_dir[1].glob(f'*{year}*.zarr')),
                # list(self.data_dir[2].glob(f'*{year}*.zarr'))
            ]
            
            pressure_level_files.extend([f for f in curr_files[0] if re.match(pattern, str(f.name))])
            single_level_merge_files.extend([f for f in curr_files[1] if re.match(pattern, str(f.name))])
            # oras5_files.extend([f for f in curr_files[2] if re.match(pattern, str(f.name))])
        
        # pressure_level_files.sort(); single_level_merge_files.sort(); oras5_files.sort()
        pressure_level_files.sort(); single_level_merge_files.sort()
        self.file_paths = [pressure_level_files, single_level_merge_files]
        
        # Subsetting
        single_level_merge_idx = [idx for idx, param in enumerate(config.SINGLE_LEVEL_PARAMS) if param in self.single_vars]
        # oras5_idx = [idx for idx, param in enumerate(config.ORAS5_PARAMS) if param in self.ocean_vars]
        
        # Retrieve climatology (i.e., mean and sigma) to normalize
        self.mean_pressure_level = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].values[:, np.newaxis, np.newaxis]
        self.mean_single_level_merge = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].sel(param=self.single_vars).values[:, np.newaxis, np.newaxis]
        self.mean_pressure_level_pred = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].sel(param=[f"{param}-{level}" for param in self.pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.mean_single_level_merge_pred = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].sel(param=self.pred_single_vars).values[:, np.newaxis, np.newaxis]
        # self.mean_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['mean'].values[oras5_idx, np.newaxis, np.newaxis]
        
        self.sigma_pressure_level = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].values[:, np.newaxis, np.newaxis]
        self.sigma_single_level_merge = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].sel(param=self.single_vars).values[:, np.newaxis, np.newaxis]
        self.sigma_pressure_level_pred = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].sel(param=[f"{param}-{level}" for param in self.pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis]
        self.sigma_single_level_merge_pred = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].sel(param=self.pred_single_vars).values[:, np.newaxis, np.newaxis]
        # self.sigma_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['sigma'].values[oras5_idx, np.newaxis, np.newaxis]
        

    def __len__(self):
        data_length = len(self.file_paths[0]) - self.n_step - self.lead_time
        return data_length

    def __getitem__(self, idx):
        pred_indices =  [target_idx for target_idx in range(idx + self.lead_time, idx + self.lead_time + self.n_step)]
        [idx] 
        # pressure_level_data, single_level_merge_data, oras5_data = list(), list(), list()
        pressure_level_data, single_level_merge_data = list(), list()
        pressure_level_data_pred, single_level_merge_data_pred = list(), list()

        pressure_level_data.append(xr.open_dataset(self.file_paths[0][idx], engine='zarr')[config.ERA5_PRESSURE_LIST].to_array().values)
        
        # Process single_level_merge
        if len(self.single_vars) > 0:
            single_level_merge_data.append(xr.open_dataset(self.file_paths[1][idx], engine='zarr')[self.single_vars].to_array().values)

        for step_idx in pred_indices:
            
            # # Process pressure_level
            pressure_level_data_pred.append(xr.open_dataset(self.file_paths[0][step_idx], engine='zarr')[self.pred_pressure_vars].to_array().values)
            
            # Process single_level_merge
            if len(self.single_vars) > 0:
                single_level_merge_data_pred.append(xr.open_dataset(self.file_paths[1][step_idx], engine='zarr')[self.pred_single_vars].to_array().values)
            
            # Process oras5
            # if len(self.ocean_vars) > 0:
            #     oras5_data.append(xr.open_dataset(self.file_paths[2][step_idx], engine='zarr')[self.ocean_vars].to_array().values)
        
        # Permutation / reshaping
        pressure_level_data, single_level_merge_data = np.array(pressure_level_data), np.array(single_level_merge_data)
        pressure_level_data = pressure_level_data.reshape(pressure_level_data.shape[0], -1, pressure_level_data.shape[-2], pressure_level_data.shape[-1]) # Merge (param, level) dims

        pressure_level_data_pred, single_level_merge_data_pred = np.array(pressure_level_data_pred), np.array(single_level_merge_data_pred)
        pressure_level_data_pred = pressure_level_data_pred.reshape(pressure_level_data_pred.shape[0], -1, pressure_level_data_pred.shape[-2], pressure_level_data_pred.shape[-1]) # Merge (param, level) dims
        # Normalize
        if self.is_normalized:
            pressure_level_data = (pressure_level_data - self.mean_pressure_level[np.newaxis, :, :, :]) / self.sigma_pressure_level[np.newaxis, :, :, :]
            single_level_merge_data = (single_level_merge_data - self.mean_single_level_merge[np.newaxis, :, :, :]) / self.sigma_single_level_merge[np.newaxis, :, :, :]
            pressure_level_data_pred = (pressure_level_data_pred - self.mean_pressure_level_pred[np.newaxis, :, :, :]) / self.sigma_pressure_level_pred[np.newaxis, :, :, :]
            single_level_merge_data_pred = (single_level_merge_data_pred - self.mean_single_level_merge_pred[np.newaxis, :, :, :]) / self.sigma_single_level_merge_pred[np.newaxis, :, :, :]
        
        # Concatenate along parameter dimension, only if they are specified (i.e., non-empty)
        input_data = [t for t in [torch.tensor(pressure_level_data), torch.tensor(single_level_merge_data)] if t.nelement() > 0]
        input_data = torch.cat(input_data, dim=1)

        output_data = [t for t in [torch.tensor(pressure_level_data_pred), torch.tensor(single_level_merge_data_pred)] if t.nelement() > 0]
        output_data = torch.cat(output_data, dim=1)

        timestamp = xr.open_dataset(self.file_paths[0][idx], engine='zarr').time.values.item()
        x, y = input_data[0].float(), torch.stack([torch.mean(output_data[0:14].float(),dim=0), torch.mean(output_data[14:28].float(),dim=0)],dim=0)
        return timestamp, x, y



    def _create_edges(self, latitude, longitude,kernel_size):
        print("--------creating edge--------")
        edge = []
        kernel_size = 2

        for lat in tqdm(range(latitude)):
            for lon in range(longitude):

                min_lat = max(0, lat - kernel_size)
                max_lat = min(latitude - 1, lat + kernel_size)

                min_lon = lon - kernel_size
                max_lon = lon + kernel_size

                for la in range(min_lat, max_lat + 1):
                    for lo in range(min_lon, max_lon + 1):
                        if la != lat or lo != lon:
                            edge.append((lat, lon, la, lo % longitude))
        edge_index = [(e[0] * longitude + e[1], e[2] * longitude + e[3]) for e in edge]
        edge_index=torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index

