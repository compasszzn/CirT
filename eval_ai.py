import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import lightning.pytorch as pl
from chaosbench.models import model
import yaml
import argparse
import torch
from chaosbench import dataset_new,criterion,config
from torch.utils.data import DataLoader
from pathlib import Path
import xarray as xr
import csv
from tqdm import tqdm
import numpy as np
def reverse_normalize(predict,data_args):
    normalization_file = [
            Path(data_args['data_dir']) / 'climatology_1.5' / 'climatology_pressure_level_1.5_new.zarr',
            Path(data_args['data_dir']) / 'climatology_1.5' / 'climatology_single_level_1.5_new.zarr',
        ]
    pred_single_vars=data_args['pred_single_vars']
    pred_pressure_vars=data_args['pred_pressure_vars']
    mean_pressure_level_pred = torch.tensor(xr.open_dataset(normalization_file[0], engine='zarr')['mean'].sel(param=[f"{param}-{level}" for param in pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis])
    mean_single_level_merge_pred = torch.tensor(xr.open_dataset(normalization_file[1], engine='zarr')['mean'].sel(param=pred_single_vars).values[:, np.newaxis, np.newaxis])
    sigma_pressure_level_pred = torch.tensor(xr.open_dataset(normalization_file[0], engine='zarr')['sigma'].sel(param=[f"{param}-{level}" for param in pred_pressure_vars for level in config.PRESSURE_LEVELS]).values[:, np.newaxis, np.newaxis])
    sigma_single_level_merge_pred = torch.tensor(xr.open_dataset(normalization_file[1], engine='zarr')['sigma'].sel(param=pred_single_vars).values[:, np.newaxis, np.newaxis])
    mean=torch.cat((mean_pressure_level_pred, mean_single_level_merge_pred), dim=0)
    sigma=torch.cat((sigma_pressure_level_pred, sigma_single_level_merge_pred), dim=0)
    predict = predict * sigma + mean
    return predict

def main(args):
    with open(args.config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    if model_args['model_name']=='sh':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S/our/checkpoints/epoch=5-step=2598.ckpt"
    elif model_args['model_name']=='shlong':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S_zzn/nl3q10iq_shlong/checkpoints/epoch=10-step=4763.ckpt"
    elif model_args['model_name']=='rsph':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S_zzn/rz73vdze_RSPH/checkpoints/epoch=13-step=6062.ckpt"
    elif model_args['model_name']=='rsphlong':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S_zzn/60577ife_rsphlong/checkpoints/epoch=9-step=4330.ckpt"
    elif model_args['model_name']=='vit':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S/ck9ozxsf-Vit/checkpoints/epoch=6-step=3031.ckpt"
    elif model_args['model_name']=='vithar':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S_zzn/5nlb5jph_vithar_new/checkpoints/epoch=9-step=4330.ckpt"
    elif model_args['model_name']=='climax':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S/7byketq3-Climax/checkpoints/epoch=16-step=7361.ckpt"
    elif model_args['model_name']=='sh_new':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S_zzn/dm2plao5_shnew/checkpoints/epoch=5-step=2598.ckpt"
    elif model_args['model_name']=='shlong_new':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S_zzn/97tmwaa6_shlong_new/checkpoints/epoch=13-step=6062.ckpt"
    elif model_args['model_name']=='vit_new':
        checkpoint_path = "/home/zinanzheng/project/iclrs2s/S2S/S2S_zzn/lyx1t0r3_vit_new/checkpoints/epoch=17-step=7794.ckpt"

    model_checkpoint = model.S2SBenchmarkModel.load_from_checkpoint(checkpoint_path, model_args=model_args, data_args=data_args)


    # 初始化 Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1)

    # 生成预测值，返回每个 batch 的预测
    predictions= trainer.predict(model_checkpoint)
    all_preds = []
    all_ys = []
    for preds, y in predictions:
        all_preds.append(preds)
        all_ys.append(y)
    all_preds = torch.cat(all_preds, dim=0)  # 依据需要拼接的维度
    all_ys = torch.cat(all_ys, dim=0)
    pred=reverse_normalize(all_preds,data_args)
    
    # 定义特征对应关系
    feature_names = []
    for param in data_args['pred_pressure_vars']:
        for level in config.PRESSURE_LEVELS:
            feature_names.append(f"{param}-{level}")
    feature_names += data_args['pred_single_vars']

    assert pred.shape == all_ys.shape
    results = {}
    RMSE = criterion.RMSE()
    RMSE_GRID = criterion.RMSE_GRID()
    ACC = criterion.ACC()
    MS_SSIM=criterion.MS_SSIM()
    device = torch.device('cpu')
    np.save(f"/home/zinanzheng/project/iclrs2s/S2S/npy/{model_args['model_name']}_pred.npy",pred)
    if args.draw:
        draw_feature=['geopotential-500','geopotential-850','temperature-500','temperature-850','2m_temperature']
        for i, feature_name in enumerate(tqdm(draw_feature)):
            index=feature_names.index(feature_name)
            x = pred[:, :, index, :, :].to(device)
            y = all_ys[:, :, index, :, :] .to(device)
            RMSE_GRID_loss_1 = RMSE_GRID(x[:,0,], y[:,0,])
            RMSE_GRID_loss_2 = RMSE_GRID(x[:,1,], y[:,1,])

    else:
        output_file = Path(f"{model_args['model_name']}_results.csv")
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['pred_vars', 'RMSE_loss_1', 'RMSE_loss_2', 'ACC_loss_1', 'ACC_loss_2','MS_SSIM_loss_1','MS_SSIM_loss_2'])

            for i, feature_name in enumerate(tqdm(feature_names)):
                if '-' in feature_name:
                    source="pressure"
                else:
                    source="single"
                x = pred[:, :, i, :, :].to(device)
                y = all_ys[:, :, i, :, :] .to(device)
                RMSE_loss_1 = RMSE(x[:,0,], y[:,0,])
                RMSE_loss_2 = RMSE(x[:,1,], y[:,1,])
                ACC_loss_1 = ACC(x[:,0,], y[:,0,],feature_name,source)
                ACC_loss_2 = ACC(x[:,1,], y[:,1,],feature_name,source)
                MS_SSIM_loss_1= MS_SSIM(x[:,0], y[:,0])
                MS_SSIM_loss_2= MS_SSIM(x[:,1], y[:,1])
                writer.writerow([feature_name, RMSE_loss_1.item(), RMSE_loss_2.item(), 
                    ACC_loss_1.item(), ACC_loss_2.item(),MS_SSIM_loss_1.item(),MS_SSIM_loss_2.item()])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',default='chaosbench/configs/segformer_s2s.yaml', help='Provide the filepath string to the model config...')
    parser.add_argument('--data_type',default='image', help='Provide the filepath string to the model config...')
    parser.add_argument('--draw',default=False, help='Provide the filepath string to the model config...')
    args = parser.parse_args()
    main(args)
