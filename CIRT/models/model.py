import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch

torch.autograd.set_detect_anomaly(True)
# from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from CIRT.models import CirT
from CIRT import dataset, config, utils, criterion

class S2SBenchmarkModel(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(S2SBenchmarkModel, self).__init__()
        self.save_hyperparameters()
        self.model_args = model_args
        self.data_args = data_args
        
        # Initialize model
        # ocean_vars = self.data_args.get('ocean_vars', [])
        input_size = self.model_args['input_size'] 
        output_size = self.model_args['output_size'] 
        

        self.model = CirT.Model(input_size = input_size)

        
        ##################################
        # INITIALIZE YOUR OWN MODEL HERE #
        ##################################
        
        self.loss = self.init_loss_fn()
        self.val_loss = criterion.RMSE()
            
    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        timestamp, x, y = batch # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        
        loss=self.loss(preds,y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        timestamp, x, y = batch # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        preds = preds[:, :, :, :121, :]
        
        loss=self.loss(preds,y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        timestamp, x, y = batch
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        preds = preds[:, :, :, :121, :]              
        
        loss=self.val_loss(preds,y)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_args['learning_rate'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['t_max'], eta_min=self.model_args['learning_rate'] / 10),
                'interval': 'epoch',
            }
        }

    def setup(self, stage=None):
        self.train_dataset = dataset.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['train_years'], 
                                                   n_step=self.data_args['n_step'],
                                                   lead_time=self.data_args['lead_time'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                  )
        self.val_dataset = dataset.S2SDataset(data_dir=self.data_args['data_dir'],
                                                  years=self.data_args['val_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 single_vars=self.data_args['single_vars'],
                                                 pred_single_vars=self.data_args['pred_single_vars'],
                                                 pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                )
        
        self.test_dataset = dataset.S2SDataset(data_dir=self.data_args['data_dir'],
                                                  years=self.data_args['test_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 single_vars=self.data_args['single_vars'],
                                                 pred_single_vars=self.data_args['pred_single_vars'],
                                                 pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                )
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
    