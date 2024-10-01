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

from chaosbench.models import mlp, cnn, ae, fno, rsphlong, vit,vit_new,vit_2_2_har, gnn,vit_2_2, climax, equis2s, sph, csph, rsph, sh,shlong,vithar,sh_new,shlong_new,rsph_new,rsphlong_new
from chaosbench import dataset_new, config, utils, criterion

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
        
        if 'mlp' in self.model_args['model_name']:
            self.model = mlp.MLP(input_size = input_size,
                                 hidden_sizes = self.model_args['hidden_sizes'], 
                                 output_size = output_size)
            
        elif 'unet' in self.model_args['model_name']:
            self.model = cnn.UNet(input_size = input_size,
                                   output_size = output_size)
            
        elif 'resnet' in self.model_args['model_name']:
            self.model = cnn.ResNet(input_size = input_size,
                                   output_size = output_size)
            
        elif 'vae' in self.model_args['model_name']:
            self.model = ae.VAE(input_size = input_size,
                                 output_size = output_size,
                                 latent_size = self.model_args['latent_size'])
            
        elif 'ed' in self.model_args['model_name']:
            self.model = ae.EncoderDecoder(input_size = input_size,
                                           output_size = output_size)
            
        elif 'fno' in self.model_args['model_name']:
            self.model = fno.FNO2d(input_size = input_size,
                                   modes1 = self.model_args['modes1'], 
                                   modes2 = self.model_args['modes2'], 
                                   width = self.model_args['width'], 
                                   initial_step = self.model_args['initial_step'])
            
        elif 'vit' == self.model_args['model_name']:
            self.model = vit.ViT(input_size = input_size)
        elif 'vit_2_2' == self.model_args['model_name']:
            self.model = vit_2_2.ViT(input_size = input_size)
        elif 'vit_2_2_har' == self.model_args['model_name']:
            self.model = vit_2_2_har.ViT(input_size = input_size)
        elif 'vit_new' == self.model_args['model_name']:
            self.model = vit_new.ViT(input_size = input_size)
        elif 'climax' in self.model_args['model_name']:
            self.model = climax.ClimaX()
        # elif 'equi' in self.model_args['model_name']:
        #     self.model = equis2s.EquiS2S(input_size = input_size)
        elif 'rsph' == self.model_args['model_name']:
            self.model = rsph.SphFormer(input_size = input_size)
        elif 'sh' == self.model_args['model_name']:
            self.model = sh.SHFormer(input_size = input_size)
        elif 'rsphlong' == self.model_args['model_name']:
            self.model = rsphlong.SphFormer(input_size = input_size)
        elif 'shlong' == self.model_args['model_name']:
            self.model = shlong.SHFormer(input_size = input_size)
        elif 'vithar' == self.model_args['model_name']:
            self.model = vithar.ViT(input_size = input_size)
        elif 'sh_new' == self.model_args['model_name']:
            self.model = sh_new.SHFormer(input_size = input_size)
        elif 'shlong_new' == self.model_args['model_name']:
            self.model = shlong_new.SHFormer(input_size = input_size)
        elif 'rsph_new' == self.model_args['model_name']:
            self.model = rsph_new.SphFormer(input_size = input_size)
        elif 'rsphlong_new' == self.model_args['model_name']:
            self.model = rsphlong_new.SphFormer(input_size = input_size)
        
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
        if self.model_args['model_name'] in ['vit','climax','vithar','vit_2_2','vit_2_2_har']:
            x = F.pad(x, (0, 0, 0, 3), "constant", 0) 
            y = F.pad(y, (0, 0, 0, 3), "constant", 0) 
        elif self.model_args['model_name'] == 'vit_new':
            x = F.pad(x, (0, 0, 0, 7), "constant", 0) 
            y = F.pad(y, (0, 0, 0, 7), "constant", 0) 
        # print(x.shape)

        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        
        loss=self.loss(preds,y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        timestamp, x, y = batch # x: [batch, input_size, height, width] y: [batch, step, input_size, height, width]
        if self.model_args['model_name'] in ['vit','climax','vithar','vit_2_2','vit_2_2_har']:
            x = F.pad(x, (0, 0, 0, 3), "constant", 0) 
        elif self.model_args['model_name'] == 'vit_new':
            x = F.pad(x, (0, 0, 0, 7), "constant", 0) 

        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        preds = preds[:, :, :, :121, :]
        
        loss=self.loss(preds,y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        timestamp, x, y = batch
        if self.model_args['model_name'] in ['vit','climax','vithar','vit_2_2','vit_2_2_har']:
            x = F.pad(x, (0, 0, 0, 3), "constant", 0) 
        elif self.model_args['model_name'] == 'vit_new':
            x = F.pad(x, (0, 0, 0, 7), "constant", 0) 
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        preds = self(x)
        preds = preds[:, :, :, :121, :]              
        
        loss=self.val_loss(preds,y)

        for i in range(10):
            loss1 = self.val_loss(preds[:, 0, i, :, :],y[:, 0, i, :, :])
            loss2=self.val_loss(preds[:, 1, i, :, :],y[:, 1, i, :, :])

            self.log("z" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("z" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i in range(10, 20):
            loss1 = self.val_loss(preds[:, 0, i, :, :],y[:, 0, i, :, :])
            loss2=self.val_loss(preds[:, 1, i, :, :],y[:, 1, i, :, :])

            self.log("q" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("q" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i in range(20, 30):
            loss1 = self.val_loss(preds[:, 0, i, :, :],y[:, 0, i, :, :])
            loss2=self.val_loss(preds[:, 1, i, :, :],y[:, 1, i, :, :])

            self.log("t" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("t" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i in range(60, 63):
            loss1 = self.val_loss(preds[:, 0, i, :, :],y[:, 0, i, :, :])
            loss2=self.val_loss(preds[:, 1, i, :, :],y[:, 1, i, :, :])

            self.log("single" + str(i) + " Week34", loss1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("single" + str(i) + " Week56", loss2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def predict_step(self, batch, batch_idx):
        timestamp, x, y = batch
        if self.model_args['model_name'] in ['vit','climax','vithar','vit_2_2','vit_2_2_har']:
            x = F.pad(x, (0, 0, 0, 3), "constant", 0) 
        elif self.model_args['model_name'] == 'vit_new':
            x = F.pad(x, (0, 0, 0, 7), "constant", 0) 
        preds = self(x)
        preds = preds[:, :, :, :121, :]  # 截断到所需的尺寸
        return preds,y

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
        self.train_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['train_years'], 
                                                   n_step=self.data_args['n_step'],
                                                   lead_time=self.data_args['lead_time'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                   type = "image"
                                                #    ocean_vars=self.data_args['ocean_vars']
                                                  )
        self.val_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                  years=self.data_args['val_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 single_vars=self.data_args['single_vars'],
                                                 pred_single_vars=self.data_args['pred_single_vars'],
                                                 pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                 type = "image"
                                                #  ocean_vars=self.data_args['ocean_vars']
                                                )
        
        self.test_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                  years=self.data_args['test_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 single_vars=self.data_args['single_vars'],
                                                 pred_single_vars=self.data_args['pred_single_vars'],
                                                 pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                 type = "image"
                                                #  ocean_vars=self.data_args['ocean_vars']
                                                )
        self.predict_dataset = dataset_new.S2SInferDataset(data_dir=self.data_args['data_dir'],
                                                  years=self.data_args['test_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 single_vars=self.data_args['single_vars'],
                                                 pred_single_vars=self.data_args['pred_single_vars'],
                                                 pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                 type = "image"
                                                #  ocean_vars=self.data_args['ocean_vars']
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
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
    
class S2SGNNModel(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(S2SGNNModel, self).__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args
        
        # Initialize model
        # ocean_vars = self.data_args.get('ocean_vars', [])
        input_size = self.model_args['input_size'] 
        output_size = self.model_args['output_size']
        
        if 'mlp' in self.model_args['model_name']:
            self.model = mlp.MLP(input_size = input_size,
                                 hidden_sizes = self.model_args['hidden_sizes'], 
                                 output_size = output_size)
        elif 'gnn' in self.model_args['model_name']:
            self.model = gnn.GNN(input_dim = input_size,
                                 hidden_nf = self.model_args['hidden_sizes'], 
                                 output_dim = output_size,
                                 pred_len = self.model_args['pred_len'])
            
        
        ##################################
        # INITIALIZE YOUR OWN MODEL HERE #
        ##################################
        
        self.loss = self.init_loss_fn()
            
    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def forward(self, nodes,edges,edge_attr=None):
        return self.model(nodes,edges,edge_attr)

    def training_step(self, batch, batch_idx):
        timestamp, x, y,edge_index = batch.timestamp, batch.x, batch.y,batch.edge_index
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        
        preds = self(nodes=x,edges=edge_index)
        # for step_idx in range(n_steps):
        #     preds = self(x)
            
        #     # Optimize for headline variables
        #     if self.model_args['only_headline']:
        #         headline_idx = [
        #             config.PARAMS.index(headline_var.split('-')[0]) * len(config.PRESSURE_LEVELS)
        #             + config.PRESSURE_LEVELS.index(int(headline_var.split('-')[1])) for headline_var in config.HEADLINE_VARS
        #         ]
                
        #         loss += self.loss(
        #             preds.view(preds.size(0), -1, preds.size(-2), preds.size(-1))[:, headline_idx],
        #             y[:, step_idx].view(y.size(0), -1, y.size(-2), y.size(-1))[:, headline_idx]
        #         )
            
        #     # Otherwise, for all variables
        #     else:
        #         loss += self.loss(preds[:, :self.model_args['output_size']], y[:, step_idx, :self.model_args['output_size']])
            
        #     x = preds
            
        # loss = loss / n_steps
        ####################################################
        loss=self.loss(preds,y.reshape(y.shape[0],-1))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        timestamp, x, y, edge_index = batch.timestamp, batch.x, batch.y,batch.edge_index
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0

        preds = self(nodes=x,edges=edge_index)
        
        # for step_idx in range(n_steps):
        #     preds = self(x)
            
        #     # Optimize for headline variables
        #     if self.model_args['only_headline']:
        #         headline_idx = [
        #             config.PARAMS.index(headline_var.split('-')[0]) * len(config.PRESSURE_LEVELS)
        #             + config.PRESSURE_LEVELS.index(int(headline_var.split('-')[1])) for headline_var in config.HEADLINE_VARS
        #         ]
                
        #         loss += self.loss(
        #             preds.view(preds.size(0), -1, preds.size(-2), preds.size(-1))[:, headline_idx],
        #             y[:, step_idx].view(y.size(0), -1, y.size(-2), y.size(-1))[:, headline_idx]
        #         )
                
        #     # Otherwise, for all variables
        #     else:
        #         loss += self.loss(preds[:, :self.model_args['output_size']], y[:, step_idx, :self.model_args['output_size']])
            
        #     x = preds
            
        # loss = loss / n_steps
        ####################################################
        loss=self.loss(preds,y.reshape(y.shape[0],-1))
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        self.train_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['train_years'], 
                                                    n_step=self.data_args['n_step'],
                                                    lead_time=self.data_args['lead_time'],
                                                    kernel_size=self.data_args['kernel_size'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    type = "graph",
                                                #    ocean_vars=self.data_args['ocean_vars']
                                                  )
        self.val_dataset = dataset_new.S2SDataset(data_dir=self.data_args['data_dir'],
                                                    years=self.data_args['val_years'], 
                                                    n_step=self.data_args['n_step'],
                                                    lead_time=self.data_args['lead_time'],
                                                    kernel_size=self.data_args['kernel_size'],
                                                    single_vars=self.data_args['single_vars'],
                                                    pred_single_vars=self.data_args['pred_single_vars'],
                                                    pred_pressure_vars=self.data_args['pred_pressure_vars'],
                                                    type = "graph",
                                                #  ocean_vars=self.data_args['ocean_vars']
                                                )
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
