import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class NATM(pl.LightningModule):

    def __init__(self, input_dim, input_length, hidden_units, lr=1e-3, min_lr=1e-7, lr_decay=0.1, target_scaling=1.0):
        super(NATM, self).__init__()
        self.lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.input_dim = input_dim
        self.input_length = input_length
        self.target_scaling = target_scaling
        self.feature_time_nets = nn.ModuleDict({
            f'f{f}_t{t}': nn.Sequential(
                nn.Linear(1, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, 1)
            ) for f in range(input_dim) for t in range(input_length)
        })

    def forward(self, x):
        # x shape: (batch_size, input_dim, input_length)
        contributions = torch.zeros_like(x)
        for f in range(self.input_dim):
            for t in range(self.input_length):
                net = self.feature_time_nets[f'f{f}_t{t}']
                contributions[:, f, t] = net(x[:, f, t].unsqueeze(-1)).squeeze(-1)
        return contributions.sum(dim=(1, 2))  # Summing over features and time steps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     # patience=7,
        #     min_lr=self.lr_decay * self.min_lr,
        #     factor=self.lr_decay,
        #     verbose=True,
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('task_loss', loss_mse, on_epoch=True)
        self.log("rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('loss', total_loss, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('val_task_loss', loss_mse)
        self.log("val_rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('val_loss', total_loss)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


class NATMFeature(pl.LightningModule):

    def __init__(self, input_length, hidden_units, lr=1e-3, min_lr=1e-7, lr_decay=0.1, target_scaling=1.0):
        super(NATMFeature, self).__init__()
        self.lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.input_length = input_length
        self.target_scaling = target_scaling
        self.time_nets = nn.ModuleDict({
            f't{t}': nn.Sequential(
                nn.Linear(1, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, 1)
            ) for t in range(input_length)
        })

    def forward(self, x):
        # x shape: (batch_size, input_dim, input_length)
        contributions = torch.zeros_like(x)
        for t in range(self.input_length):
            net = self.time_nets[f't{t}']
            contributions[:, :, t] = net(x[:, :, t].unsqueeze(-1)).squeeze(-1)
        return contributions.sum(dim=(1, 2))  # Summing over features and time steps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     # patience=7,
        #     min_lr=self.lr_decay * self.min_lr,
        #     factor=self.lr_decay,
        #     verbose=True,
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('task_loss', loss_mse, on_epoch=True)
        self.log("rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('loss', total_loss, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('val_task_loss', loss_mse)
        self.log("val_rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('val_loss', total_loss)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


class NATMTime(pl.LightningModule):

    def __init__(self, input_dim, hidden_units, lr=1e-3, min_lr=1e-7, lr_decay=0.1, target_scaling=1.0):
        super(NATMTime, self).__init__()
        self.lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.input_dim = input_dim
        self.target_scaling = target_scaling
        self.feature_nets = nn.ModuleDict({
            f'f{f}': nn.Sequential(
                nn.Linear(1, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, 1)
            ) for f in range(input_dim)
        })

    def forward(self, x):
        # x shape: (batch_size, input_dim, input_length)
        contributions = torch.zeros_like(x)
        for f in range(self.input_dim):
            net = self.feature_nets[f'f{f}']
            contributions[:, f, :] = net(x[:, f, :].unsqueeze(-1)).squeeze(-1)
        return contributions.sum(dim=(1, 2))  # Summing over features and time steps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     # patience=7,
        #     min_lr=self.lr_decay * self.min_lr,
        #     factor=self.lr_decay,
        #     verbose=True,
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('task_loss', loss_mse, on_epoch=True)
        self.log("rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('loss', total_loss, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('val_task_loss', loss_mse)
        self.log("val_rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('val_loss', total_loss)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])
