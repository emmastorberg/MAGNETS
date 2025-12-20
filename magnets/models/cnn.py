import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class CNN(pl.LightningModule):
    def __init__(self, input_dim=1, lr=1e-3, min_lr=1e-7, lr_decay=0.1, target_scaling=1.0, input_length=128):
        super().__init__()
        self.lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay
        self.input_length = input_length
        self.target_scaling = target_scaling

        self.latent_dim = 128

        self.backbone = torch.nn.Sequential(
            #torch.nn.Flatten(),
            torch.nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(16),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(32),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(64),
            torch.nn.MaxPool1d(kernel_size=2),
            # torch.nn.Conv1d(in_channels=64, out_channels=10, kernel_size=3, padding='same'),
            # torch.nn.ReLU(),
            torch.nn.Flatten(),
            # torch.nn.Linear(16*64, self.latent_dim),
            torch.nn.Linear(64 * (self.input_length // 8), self.latent_dim),
            torch.nn.ReLU(),
        )

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 1),
            torch.nn.Flatten(start_dim=0)
        )


    def forward(self, x):
        feat = self.backbone(x)
        regres = self.regression(feat)
        # return regres, feat
        return regres

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
        # x, y, c = train_batch
        x, y = train_batch
        #x = x.view(x.size(0),-1)
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('task_loss', loss_mse, on_epoch=True)
        self.log("rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('loss', total_loss, on_epoch=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True)
        return total_loss

    def validation_step(self, val_batch, batch_idx):
        # x, y, c = val_batch
        x, y = val_batch
        #x = x.view(x.size(0),-1)
        regres = self.forward(x)
        loss_mse = F.mse_loss(regres, y)
        total_loss = loss_mse
        self.log('val_task_loss', loss_mse)
        self.log("val_rmse_unscaled", np.sqrt(loss_mse.cpu().detach())*self.target_scaling, on_epoch=True)
        self.log('val_loss', total_loss)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])
