import random
import torch
from torch import nn
import torch.nn.functional as F

from models.unet import UNet1D


class SimpleMaskGenerator(nn.Module):
    def __init__(self,
            input_dim,
            input_length,
            latent_dim,
            n_masks,
            tau=1.0,
            use_ste=True,
            kernel_size=3,
            mask_smoothing=None,
        ):
        super(SimpleMaskGenerator, self).__init__()

        self.input_dim = input_dim
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.n_masks = n_masks
        self.tau = tau
        self.use_ste = use_ste
        self.mask_smoothing = mask_smoothing

        # CNN version
        self.feature_encoders = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_dim, out_channels=16*input_dim, kernel_size=kernel_size, padding='same', groups=input_dim),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(16*input_dim),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=16*input_dim, out_channels=32*input_dim, kernel_size=kernel_size, padding='same', groups=input_dim),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(32*input_dim),
            torch.nn.MaxPool1d(kernel_size=2),
            torch.nn.Conv1d(in_channels=32*input_dim, out_channels=64*input_dim, kernel_size=kernel_size, padding='same', groups=input_dim),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm1d(64*input_dim),
            torch.nn.MaxPool1d(kernel_size=2),
            # torch.nn.Conv1d(in_channels=64*input_dim, out_channels=self.latent_dim*input_dim, kernel_size=16*64*input_dim, padding='valid', groups=input_dim),
            # torch.nn.Flatten(),
            # torch.nn.Linear(16*64, self.latent_dim),
            torch.nn.ReLU(),
        )

        self.linear_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_length // 8 * 64, self.latent_dim) for _ in range(input_dim)
        ])

        self.time_prob_nets = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.latent_dim, input_length),
                    nn.Sigmoid()
                )
                for _ in range(input_dim)
            ])
            for _ in range(n_masks)
        ])

        if self.mask_smoothing == "avgpool":
            # Add average pooling to blur the mask
            self.mask_smoothing_layer = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        elif self.mask_smoothing == "gaussian":
            # Blur with a gaussian kernel
            self.mask_smoothing_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            self.mask_smoothing_layer.weight.data = torch.tensor([[[0.1, 0.8, 0.1]]])
        elif self.mask_smoothing == "conv":
            # Blur with a learnable kernel
            self.mask_smoothing_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            # torch.nn.init.xavier_uniform(self.mask_smoothing.weight)

        self.init_weights()

    def init_weights(self):
        def iweights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                # torch.nn.init.xavier_uniform(m.weight)
                # m.bias.data.fill_(0.01)
                torch.nn.init.zeros_(m.weight)
                # torch.nn.init.zeros_(m.bias)

        def ibias(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                if random.random() < 0.5:
                    torch.nn.init.constant_(m.bias, 10.0)
                else:
                    torch.nn.init.constant_(m.bias, -10.0)

        # self.time_prob_nets.apply(iweights)
        self.time_prob_nets.apply(ibias)

        # initialize all layers with zeros
        # self.feature_encoders.apply(iweights)
        # self.linear_layers.apply(iweights)

    def reparameterize(self, total_mask):
        # print(total_mask.shape)
        # print(total_mask[0])
        # print(self.d_inp)
        # Need to add extra dim:
        inv_probs = 1 - total_mask
        total_mask_prob = torch.stack([inv_probs, total_mask], dim=-1)
        total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau=self.tau, hard=self.use_ste)[...,1]
        return total_mask_reparameterize

    def forward(self, x):
        # CNN version
        # print("x shape", x.shape)
        z = self.feature_encoders(x)
        # print("z shape", z.shape)
        z = z.reshape(z.shape[0], self.input_dim, -1)
        # print("z shape", z.shape)
        z = torch.stack([linear(z[:,i,:]) for i, linear in enumerate(self.linear_layers)], dim=1)
        # print("z shape", z.shape)
        p_time = torch.stack([
            torch.stack([self.time_prob_nets[m][i](z[:,i])
                for i in range(self.input_dim)], dim=1)
            for m in range(self.n_masks)
        ], dim=1)

        total_mask_reparameterize = torch.stack([
            torch.stack([self.reparameterize(p_time[:, m, i, :])
                for i in range(self.input_dim)], dim=1)
            for m in range(self.n_masks)
        ], dim=1)
        # print("total_mask_reparameterize shape", total_mask_reparameterize.shape)
        # if self.input_dim == 1:
        #     total_mask = p_time.softmax(dim=-1)[...,1].unsqueeze(-1)
        # else:
        #     total_mask = p_time # Already sigmoid transformed
        total_mask = p_time

        # print("total_mask shape", total_mask.shape)

        if self.mask_smoothing:
            total_mask = torch.stack([
                self.mask_smoothing_layer(total_mask[:, m, :, :])
                for m in range(self.n_masks)], dim=1)
            total_mask_reparameterize = torch.stack([
                self.mask_smoothing_layer(total_mask_reparameterize[:, m, :, :])
                for m in range(self.n_masks)], dim=1)

        # print("total_mask shape after smoothing", total_mask.shape)

        return total_mask, total_mask_reparameterize


class UnetMaskGenerator(nn.Module):
    def __init__(self,
            input_dim,
            input_length,
            n_masks,
            tau=1.0,
            use_ste=True,
            kernel_size=3,
            mask_smoothing=None,
        ):
        super(UnetMaskGenerator, self).__init__()

        self.input_dim = input_dim
        self.input_length = input_length
        self.n_masks = n_masks
        self.tau = tau
        self.use_ste = use_ste
        self.mask_smoothing = mask_smoothing

        # Unet version
        self.unet = UNet1D(input_dim, output_dim=n_masks * input_dim, kernel_size=kernel_size)

        if self.mask_smoothing == "avgpool":
            # Add average pooling to blur the mask
            self.mask_smoothing_layer = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        elif self.mask_smoothing == "gaussian":
            # Blur with a gaussian kernel
            self.mask_smoothing_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            self.mask_smoothing_layer.weight.data = torch.tensor([[[0.1, 0.8, 0.1]]])
        elif self.mask_smoothing == "conv":
            # Blur with a learnable kernel
            self.mask_smoothing_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
            # torch.nn.init.xavier_uniform(self.mask_smoothing.weight)

    def reparameterize(self, total_mask):
        # print(total_mask.shape)
        # print(total_mask[0])
        # print(self.d_inp)
        # Need to add extra dim:
        inv_probs = 1 - total_mask
        total_mask_prob = torch.stack([inv_probs, total_mask], dim=-1)
        total_mask_reparameterize = F.gumbel_softmax(torch.log(total_mask_prob + 1e-9), tau=self.tau, hard=self.use_ste)[...,1]
        return total_mask_reparameterize

    def forward(self, x):
        # Unet version
        p_time = self.unet(x).sigmoid()
        # print("p_time shape", p_time.shape)
        p_time = p_time.reshape(-1, self.n_masks, self.input_dim, self.input_length)
        # print("p_time shape", p_time.shape)
        # print("p_time", p_time)

        total_mask_reparameterize = torch.stack([
            torch.stack([self.reparameterize(p_time[:, m, i, :])
                for i in range(self.input_dim)], dim=1)
            for m in range(self.n_masks)
        ], dim=1)
        # print("total_mask_reparameterize", total_mask_reparameterize)
        # print("total_mask_reparameterize shape", total_mask_reparameterize.shape)
        # if self.input_dim == 1:
        #     total_mask = p_time.softmax(dim=-1)[...,1].unsqueeze(-1)
        # else:
        #     total_mask = p_time # Already sigmoid transformed
        total_mask = p_time

        # print("total_mask shape", total_mask.shape)

        if self.mask_smoothing:
            total_mask = torch.stack([
                self.mask_smoothing_layer(total_mask[:, m, :, :])
                for m in range(self.n_masks)], dim=1)
            total_mask_reparameterize = torch.stack([
                self.mask_smoothing_layer(total_mask_reparameterize[:, m, :, :])
                for m in range(self.n_masks)], dim=1)

        # print("total_mask shape after smoothing", total_mask.shape)

        return total_mask, total_mask_reparameterize
