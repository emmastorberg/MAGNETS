import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """A block with two convolutional layers followed by ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 2,
        num_filters=[32, 64, 128]
    ) -> None:
        super().__init__()

        # Downsampling path
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        prev_channels = input_dim
        for feature_size in num_filters:
            self.encoders.append(ConvBlock1D(prev_channels, feature_size))
            prev_channels = feature_size

        # Bottleneck
        self.bottleneck = ConvBlock1D(prev_channels, prev_channels * 2)
        prev_channels *= 2

        # Upsampling path
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for feature_size in reversed(num_filters):
            self.upconvs.append(nn.ConvTranspose1d(prev_channels, feature_size, kernel_size=kernel_size, stride=2))
            self.decoders.append(ConvBlock1D(feature_size * 2, feature_size))
            prev_channels = feature_size

        # Final convolution
        self.final_conv = nn.Conv1d(num_filters[0], output_dim, kernel_size=1)

        # Initialize weights of final convolution with large positive bias
        # self.final_conv.bias.data.fill_(10)
        # Initialize bias of final convolution layer with random 50% 10 and -10
        # self.final_conv.bias.data = 20 * (torch.rand_like(self.final_conv.bias.data) > 0.5).float() - 10
        # print("Random bias of final conv layer in UNet:", self.final_conv.bias.data)

    def forward(self, x):
        # Downsampling path
        enc_features = []
        for encoder in self.encoders:
            # print("x.shape", x.shape)
            x = encoder(x)
            # print("(encoder) x.shape", x.shape)
            enc_features.append(x)
            x = self.pool(x)
            # print("(pool) x.shape", x.shape)

        # Bottleneck
        x = self.bottleneck(x)
        # print("(bottleneck) x.shape", x.shape)

        # Upsampling path
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            # print(f"(upconv {i}) x.shape", x.shape)
            # Crop and concatenate
            enc_feature = enc_features[-(i + 1)]
            if x.size(-1) != enc_feature.size(-1):  # Handle size mismatch
                enc_feature = F.interpolate(enc_feature, size=x.size(-1))
            x = torch.cat([x, enc_feature], dim=1)
            # print(f"(concat {i}) x.shape", x.shape)
            x = decoder(x)
            # print(f"(decoder {i}) x.shape", x.shape)

        # Final output
        x = self.final_conv(x)
        # print("(final_conv) x.shape", x.shape)
        return x
