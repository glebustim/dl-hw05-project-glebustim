import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.convs(x)


class DRUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=[32, 64, 128, 256], num_res_blocks=4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.down_convs = nn.ModuleList()
        self.down_scales = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_convs.append(nn.Sequential(*[ResBlock(channels[i]) for _ in range(num_res_blocks)]))
            self.down_scales.append(nn.Conv2d(channels[i], channels[i + 1], 2, stride=2))

        self.mid = nn.Sequential(*[ResBlock(channels[-1]) for _ in range(num_res_blocks)])
        
        self.up_convs = nn.ModuleList()
        self.up_scales = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_scales.append(nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2))
            self.up_convs.append(nn.Sequential(*[ResBlock(channels[i - 1]) for _ in range(num_res_blocks)]))
            
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        output_size = x.shape[-2:]

        skips = []
        for down_conv, down_scale in zip(self.down_convs, self.down_scales):
            x = down_conv(x)
            x = down_scale(x)
            skips.append(x)

        x = self.mid(x)
        skips = list(reversed(skips))
        
        for i, (up_conv, up_scale, skip) in enumerate(zip(self.up_convs, self.up_scales, skips)):
            target_size = skips[i + 1].shape[-2:] if i + 1 < len(skips) else output_size
            target_shape = (x.shape[0], up_scale.out_channels, *target_size)
            x = up_scale(x + skip, output_size=target_shape)
            x = up_conv(x)
            
        return self.out_conv(x)
