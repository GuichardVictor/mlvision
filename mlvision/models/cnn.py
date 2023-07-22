import torch
import torch.nn as nn


class SimpleConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, pooling_layer):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
            pooling_layer((2, 2))
            if pooling_layer != nn.Identity()
            else pooling_layer(),
        )

    def forward(self, x):
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, input_features, linears, num_classes):
        super().__init__()

        def linear_block(in_feat, out_feat):
            return nn.Sequential(nn.Linear(in_feat, out_feat), nn.ReLU(inplace=True))

        layers = []
        for i, features in enumerate(linears):
            layers.append(
                linear_block(linears[i - 1] if i > 0 else input_features, features)
            )

        self.mlp = nn.Sequential(*layers, nn.Linear(linears[-1], num_classes))

    def forward(self, x):
        return self.mlp(x)


class SimpleCnn(nn.Module):
    def __init__(self, num_classes, convolutions, pooling, last_pool, linears):
        super().__init__()
        pooling_layer = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d
        convs = []

        convs.append(
            SimpleConvBlock(
                3,
                convolutions[0][0],
                kernel_size=convolutions[0][1],
                pooling_layer=pooling_layer,
            )
        )

        for i, (channels, kernel) in enumerate(convolutions[1:-1]):
            convs.append(
                SimpleConvBlock(convolutions[i][0], channels, kernel, pooling_layer)
            )

        convs.append(
            SimpleConvBlock(
                convolutions[-2][0],
                convolutions[-1][0],
                kernel_size=convolutions[-1][1],
                pooling_layer=pooling_layer if last_pool else nn.Identity,
            )
        )

        self.conv_encoder = nn.Sequential(*convs)

        self.mlp = MLP(convolutions[-1][0], linears, num_classes)

    def forward(self, x):
        x = self.conv_encoder(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        return self.mlp(x)
