import torch.nn as nn
from torch.nn.functional import normalize
import torch

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # global features fusion layer
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(self.view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        # view-consensus features learning layer
        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

    # global feature fusion
    def feature_fusion(self, zs, zs_gradient):
        input = torch.cat(zs, dim=1) if zs_gradient else torch.cat(zs, dim=1).detach()
        return normalize(self.feature_fusion_module(input),dim=1)

    def forward(self, xs, zs_gradient=True):
        rs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_module(z),dim=1)

            rs.append(r)
            zs.append(z)
            xrs.append(xr)

        H = self.feature_fusion(zs,zs_gradient)
        return xrs,zs,rs,H

