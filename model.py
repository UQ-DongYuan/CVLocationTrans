import torch
import torch.nn as nn
from featurefusion_network import FeatureFusionNetwork
from resnet import resnet50
import torch.nn.functional as F


class CVLocationTrans(nn.Module):
    def __init__(self, d_model=256):
        super(CVLocationTrans, self).__init__()
        self.sat_extractor = resnet50('sat', output_layers=['layer3'], pretrained=True)
        self.grd_extractor = resnet50('grd', output_layers=['layer3'], pretrained=True)

        # conv 1x1
        self.sat_proj = nn.Conv2d(in_channels=1024, out_channels=d_model, kernel_size=1)
        self.grd_proj = nn.Conv2d(in_channels=1024, out_channels=d_model, kernel_size=1)

        self.feature_fusion_network = FeatureFusionNetwork()

        self.location_head = MLP(d_model, d_model, 1, num_layers=3)
        self.coordinator_head = MLP(d_model, d_model, 2, num_layers=3)

    def forward(self, sat, grd):
        sat_feats = self.sat_extractor(sat)
        grd_feats = self.grd_extractor(grd)

        sat_feats = self.sat_proj(sat_feats)
        grd_feats = self.grd_proj(grd_feats)

        feat_fusion_output = self.feature_fusion_network(sat_feats, grd_feats)

        pred_location = self.location_head(feat_fusion_output)
        coordinate_reg = self.coordinator_head(feat_fusion_output).sigmoid()

        return pred_location, coordinate_reg


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self._initial_weights()

    def _initial_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == '__main__':
    model = CVLocationTrans().to('cuda')
    sat = torch.randn(4, 3, 512, 512).to('cuda')
    grd = torch.randn(4, 3, 160, 240).to('cuda')

    location, xy = model(sat, grd)
    print(location.shape)
    print(xy.shape)


