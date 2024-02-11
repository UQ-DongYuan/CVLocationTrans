from torch import nn, Tensor
from position_encoding import PositionEncodingSine
import copy
from einops.einops import rearrange

class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source1, source2):
        # for Position embedding
        _, _, src1_h, src1_w = source1.shape
        _, _, src2_h, src2_w = source2.shape

        source1 = rearrange(source1, 'n c h w -> n (h w) c')
        source2 = rearrange(source2, 'n c h w -> n (h w) c')
        source1_output, source2_output = self.encoder(source1, source2, src1_h, src1_w, src2_h, src2_w)
        output = self.decoder(source1_output, source2_output, src1_h, src1_w, src2_h, src2_w)
        return output

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # multi-head self attention
        self.self_attn_source1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_source2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # multi-head cross attention
        self.cros_attn_source1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cros_attn_source2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Sine-Cosine Position Encoding
        self.pos_encoding_source1 = PositionEncodingSine(d_model)
        self.pos_encoding_source2 = PositionEncodingSine(d_model)

        # Implementation of Feedforward model in Cross Attention Block
        self.source_1_FFN = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),  # activation: 'relu'
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.source_2_FFN = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Add & Norm for self attention block
        self.sa_norm1 = nn.LayerNorm(d_model)
        self.sa_norm2 = nn.LayerNorm(d_model)
        self.sa_dropout1 = nn.Dropout(dropout)
        self.sa_dropout2 = nn.Dropout(dropout)

        # Add & Norm for cross attention block
        # source 1
        self.ca_norm1_1 = nn.LayerNorm(d_model)
        self.ca_norm1_2 = nn.LayerNorm(d_model)
        self.ca_dropout1_1 = nn.Dropout(dropout)
        self.ca_dropout1_2 = nn.Dropout(dropout)
        # source 2
        self.ca_norm2_1 = nn.LayerNorm(d_model)
        self.ca_norm2_2 = nn.LayerNorm(d_model)
        self.ca_dropout2_1 = nn.Dropout(dropout)
        self.ca_dropout2_2 = nn.Dropout(dropout)

    def forward(self, src1, src2, src1_h, src1_w, src2_h, src2_w):
        # source 1 self attention
        q1 = k1 = self.pos_encoding_source1(src1, src1_h, src1_w)
        src1_att = self.self_attn_source1(q1, k1, value=src1)[0]
        src1 = src1 + self.sa_dropout1(src1_att)
        src1 = self.sa_norm1(src1)
        # source 2 self attention
        q2 = k2 = self.pos_encoding_source2(src2, src2_h, src2_w)
        src2_att = self.self_attn_source2(q2, k2, value=src2)[0]
        src2 = src2 + self.sa_dropout2(src2_att)
        src2 = self.sa_norm2(src2)

        # source 1 cross attention
        src1_cros = self.cros_attn_source1(query=self.pos_encoding_source1(src1, src1_h, src1_w),
                                           key=self.pos_encoding_source2(src2, src2_h, src2_w), value=src2)[0]
        src1 = src1 + self.ca_dropout1_1(src1_cros)
        src1 = self.ca_norm1_1(src1)
        src1_ffn = self.source_1_FFN(src1)
        src1 = src1 + self.ca_dropout1_2(src1_ffn)
        src1 = self.ca_norm1_2(src1)

        # source 2 cross attention
        src2_cros = self.cros_attn_source2(query=self.pos_encoding_source2(src2, src2_h, src2_w),
                                           key=self.pos_encoding_source1(src1, src1_h, src1_w), value=src1)[0]
        src2 = src2 + self.ca_dropout2_1(src2_cros)
        src2 = self.ca_norm2_1(src2)
        src2_ffn = self.source_2_FFN(src2)
        src2 = src2 + self.ca_dropout2_2(src2_ffn)
        src2 = self.ca_norm2_2(src2)

        return src1, src2

class DecoderCFALayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()

        # Sine-Cosine Position Encoding
        self.pos_encoding_source1 = PositionEncodingSine(d_model)
        self.pos_encoding_source2 = PositionEncodingSine(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.FFN = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src1, src2, src1_h, src1_w, src2_h, src2_w):
        src1_cross = self.cross_attn(query=self.pos_encoding_source1(src1, src1_h, src1_w),
                                     key=self.pos_encoding_source2(src2, src2_h, src2_w), value=src2)[0]
        src1 = src1 + self.dropout1(src1_cross)
        src1 = self.norm1(src1)
        src1_ffn = self.FFN(src1)
        src1 = src1 + self.dropout2(src1_ffn)
        src1 = self.norm2(src1)

        return src1

class Encoder(nn.Module):
    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2, src1_h, src1_w, src2_h, src2_w):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_h, src1_w, src2_h, src2_w)

        return output1, output2

class Decoder(nn.Module):
    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, source1, source2, src1_h, src1_w, src2_h, src2_w):
        output = source1

        for layer in self.layers:
            output = layer(output, source2, src1_h, src1_w, src2_h, src2_w)

        if self.norm is not None:
            output = self.norm(output)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
