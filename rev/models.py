# rev/models.py
"""
Models. All share the interface forward(x, return_feat=False), x: (B, 10, 512).
  cnn_transformer : 3 strided Conv1d blocks + 3 Transformer encoder layers + MLP head (~0.48M params);
                    no positional embedding and no separate linear projection
  eegnet          : EEGNet-8,2 (Lawhern et al. 2018), kernels set for 128 Hz
  shallow_convnet : ShallowConvNet (Schirrmeister et al. 2017), kernels scaled to 128 Hz
  pure_cnn        : the CNN stem of cnn_transformer + global average pooling (no Transformer), ablation
  pure_transformer: patch embedding (16 samples) + learned positional embedding + 3 Transformer layers
"""
import torch
import torch.nn as nn


class CNNTransformer(nn.Module):
    def __init__(self, in_ch=10, n_classes=2, d_model=128, nhead=4, n_layers=3, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(64, d_model, 7, stride=2, padding=3), nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def features(self, x):
        z = self.cnn(x).transpose(1, 2)
        return self.transformer(z).mean(dim=1)

    def forward(self, x, return_feat=False):
        f = self.features(x)
        out = self.head(f)
        return (out, f) if return_feat else out


class PureCNN(nn.Module):
    """Same CNN stem without the Transformer (ablation)."""
    def __init__(self, in_ch=10, n_classes=2, d_model=128, dropout=0.3):
        super().__init__()
        self.cnn = CNNTransformer(in_ch, n_classes, d_model, dropout=dropout).cnn
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def features(self, x):
        return self.cnn(x).mean(dim=2)

    def forward(self, x, return_feat=False):
        f = self.features(x); out = self.head(f)
        return (out, f) if return_feat else out


class PureTransformer(nn.Module):
    """Non-overlapping 16-sample patches (125 ms) -> 32 tokens, learned positional embedding."""
    def __init__(self, in_ch=10, n_classes=2, d_model=128, nhead=4, n_layers=3, dropout=0.3,
                 patch=16, n_samples=512):
        super().__init__()
        self.embed = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_samples // patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def features(self, x):
        z = self.embed(x).transpose(1, 2) + self.pos
        return self.norm(self.transformer(z)).mean(dim=1)

    def forward(self, x, return_feat=False):
        f = self.features(x); out = self.head(f)
        return (out, f) if return_feat else out


class EEGNet(nn.Module):
    """EEGNet-8,2. Temporal kernel = fs/2 = 64 samples."""
    def __init__(self, in_ch=10, n_classes=2, F1=8, D=2, F2=16, k1=64, k2=16, dropout=0.25,
                 n_samples=512):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), padding=(0, k2 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, in_ch, n_samples))).numel()
        self.head = nn.Linear(n_feat, n_classes)

    def features(self, x):
        return self.block2(self.block1(x.unsqueeze(1))).flatten(1)

    def forward(self, x, return_feat=False):
        f = self.features(x); out = self.head(f)
        return (out, f) if return_feat else out


class _Square(nn.Module):
    def forward(self, x): return x * x


class _SafeLog(nn.Module):
    def forward(self, x): return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """ShallowConvNet; the 250 Hz kernels (25, pool 75/15) scaled to 128 Hz (13, pool 38/8)."""
    def __init__(self, in_ch=10, n_classes=2, n_filt=40, k=13, pool=38, stride=8, dropout=0.5,
                 n_samples=512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, n_filt, (1, k)),
            nn.Conv2d(n_filt, n_filt, (in_ch, 1), bias=False), nn.BatchNorm2d(n_filt),
            _Square(), nn.AvgPool2d((1, pool), stride=(1, stride)), _SafeLog(), nn.Dropout(dropout))
        with torch.no_grad():
            n_feat = self.body(torch.zeros(1, 1, in_ch, n_samples)).numel()
        self.head = nn.Linear(n_feat, n_classes)

    def features(self, x):
        return self.body(x.unsqueeze(1)).flatten(1)

    def forward(self, x, return_feat=False):
        f = self.features(x); out = self.head(f)
        return (out, f) if return_feat else out


MODELS = {"cnn_transformer": CNNTransformer, "pure_cnn": PureCNN,
          "pure_transformer": PureTransformer, "eegnet": EEGNet,
          "shallow_convnet": ShallowConvNet}


def n_params(m):
    return sum(p.numel() for p in m.parameters())
