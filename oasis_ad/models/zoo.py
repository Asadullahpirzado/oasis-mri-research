"""Model zoo. Every model exposes the same interface so training, calibration, OOD and Grad-CAM are model-agnostic:

    feature_map(x)  -> (B, F, d, h, w)  last convolutional features (Grad-CAM target); None for 2D / foundation models
    classify(fm)    -> (logits, embedding)
    forward(x)      -> logits
    embed(x)        -> penultimate embedding used for Mahalanobis OOD scoring
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    has_feature_map = True

    def feature_map(self, x):
        raise NotImplementedError

    def classify(self, fm):
        raise NotImplementedError

    def forward(self, x):
        return self.classify(self.feature_map(x))[0]

    def embed(self, x):
        return self.classify(self.feature_map(x))[1]


# ───────────────────────── compact 3D CNN (trained from scratch) ─────────────────────────
def _gn(c):
    g = 8
    while c % g:
        g //= 2
    return nn.GroupNorm(g, c)


class _Block(nn.Module):
    def __init__(self, i, o, stride, sp_drop=0.0):
        super().__init__()
        layers = [nn.Conv3d(i, o, 3, stride, 1, bias=False), _gn(o), nn.LeakyReLU(0.1, inplace=True),
                  nn.Conv3d(o, o, 3, 1, 1, bias=False), _gn(o), nn.LeakyReLU(0.1, inplace=True)]
        if sp_drop > 0:
            layers.append(nn.Dropout3d(sp_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Compact3DCNN(BaseNet):
    """4 conv stages (channels w,2w,4w,8w) + global average pooling + small MLP head. ~1.2M parameters for w=16."""
    def __init__(self, n_classes=2, width=16, dropout=0.4, spatial_dropout=0.1):
        super().__init__()
        w = width
        self.stages = nn.Sequential(_Block(1, w, 2), _Block(w, 2 * w, 2, spatial_dropout),
                                    _Block(2 * w, 4 * w, 2, spatial_dropout), _Block(4 * w, 8 * w, 2))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * w, 64), nn.LeakyReLU(0.1),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))
        self.emb_dim = 8 * w

    def feature_map(self, x):
        return self.stages(x)

    def classify(self, fm):
        emb = fm.mean(dim=(2, 3, 4))
        return self.head(emb), emb


# ───────────────────────── 3D ResNet-10 (MedicalNet / Med3D layout) ─────────────────────────
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inp, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        r = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            r = self.downsample(x)
        return self.relu(out + r)


class ResNet10_3D(BaseNet):
    """Same layer names as the Med3D (MedicalNet) ResNet-10 so pretrained weights can be loaded (strict=False)."""
    def __init__(self, n_classes=2, dropout=0.4):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._make(64, 1, 1, 1)
        self.layer2 = self._make(128, 1, 2, 1)
        self.layer3 = self._make(256, 1, 1, 2)
        self.layer4 = self._make(512, 1, 1, 4)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, n_classes))
        self.emb_dim = 512
        self.pretrained_report = None

    def _make(self, planes, blocks, stride, dilation):
        ds = None
        if stride != 1 or self.inplanes != planes:
            ds = nn.Sequential(nn.Conv3d(self.inplanes, planes, 1, stride=stride, bias=False), nn.BatchNorm3d(planes))
        layers = [_BasicBlock(self.inplanes, planes, stride, dilation, ds)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(planes, planes, 1, dilation))
        return nn.Sequential(*layers)

    def feature_map(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def classify(self, fm):
        emb = fm.mean(dim=(2, 3, 4))
        return self.head(emb), emb

    def load_medicalnet(self, path):
        """Load Med3D weights. Reports how many tensors matched - CHECK the report before trusting transfer learning."""
        ck = torch.load(path, map_location="cpu")
        sd = ck.get("state_dict", ck)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items() if not k.replace("module.", "", 1).startswith("conv_seg")}
        own = self.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict(ok, strict=False)
        rep = {"matched": len(ok), "own_tensors": len(own), "checkpoint_tensors": len(sd),
               "unmatched_checkpoint_keys": [k for k in sd if k not in ok][:10]}
        self.pretrained_report = rep
        return rep


# ───────────────────────── 2D slice CNN with participant-level aggregation ─────────────────────────
class SliceCNN2D(BaseNet):
    """Takes the whole volume, picks a few slices inside the model, classifies every slice with a small 2D CNN and
    averages the logits -> ONE prediction per participant (slices of one person can never be split)."""
    has_feature_map = False

    def __init__(self, n_classes=2, width=16, dropout=0.4, slice_axis=2, fractions=(0.35, 0.42, 0.5, 0.58, 0.65)):
        super().__init__()
        w = width
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), _gn(o), nn.LeakyReLU(0.1, inplace=True),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False), _gn(o), nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(blk(1, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 8 * w))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(8 * w, 64), nn.LeakyReLU(0.1), nn.Dropout(dropout), nn.Linear(64, n_classes))
        self.axis, self.fractions, self.emb_dim = slice_axis, tuple(fractions), 8 * w

    def _slices(self, x):                     # x: (B,1,D,H,W) -> (B*S,1,h,w)
        n = x.shape[2 + self.axis]
        idx = [min(n - 1, max(0, int(round(f * (n - 1))))) for f in self.fractions]
        sl = [x.select(2 + self.axis, i) for i in idx]          # each (B,1,h,w)
        return torch.stack(sl, 1).flatten(0, 1)

    def feature_map(self, x):
        return self.cnn(self._slices(x))

    def classify(self, fm):                    # fm: (B*S, C, h, w) ; B is recovered from S
        S = len(self.fractions)
        e = fm.mean(dim=(2, 3))
        logit = self.head(e)
        B = e.shape[0] // S
        return logit.view(B, S, -1).mean(1), e.view(B, S, -1).mean(1)


# ───────────────────────── frozen foundation-model encoder + trainable head ─────────────────────────
class FoundationClassifier(BaseNet):
    """Frozen external encoder (TorchScript file supplied by the user) + small trainable head.
    The encoder is NOT bundled: check licence, expected input size/normalisation and weights of the model you use."""
    has_feature_map = False

    def __init__(self, encoder, feat_dim, n_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, n_classes))
        self.emb_dim = feat_dim

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def feature_map(self, x):
        with torch.no_grad():
            f = self.encoder(x)
        if f.dim() > 2:
            f = f.flatten(2).mean(-1)
        return f

    def classify(self, fm):
        return self.head(fm), fm


def build_model(name, cfg, n_classes, paths=None):
    m = cfg.models
    if name == "cnn3d":
        c = m.cnn3d
        return Compact3DCNN(n_classes, c.width, c.dropout, c.spatial_dropout)
    if name == "resnet10":
        net = ResNet10_3D(n_classes, m.resnet10.dropout)
        wpath = cfg.paths.medicalnet_weights
        if m.resnet10.pretrained and wpath:
            rep = net.load_medicalnet(wpath)
            print(f"[MedicalNet] matched {rep['matched']}/{rep['own_tensors']} tensors", flush=True)
        elif m.resnet10.pretrained:
            print("[MedicalNet] paths.medicalnet_weights not set -> ResNet10 trains from scratch", flush=True)
        return net
    if name == "cnn2d":
        c = m.cnn2d
        return SliceCNN2D(n_classes, c.width, c.dropout, c.slice_axis, c.slice_fractions)
    if name == "foundation":
        p = cfg.paths.foundation_encoder
        if not p:
            raise ValueError("paths.foundation_encoder is not set (TorchScript encoder file).")
        enc = torch.jit.load(p, map_location="cpu")
        return FoundationClassifier(enc, m.foundation.feat_dim, n_classes, m.foundation.dropout)
    raise ValueError(f"unknown model '{name}'")


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
