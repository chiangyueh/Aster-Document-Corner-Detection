                   
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BN_EPS = 1e-3


CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


TAPS = {3: "c2", 6: "c3", 13: "c4"}


def make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k=1, stride=1, groups=1, act=True):
        layers = [nn.Conv2d(cin, cout, k, stride, 0 if stride == 2 else k // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout, eps=BN_EPS)]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)
        self.tf_pad = stride == 2 and k > 1

    def forward(self, x):
        if self.tf_pad:
            x = F.pad(x, (0, 1, 0, 1))
        return super().forward(x)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        hidden = cin * expand
        self.expand = ConvBN(cin, hidden, 1) if expand != 1 else None
        self.depthwise = ConvBN(hidden, hidden, 3, stride, groups=hidden)
        self.project = ConvBN(hidden, cout, 1, act=False)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        e = self.expand(x) if self.expand is not None else x
        y = self.project(self.depthwise(e))
        return (x + y if self.residual else y), e


class MobileNetV2(nn.Module):
    def __init__(self, alpha=0.35):
        super().__init__()
        cin = make_divisible(32 * alpha)
        self.stem = ConvBN(3, cin, 3, 2)
        blocks, self.tap_channels = [], {}
        for t, c, n, s in CFG:
            cout = make_divisible(c * alpha)
            for i in range(n):
                idx = len(blocks)
                if idx in TAPS:
                    self.tap_channels[TAPS[idx]] = cin * t
                blocks.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.top_channels = 1280 if alpha <= 1.0 else make_divisible(1280 * alpha)
        self.top = ConvBN(cin, self.top_channels, 1)

    def forward(self, x):
        out = {}
        x = self.stem(x)
        for i, b in enumerate(self.blocks):
            x, e = b(x)
            if i in TAPS:
                out[TAPS[i]] = e
        out["top"] = self.top(x)
        return out


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=1, groups=1, act="silu", dilation=(1, 1), kernel=None):
        kernel = kernel or (k, k)
        pad = tuple(d * (kk // 2) for kk, d in zip(kernel, dilation))
        layers = [nn.Conv2d(cin, cout, kernel, 1, pad, dilation=dilation, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act == "silu":
            layers.append(nn.SiLU(inplace=True))
        elif act == "relu":
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SepConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cin, 3, 1, 1, groups=cin, bias=False),
                         ConvBNAct(cin, cout, 1))


class MiniFPN(nn.Module):
    def __init__(self, ch, width=32):
        super().__init__()
        self.lat2 = ConvBNAct(ch["c2"], width, act=None)
        self.lat3 = ConvBNAct(ch["c3"], width, act=None)
        self.lat4 = ConvBNAct(ch["c4"], width, act=None)
        self.ref3 = SepConv(width, width)
        self.ref2 = SepConv(width, width)
        self.fuse = nn.Sequential(SepConv(2 * width, 2 * width), SepConv(2 * width, width))

    def forward(self, f):
        p4 = self.lat4(f["c4"])
        p3 = self.ref3(self.lat3(f["c3"]) + F.interpolate(p4, scale_factor=2.0, mode="nearest"))
        p2 = self.ref2(self.lat2(f["c2"]) + F.interpolate(p3, scale_factor=2.0, mode="nearest"))
        p3u = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([p2, p3u], 1))


class AxisHead(nn.Module):
    def __init__(self, cin, gdim, n_inside, pad, width=48, dilations=(1, 2, 4, 8, 16, 32, 64, 2, 8, 32)):
        super().__init__()
        self.pad = pad
        n = n_inside + 2 * pad
        idx = torch.arange(n, dtype=torch.float32) - pad
        inside = ((idx >= 0) & (idx < n_inside)).float()
        dist = torch.where(idx < 0, idx, torch.where(idx >= n_inside, idx - n_inside + 1, torch.zeros_like(idx))) / pad
        self.register_buffer("pos", torch.stack([inside, dist]).view(1, 2, 1, n), persistent=False)
        self.inp = ConvBNAct(cin + gdim + 2, width, kernel=(1, 5), act="relu")
        self.blocks = nn.ModuleList(ConvBNAct(width, width, kernel=(1, 3), dilation=(1, d), act="relu")
                                    for d in dilations)
        self.out = nn.Conv2d(width, 4, 1)

    def forward(self, prof, g):
        b = prof.shape[0]
        x = F.pad(prof, (self.pad, self.pad))
        n = x.shape[-1]
        x = torch.cat([x, g.view(b, -1, 1, 1).expand(-1, -1, 1, n), self.pos.expand(b, -1, -1, -1)], 1)
        x = self.inp(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x).squeeze(2)


class SimCCHead(nn.Module):
    def __init__(self, cin, n_inside=224, pad=112, gdim=48):
        super().__init__()
        self.n_inside = n_inside
        self.glob = nn.Sequential(nn.Linear(cin, gdim), nn.ReLU(inplace=True))
        self.hx = AxisHead(cin, gdim, n_inside, pad)
        self.hy = AxisHead(cin, gdim, n_inside, pad)
        n = n_inside + 2 * pad
        centers = (torch.arange(n, dtype=torch.float32) - pad + 0.5) / n_inside
        self.register_buffer("centers", centers, persistent=False)

    def forward(self, p):
        g = self.glob(p.mean(dim=(2, 3)))
        px = F.interpolate(p.mean(dim=2, keepdim=True), size=(1, self.n_inside), mode="bilinear", align_corners=False)
        py = F.interpolate(p.mean(dim=3, keepdim=True).transpose(2, 3), size=(1, self.n_inside), mode="bilinear",
                           align_corners=False)
        lx = self.hx(px, g)
        ly = self.hy(py, g)
        return lx, ly

    def decode(self, lx, ly):
        x = (lx.softmax(-1) * self.centers).sum(-1)
        y = (ly.softmax(-1) * self.centers).sum(-1)
        return torch.stack([x, y], -1).flatten(1)


def _ds(f, group):
    g = f[group][group]
    return {k.split(":")[0]: np.array(v) for k, v in g.items()}


def _conv(conv, w, depthwise=False):
    k = w["depthwise_kernel"] if depthwise else w["kernel"]
    k = k.transpose(2, 3, 0, 1) if depthwise else k.transpose(3, 2, 0, 1)
    conv.weight.data.copy_(torch.from_numpy(k))


def _bn(bn, w):
    bn.weight.data.copy_(torch.from_numpy(w["gamma"]))
    bn.bias.data.copy_(torch.from_numpy(w["beta"]))
    bn.running_mean.copy_(torch.from_numpy(w["moving_mean"]))
    bn.running_var.copy_(torch.from_numpy(w["moving_variance"]))


def load_keras_mobilenetv2(model, path):
    import h5py
    with h5py.File(path, "r") as f:
        _conv(model.stem[0], _ds(f, "Conv1"))
        _bn(model.stem[1], _ds(f, "bn_Conv1"))
        for i, b in enumerate(model.blocks):
            if b.expand is not None:
                _conv(b.expand[0], _ds(f, f"mobl{i}_conv_{i}_expand"))
                _bn(b.expand[1], _ds(f, f"bn{i}_conv_{i}_bn_expand"))
            _conv(b.depthwise[0], _ds(f, f"mobl{i}_conv_{i}_depthwise"), depthwise=True)
            _bn(b.depthwise[1], _ds(f, f"bn{i}_conv_{i}_bn_depthwise"))
            _conv(b.project[0], _ds(f, f"mobl{i}_conv_{i}_project"))
            _bn(b.project[1], _ds(f, f"bn{i}_conv_{i}_bn_project"))
        _conv(model.top[0], _ds(f, "Conv_1"))
        _bn(model.top[1], _ds(f, "Conv_1_bn"))
    return model


class Aster(nn.Module):
    def __init__(self, alpha=0.35, fpn_width=32, n_inside=224, pad_bins=112):
        super().__init__()
        self.backbone = MobileNetV2(alpha)
        self.fpn = MiniFPN(self.backbone.tap_channels, fpn_width)
        self.simcc = SimCCHead(fpn_width, n_inside, pad_bins)
        self.score = nn.Linear(self.backbone.top_channels, 1)

    def forward(self, x):
        f = self.backbone(x)
        lx, ly = self.simcc(self.fpn(f))
        score = self.score(f["top"].mean(dim=(2, 3)))
        return dict(logits_x=lx, logits_y=ly, coords=self.simcc.decode(lx, ly), score_logit=score)


def build_model(cfg, pretrained_path=None):
    m = cfg["model"]
    net = Aster(m["alpha"], m["fpn_width"], m["bins_inside"], m["pad_bins"])
    if pretrained_path:
        load_keras_mobilenetv2(net.backbone, pretrained_path)
    return net


EXPORT_OUTPUTS = ["coords", "score_logit", "logits_x", "logits_y"]


class ExportWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out["coords"], out["score_logit"], out["logits_x"], out["logits_y"]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["cfg"])
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint["cfg"]
