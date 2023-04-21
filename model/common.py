import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=0.1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class MultiResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=0.1, num_blocks=3):

        super(MultiResBlock, self).__init__()
        m = []

        for i in range(num_blocks):
            m.append(ResBlock(conv, n_feat, kernel_size, bias, bn, act, res_scale))

        self.convs = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.convs(x)
        res += x
        return res


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, keep_prob, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.keep_prob = keep_prob

        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            # nn.BatchNorm2d(G),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        # return torch.cat((self.keep_prob * x, out), 1)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, keep_prob, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        self.keep_prob = keep_prob

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G, keep_prob=self.keep_prob))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion  mulit-layer->single layer
        # self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        self.LFF = nn.Sequential(
            nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1),
            # nn.BatchNorm2d(G0),
            nn.ReLU()
        )

    def forward(self, x):
        return self.LFF(self.convs(x)) + self.keep_prob * x


class PercepJump(object):
    def __init__(self, alpha=1.0, beta=10, J_min=1.1, Max=9):
        super(PercepJump, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.J_min = J_min
        self.Max = Max

    def non_trans(self, x, c):
        return (x ** 2 + c ** 2).sqrt()

    def pjnorm(self, x):
        return (x - F.relu(x - self.Max)) / self.Max

    def __call__(self, jnd, err):
        err = err.float()

        percep_err = err - jnd
        percep_err = F.relu(percep_err)
        err_v = torch.log(self.non_trans(percep_err, 1)) / torch.log(self.non_trans(jnd * self.alpha, self.J_min))
        err_v = self.pjnorm(err_v + 1)
        return err_v


class GlobalQuality(nn.Module):
    def __init__(self, n_feat):
        super(GlobalQuality, self).__init__()
        self.n_feat = n_feat

        self.mlp = nn.Sequential(
            nn.Linear(n_feat, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, -1)
        return self.mlp(x)