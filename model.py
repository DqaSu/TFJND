from argparse import Namespace
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


class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        self.fc = True
        G0 = args.G0
        kSize = args.RDNkSize
        keep_prob = args.keep_prob

        # number of RDB blocks, conv layers, out channels, C should be more than 6
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (12, 6, 16),
            'D': (4, 4, 32),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Sequential(
            nn.Conv2d(args.n_colors, G0 // 2, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        )
        self.SFENet2 = nn.Sequential(
            nn.Conv2d(G0 // 2, G0, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        )

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C, keep_prob=keep_prob)
            )

        # Global Feature Fusion
        fd = self.D * G0
        self.GFF = nn.Sequential(*[
            nn.Conv2d(fd, fd//2, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(fd//2, fd//4, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(fd//4, fd//8, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(fd//8, args.out_dim, 1, padding=0, stride=1),
            nn.ReLU(),
        ])

        self.lpd = PercepJump()
        self.avg_pool = nn.AdaptiveAvgPool2d((32, 32))

        self.globq = nn.Sequential(
            nn.Linear(32**2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x, err=None, training=False):
        if training:
            if self.fc:
                x = self._jnd_predictor(x)
                x = self._quality_predictor(x, err)
            else:
                x = self._jnd_predictor(x)
                x = self.lpd(x, err)
                x = x.mean(dim=(1, 2, 3))
            return x
        else:
            """
            A JND predictor is trained to estimate a threshold of error strength.
            In the inference stage, however, thresholds for pixel-level content changes are required.
            The square root of the output of the JND predictor is therefore taken as the final JND threshold,
            """
            x = self._jnd_predictor(x)
            return x ** 0.5

    def _jnd_predictor(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        return x

    def _quality_predictor(self, jnd, err):
        x = self.lpd(jnd, err)
        x = self.avg_pool(x)
        b, c, h, w = x.shape
        x = x.view(b, -1)
        return self.globq(x)

    def _freeze_fc(self):
        for p in self.globq.parameters():
            p.requires_grad = False

    def _unfreeze_fc(self):
        for p in self.globq.parameters():
            p.requires_grad = True

    def _freeze_jnd_pred(self):
        for p in self.SFENet1.parameters():
            p.requires_grad = False
        for p in self.SFENet2.parameters():
            p.requires_grad = False
        for p in self.RDBs.parameters():
            p.requires_grad = False
        for p in self.GFF.parameters():
            p.requires_grad = False

    def _unfreeze_jnd_pred(self):
        for p in self.SFENet1.parameters():
            p.requires_grad = True
        for p in self.SFENet2.parameters():
            p.requires_grad = True
        for p in self.RDBs.parameters():
            p.requires_grad = True
        for p in self.GFF.parameters():
            p.requires_grad = True

    def _wfc(self):
        self.fc = True

    def _wofc(self):
        self.fc = False


# @register('rdn')
def make_rdn(out_dim=1, G0=32, RDNkSize=3, RDNconfig='D', keep_prob=0.1):
    args = Namespace()   # namespace is actually a dictionary    abandon params: img_size, hidden_list,
    # args.dims = img_size**2
    # args.hidden_list = hidden_list
    args.out_dim = out_dim
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig
    args.keep_prob = keep_prob

    args.n_colors = 3
    return RDN(args)


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