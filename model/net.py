import torch
import torch.nn as nn
from model.common import RDB, PercepJump


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