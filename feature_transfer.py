import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
from PIL import Image, ImageFile


c1 = 1.0001
c2 = 1.0001


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
        return x


def feature_trans(feature_map, rx, dx, model='MAE'):

    if model == 'MSE':
        pix_error = (rx - dx).pow(2)
    elif model == 'MAE':
        pix_error = (rx - dx).abs()
    else:
        raise NotImplementedError('Initialization method [{}] is not implemented'.format(model))
    error_map = GaussianBlurConv(pix_error)
    err_visibility = torch.log(error_map + c1) / torch.log(feature_map + c2)
    return err_visibility



