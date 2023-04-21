import torch
import os
import time
from tensorboardX import SummaryWriter
import shutil
import numpy as np
import random
import math


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_npy_file(filename):
    return filename.endswith('.npy')


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters of the above model [{}] is [{}] or [{:>.4f}M]".format(name, num_params,
                                                                                          num_params / 1e6))


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    # print('Initialize network with [{}]'.format(init_type))
    net.apply(init_func)


def create_folder(root_dir, path, version):
    if not os.path.exists(os.path.join(root_dir, path, version)):
        os.makedirs(os.path.join(root_dir, path, version))


_log_path = None


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def set_log_path(path):
    global _log_path
    _log_path = path


def set_log_path(path):
    global _log_path
    _log_path = path


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_log_path(path):
    global _log_path
    _log_path = path


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

    def sleep(self, x):
        time.sleep(x)

    def ts(self):
        now = time.localtime()
        return time.strftime('%Y%m%d%H%M', now)


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def convert_to_patches(img, patch_size=176, overlapping=32, mandatory=False):
    # timer = Timer()
    # ts = timer.t()
    [img_col, img_row] = img.size
    img = np.array(np.reshape(img.getdata(), (-1, img_row, img_col)), dtype='float32')
    """
    if len(np.shape(img)) == 3:
        if np.shape(img)[2] == 3:
            img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        elif np.shape(img)[2] == 1:
            img = np.array(img.squeeze(), dtype='float32')
        else:
            raise AssertionError('Image shape[2] should be 1 or 3, but [{}] is received'.format(np.shape(img)[2]))
    elif len(np.shape(img)) == 2:
        img = img
    else:
        raise AssertionError('Image should have 2 or 3 dimension, but [{}] is received'.format(np.shape(img)))"""

    img /= 255.
    max_row = patch_num(img_row, patch_size, overlapping, mandatory)
    max_col = patch_num(img_col, patch_size, overlapping, mandatory)
    step = patch_size - overlapping
    patches = []
    for i in range(max_row):
        for j in range(max_col):
            # judge the right most incomplete part
            if ((patch_size + j*step) > img_col and (patch_size + i*step) <= img_row):
                patch = img[:, i*step:i*step + patch_size, img_col-patch_size:img_col]  # j*step:img_col
            # judge the bottom most incomplete part
            elif ((patch_size + i*step) > img_row and (patch_size + j*step) <= img_col):
                patch = img[:, img_row-patch_size:img_row, j*step:j*step + patch_size]  # i*step:img_row
            elif ((patch_size + j*step) > img_col and (patch_size + i*step) > img_row):
                patch = img[:, img_row-patch_size:img_row, img_col-patch_size:img_col]  # i*step:img_row, j*step:img_col
            else:
                patch = img[:, i*step:i*step + patch_size, j*step:j*step + patch_size]
            patches.append(patch)
    return patches


def patch_num(img_size, patch_size=176, overlapping=32, mandatory=False):
    step = patch_size - overlapping
    max_size = float((img_size - patch_size)/step) + 1
    return math.ceil(max_size)if mandatory == False else math.floor(max_size)


def reconstruct2img(patches, img_row, img_col, patch_size=176, overlapping=32, mandatory=False):
    step = patch_size - overlapping
    o2 = int(overlapping / 2)
    max_row = patch_num(img_row, patch_size, overlapping, mandatory)
    max_col = patch_num(img_col, patch_size, overlapping, mandatory)
    img = np.zeros((img_row, img_col), dtype='float32')

    if isinstance(patches, list):
        patches = np.array(patches)

    for i in range(max_row):
        for j in range(max_col):
            # judge the right most incomplete part
            if ((patch_size + j * step) > img_col and (patch_size + i * step) <= img_row):
                img[i * step + int(i!=0)*o2:i * step + patch_size - o2, j * step + int(j!=0)*o2:img_col] = \
                    patches[i*max_col + j, int(i!=0)*o2:patch_size-int(i!=(max_row - 1))*o2, int(j!=0)*o2:patch_size-int(j!=(max_col - 1))*o2]
                # img[i * step:i * step + patch_size, j * step:img_col] = patches[i*max_col + j]
            # judge the bottom most incomplete part
            elif ((patch_size + i * step) > img_row and (patch_size + j * step) <= img_col):
                img[i * step + int(i!=0)*o2:img_row, j * step + int(j!=0)*o2:j * step + patch_size - o2] = \
                    patches[i*max_col + j, int(i!=0)*o2:patch_size-int(i!=(max_row - 1))*o2, int(j!=0)*o2:patch_size-int(j!=(max_col - 1))*o2]
                # img[i * step:img_row, j * step:j * step + patch_size] = patches[i*max_col + j]
            elif ((patch_size + j * step) > img_col and (patch_size + i * step) > img_row):
                img[i * step + int(i!=0)*o2:img_row, j * step + int(j!=0)*o2:img_col] = \
                    patches[i*max_col + j, int(i!=0)*o2:patch_size-int(i!=(max_row - 1))*o2, int(j!=0)*o2:patch_size-int(j!=(max_col - 1))*o2]
                # img[i * step:img_row, j * step:img_col] = patches[i*max_col + j]
            else:
                img[i * step + int(i!=0)*o2:i * step + patch_size - o2, j * step + int(j!=0)*o2:j * step + patch_size - o2] = \
                    patches[i*max_col + j, int(i!=0)*o2:patch_size-int(i!=(max_row - 1))*o2, int(j!=0)*o2:patch_size-int(j!=(max_col - 1))*o2]
                # img[i * step:i * step + patch_size, j * step:j * step + patch_size] = patches[i*max_col + j]
    return img


def random_mat(r, c):
    ran_mat = []
    length = r*c
    for i in range(length):
        ran_mat.append(random.random())
    ran_mat = np.reshape(ran_mat, [r, c])
    ran_mat[ran_mat < 0.5] = -1
    ran_mat[ran_mat >= 0.5] = 1
    return ran_mat


def bound(img):
    img[img > 1.] = 1.
    img[img < 0.] = 0.
    return img


def _cal_num_blocks(h, w, crop_size, shave):
    nh = math.ceil((h - shave) / (crop_size - shave))
    nw = math.ceil((w - shave) / (crop_size - shave))
    return nh, nw


def win2linux(win_path):
    return win_path.replace('\\', '/')