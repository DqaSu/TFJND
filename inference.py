import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
import argparse
import scipy.io as scio
import torch.nn as nn
import torchvision.transforms as transforms
import time

import utils

from PIL import Image
from model import make_rdn
from tqdm import tqdm


DB = ['csiq', 'live']
transform = transforms.ToTensor()


def predictor_ori(img_path, model, device='gpu', patch_size=176, overlapping=32):
    img = Image.open(img_path)
    [W, H] = img.size
    patches = utils.convert_to_patches(img, patch_size, overlapping)
    patches = np.stack(patches, axis=0)
    patches_tensor = torch.FloatTensor(patches)

    if len(patches_tensor.shape) < 4:
        # for the situation that input image can not be divided into patches
        patches_tensor = patches_tensor.unsqueeze(0)

    with torch.no_grad():
        if device == 'cpu':
            preds = model.forward(patches_tensor).squeeze(1)
        elif device == 'gpu':
            preds = model.forward(patches_tensor.cuda()).squeeze(1).cpu()
        else:
            raise AssertionError('device should be either [cpu] or [gpu], but [{}] is received'.format(device))

    jnd_profile = np.array(utils.reconstruct2img(preds, H, W, patch_size, overlapping), dtype=object)
    return jnd_profile


def predictor(img_path, model, device='gpu'):
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)
    if device == 'gpu':
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        pred = overlap_crop_forward(model, img_tensor).squeeze(0).squeeze(0).cpu()

    jnd_profile = pred.numpy()
    return jnd_profile


def overlap_crop_forward(model, x, shave=10, min_size=100000):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 4
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    x_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        pred_list = []
        for i in range(0, 4, n_GPUs):
            x_batch = torch.cat(x_list[i:(i + n_GPUs)], dim=0)
            pred_batch = model.forward(x_batch)

            pred_list.extend(pred_batch.chunk(n_GPUs, dim=0))
    else:
        pred_list = [
            overlap_crop_forward(model, patch, shave=shave, min_size=min_size) \
            for patch in x_list
        ]

    output = x.new(b, 1, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = pred_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = pred_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = pred_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = pred_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Just noticeable difference estimation.')
    parser.add_argument('-image_pth',
                        type=str,
                        default="/mnt/diskplus/myw/database/CSIQ/src_imgs",
                        help="Path of images to be processed.")
    parser.add_argument('-model_pth',
                        type=str,
                        default="/mnt/diskplus/myw/TFJND/results/csiq/epoch-last.pth",
                        help="Path of the pre-trained JND predictor")
    parser.add_argument('-device',
                        type=str,
                        default='gpu')

    opt = parser.parse_args()
    image_pth = opt.image_pth
    model_pth = opt.model_pth
    device = opt.device
    save_pth = ''

    for db in DB:
        if model_pth.find(db) >= 0:
            save_pth = './inference_results/LPD' + db
            if not os.path.exists(save_pth):
                os.mkdir(save_pth)
            break

    model = make_rdn(out_dim=1, G0=64, RDNkSize=3, RDNconfig='D', keep_prob=0.1)

    if os.path.exists(model_pth):
        sv_file = torch.load(model_pth)
        model.load_state_dict(sv_file['model']['dict'])
    else:
        raise AssertionError('sv_file [{}] is not found'.format(model_pth))

    if device == 'gpu':
        model = model.cuda()
        model = nn.parallel.DataParallel(model)
    elif device == 'cpu':
        model = model
    else:
        raise AssertionError('device should be either [cpu] or [gpu], but [{}] is received'.format(device))

    if not os.path.isdir(image_pth):
        jnd = predictor(image_pth, model, device=device)
        sd = {'tfjnd': jnd}
        scio.savemat('test_jnd.mat', sd)
    else:
        image_pths = []
        for dir_pth, _, fnames in sorted(os.walk(image_pth)):
            for fname in sorted(fnames):
                if utils.is_image_file(fname):
                    pth = os.path.join(dir_pth, fname)
                    image_pths.append(pth)
        assert image_pths, "[%s] has no valid image file" % image_pth

        times = []
        for img_pth in tqdm(image_pths):
            t1 = time.time()
            jnd_profile = predictor(img_pth, model, device=device)
            t2 = time.time()
            t = t2 - t1
            times.append(t)
            sd = {'tfjnd': jnd_profile}
            sp = os.path.join(save_pth, os.path.basename(img_pth)[:-4] + '.mat')
            # np.save(sp, jnd_profile)
            scio.savemat(sp, sd)

        avg_t = np.array(times).mean()
        print("Avg Time Per Image: {}".format(avg_t))


