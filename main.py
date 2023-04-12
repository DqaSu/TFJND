import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import SGD, Adam
from torch.optim.rmsprop import RMSprop
from PIL import Image
import sys

import utils
from scipy import stats
from data_generator import MyDataSet, LoadInfo
from Loader import make_loader
from configs import get_config, get_transform
from model import make_rdn, PercepJump


def main(config):
    # for fast training.
    torch.backends.cudnn.benchmark = True
    global log, writer

    log_save_path = config.save_log_dir
    model_save_path = config.save_model_dir
    log, writer = utils.set_save_path(log_save_path)

    model_spec = {}
    optimizer_spec = {}
    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    timer = utils.Timer()

    transforms = get_transform(config.to_tensor, config.resize, config.normalize, config.gray_scale,
                               config.rand_crop, config.cent_crop,
                               config.flip_h, config.flip_v, config.color_jitter)

    Info = LoadInfo(
        root_dir=config.data_root_dir,
        db_dict=config.db_dict,
        transform=transforms,
        enhance_level=config.enhance_level,
        db_key=config.db_key,
        data_enhance=config.data_enhance,
        tvr=config.tvr,
    )

    info_train = Info.get_info_train()
    info_val = Info.get_info_val()

    dataset = MyDataSet(info_train)
    dataset_val = MyDataSet(info_val)

    train_loader = make_loader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )
    val_loader = make_loader(
        dataset=dataset_val,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model, optimizer, lr_scheduler, epoch_start, loss_fn = prepare_training(config)

    if config.epoch_val is not None:
        t_str = timer.ts()
        val_save_path = config.val_save_dir + '/' + t_str
        os.mkdir(val_save_path)

    epoch_max = config.epoch_max
    epoch_val = config.epoch_val
    epoch_save = config.epoch_save

    exit()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        train_loss, preds, labels = train(train_loader, model, optimizer, loss_fn, 'train')
        log_info.append('train| loss: {:.4f}'.format(train_loss))

        if lr_scheduler is not None:
            lr_scheduler.step()

        if n_gpus > 1 and config.multiple_gpu:
            model_ = model.module
        else:
            model_ = model

        model_spec['name'] = 'RDN-JND'
        model_spec['dict'] = model_.state_dict()
        optimizer_spec['name'] = config.optimizer
        optimizer_spec['dict'] = optimizer.state_dict()

        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(model_save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(model_save_path, 'epoch-{}.pth'.format(epoch)))

        with torch.no_grad():
            log_info, plcc, srocc, krocc = qa_val(val_loader, model_, log_info)

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            with torch.no_grad():
                val(config.val_data_dir, val_save_path, model_, epoch, transforms, config.patch_size, config.overlapping)


def prepare_training(config):
    model = make_rdn(out_dim=config.out_dim,
                     G0=config.grow_rate0, RDNkSize=config.k_size, RDNconfig=config.block_style,
                     keep_prob=config.keep_prob).cuda()
    # img_size=config.image_size, hidden_list=config.hidden_list,   abandon params

    if config.is_print_network:
        utils.print_network(model, 'RDN')

    if config.init_type:
        utils.init_weights(model, config.init_type, config.gain)
        log('Parameters init type: [{}]'.format(config.init_type))

    if config.optimizer == 'adam':
        optimizer = Adam(params=model.parameters(), lr=config.lr, betas=[config.beta1, config.beta2], weight_decay=config.weight_decay)
        log('Optimizer: [{}], lr:[{}], beta1:[{}], beta2:[{}]'.format(config.optimizer, config.lr, config.beta1, config.beta2))
    elif config.optimizer == 'sgd':
        # optimizer = SGD(params=model.parameters(), lr=config.lr, alpha=config.alpha)
        optimizer = SGD(params=model.parameters(), lr=config.lr,
                        momentum=config.momentum, weight_decay=config.weight_decay, nesterov=False)
    elif config.optimizer == 'rmsp':
        optimizer = RMSprop(params=model.parameters(), lr=config.lr, alpha=config.alpha,
                            weight_decay=config.weight_decay, momentum=config.momentum, centered=False)
    else:
        raise NotImplementedError("Optimizer [{}] is not found".format(config.optimizer_type))
    # log('Optimizer: [{}]'.format(config.optimizer))

    if config.lr_decay:
        # def lambda_rule(epoch):
        #     return 1.0 - max(0, epoch + 1 - config.lr_num_epochs_decay) / config.lr_decay_ratio
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        _lambda = lambda epoch: config.lr_decay_ratio ** (epoch // config.lr_num_epochs_decay)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lambda)
        print("=== Set learning rate decay policy for RDN ===")

    else:
        lr_scheduler = None

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    if config.loss_fn == 'l1':
        loss_fn = nn.L1Loss()
    elif config.loss_fn == 'l2':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError('Loss function [{}] is not found'.format(config.loss_fn))
    log('Loss function: [{}]'.format(config.loss_fn))

    # only recover the model, but the optimizer.
    if config.load_sd:
        if os.path.exists(os.path.join(config.save_model_dir, 'epoch-last.pth')):
            sv_file = torch.load(os.path.join(config.save_model_dir, 'epoch-last.pth'))
            model.load_state_dict(sv_file['model']['dict'])
            optimizer.load_state_dict(sv_file['optimizer']['dict'])
            epoch_start = sv_file['epoch']+1
            log('Model state dict has been recovered from [{}]'.format(os.path.join(config.save_model_dir, 'epoch-last.pth')))

        else:
            raise AssertionError('sv_file [{}] is not found'.format(os.path.join(config.save_model_dir, 'epoch-last.pth')))
    else:
        epoch_start = 0

    if config.multiple_gpu:
        n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if n_gpus > 1:
            model = nn.parallel.DataParallel(model)

    return model, optimizer, lr_scheduler, epoch_start, loss_fn


def train(loader, model, optimizer, loss_fn, desc):
    model.train()
    train_loss = utils.Averager()
    preds = []
    labels = []
    try:
        with tqdm(loader, leave=False, desc=desc) as tqdm_range:
            for b_ref, b_dst, b_label in tqdm_range:
                # if crop
                # b, c*patch_n, h, w
                b, _, h, w = b_ref.shape
                b_ref = b_ref.view(-1, 3, h, w)
                b_dst = b_dst.view(-1, 1, h, w)

                pred = model.forward(b_ref.cuda(), b_dst.cuda(), training=True)

                pred = pred.view(b, -1).mean(dim=1)
                loss = loss_fn(pred, b_label.float().cuda())

                train_loss.add(loss.cpu().item())

                reg = 0.
                for param in model.parameters():
                    reg += torch.sum(torch.square(param))
                loss += 0.0001 * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_n = pred.detach().cpu().numpy()
                preds.extend(pred_n)
                labels.extend(b_label.numpy())
                pred = None
                loss = None
    except KeyboardInterrupt:
        tqdm_range.close()
        sys.exit(0)
    tqdm_range.close()
    return train_loss.item(), preds, labels


def qa_val(loader, model, log_info):
    model.eval()
    preds = []
    labels = []
    try:
        with tqdm(loader, leave=False, desc='val') as tqdm_range:
            for b_ref, b_dst, b_label in tqdm_range:
                b, _, h, w = b_ref.shape
                b_ref = b_ref.view(-1, 3, h, w)
                b_dst = b_dst.view(-1, 1, h, w)
                pred = model.forward(b_ref.cuda(), b_dst.cuda(), training=True)
                pred = pred.view(b, -1).mean(dim=1).float().cpu()

                preds.extend(pred)
                labels.extend(b_label)
    except KeyboardInterrupt:
        tqdm_range.close()
    tqdm_range.close()

    preds = np.reshape(np.array(preds), (-1, 1))
    labels = np.reshape(np.array(labels), (-1, 1))

    plcc = stats.pearsonr(preds, labels)[0][0]
    srocc = stats.spearmanr(preds, labels)[0]
    krocc = stats.stats.kendalltau(preds, labels)[0]

    log_info.append('plcc: {:.4f}'.format(plcc))
    log_info.append('srocc: {:.4f}'.format(srocc))
    log_info.append('krocc: {:.4f}'.format(krocc))
    model.train()
    return log_info, plcc, srocc, krocc


def val(data_dir, save_dir, model, epoch, transforms, patch_size, overlapping):
    print('\n======Validation======')
    model.eval()
    ep = 'epoch{}'.format(epoch)
    sd = save_dir + '/' + ep
    if not os.path.exists(sd):
        os.mkdir(sd)

    for name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, name)
        img = Image.open(img_path)
        [W, H] = img.size

        if np.mod(W, 4)!=0 or np.mod(H, 4)!=0:
            W = W - np.mod(W, 4)
            H = H - np.mod(H, 4)
            img = img.crop((0, 0, W, H))
        ref_img = np.reshape(np.array(img.getdata()), [H, W, -1]).copy()

        img = transforms(img).unsqueeze(0)
        if torch.cuda.is_available:
            img = img.cuda()
        pred = model.forward(img).squeeze(0).squeeze(0).cpu().numpy()
        pred = np.array(pred)

        if name[-3:] == 'bmp':
            name = name.replace('.bmp', '.png')
        jnd_name = sd + '/' + 'jnd_' + name
        plt.imsave(jnd_name, pred, cmap='gray')

        ran_mat = np.reshape(utils.random_mat(H, W), [H, W, 1])

        if ref_img.shape[2] == 3:
            _pred = np.expand_dims(pred, 2).repeat(3, 2)
            ran_mat = ran_mat.repeat(3, 2)
        elif ref_img.shape[2] == 1:
            _pred = np.expand_dims(pred, 2)
        else:
            raise AssertionError('The 3th dimension should be 1 or 3, but [{}] is received'.format(ref_img.shape[2]))

        jnd_mse = np.mean(pred ** 2)
        alpha = 12 / np.sqrt(jnd_mse + 1e-10)
        dst_img = utils.bound((ref_img + alpha * _pred * ran_mat) / 255.)
        dst_mse = np.mean((dst_img - ref_img / 255.) ** 2)
        dst_psnr = int(10 * math.log10(1 / (dst_mse + 1e-10)))
        dst_name = sd + '/' + 'psnr_' + str(dst_psnr) + name
        plt.imsave(dst_name, dst_img)

    model.train()
    print('The mean of JND pred is : [{}]'.format(np.mean(pred)))


if __name__ == '__main__':

    config = get_config()

    if config.is_print_network:
        print(config)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if torch.cuda.is_available():
        print('Found [{}] gpu'.format(torch.cuda.device_count()))
        gpu_id = torch.cuda.current_device()
        print('gpu id: [{}], device name: [{}]'.format(gpu_id, torch.cuda.get_device_name(gpu_id)))
    main(config)