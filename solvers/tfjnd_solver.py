import torch
import torch.nn as nn
import utils
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import SGD, Adam, RMSprop
from tqdm import tqdm
from scipy import stats
from PIL import Image

from solvers.base_solver import BaseSolver
from configs.config import get_transform
from data.load_info import LoadInfo
from data._dataset import MyDataSet
from data._loader import make_loader
from model import make_rdn


class TFJNDSolver(BaseSolver):
    def __init__(self, config):
        super().__init__(config)
        self.val_sv_pth = config.val_save_dir
        self.val_data_pth = config.val_data_dir

        self.load_sd = config.load_sd
        self.init_type = config.init_type
        self.gain = config.gain
        self.optimizer_type = config.optimizer

        self.loss_type = config.loss_fn
        self.lr = config.lr
        self.beta1, self.beta2 = config.beta1, config.beta2
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.alpha = config.alpha

        self.lr_decay = config.lr_decay
        self.lr_decay_ratio = config.lr_decay_ratio
        self.lr_num_epochs_decay = config.lr_num_epochs_decay

        self.transforms = get_transform(config.to_tensor, config.resize, config.normalize, config.gray_scale,
                                   config.rand_crop, config.cent_crop,
                                   config.flip_h, config.flip_v, config.color_jitter)

        self.Info = LoadInfo(
            root_dir=config.data_root_dir,
            db_dict=config.db_dict,
            transform=self.transforms,
            enhance_level=config.enhance_level,
            db_key=config.db_key,
            data_enhance=config.data_enhance,
            tvr=config.tvr,
        )

        self.info_train = self.Info.get_info_train()
        self.info_val = self.Info.get_info_val()

        self.dataset = MyDataSet(self.info_train)
        self.dataset_val = MyDataSet(self.info_val)

        self.train_loader = make_loader(
            dataset=self.dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            drop_last=config.drop_last,
        )
        self.val_loader = make_loader(
            dataset=self.dataset_val,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        if self.use_gpu:
            self.model = make_rdn(out_dim=config.out_dim,
                                  G0=config.grow_rate0, RDNkSize=config.k_size, RDNconfig=config.block_style,
                                  keep_prob=config.keep_prob).cuda()
        else:
            self.model = make_rdn(out_dim=config.out_dim,
                                  G0=config.grow_rate0, RDNkSize=config.k_size, RDNconfig=config.block_style,
                                  keep_prob=config.keep_prob)

        if self.phase == 'train':
            self.prepare_training()
        elif self.phase == 'test':
            self.prepare_test()
        else:
            raise NotImplementedError('phase can only be "train" or "test", but [{}] was received'.format(self.phase))

    def prepare_training(self):

        if self.is_print_network:
            # utils.print_network(self.model, 'RDN')
            self.print_network()

        if self.init_type:
            utils.init_weights(self.model, self.init_type, self.gain)
            self.log('Parameters init type: [{}]'.format(self.init_type))

        if self.optimizer_type == 'adam':
            self.optimizer = Adam(params=self.model.parameters(), lr=self.lr, betas=[self.beta1, self.beta2],
                                  weight_decay=self.weight_decay)
            self.log('Optimizer: [{}], lr:[{}], beta1:[{}], beta2:[{}]'.format(self.optimizer_type, self.lr,
                                                                               self.beta1, self.beta2))
        elif self.optimizer_type == 'sgd':
            # optimizer = SGD(params=model.parameters(), lr=config.lr, alpha=config.alpha)
            self.optimizer = SGD(params=self.model.parameters(), lr=self.lr,
                                 momentum=self.momentum, weight_decay=self.weight_decay, nesterov=False)
            self.log('Optimizer: [{}], lr:[{}], momentum:[{}], weight_decay:[{}]'.format(self.optimizer_type, self.lr,
                                                                                         self.momentum,
                                                                                         self.weight_decay))
        elif self.optimizer_type == 'rmsp':
            self.optimizer = RMSprop(params=self.model.parameters(), lr=self.lr, alpha=self.alpha,
                                          weight_decay=self.weight_decay, momentum=self.momentum, centered=False)
            self.log('Optimizer: [{}], lr:[{}], alpha:[{}]'.format(self.optimizer_type, self.lr, self.alpha))
        else:
            raise NotImplementedError("Optimizer [{}] is not found".format(self.optimizer_type))

        if self.lr_decay:
            # def lambda_rule(epoch):
            #     return 1.0 - max(0, epoch + 1 - config.lr_num_epochs_decay) / config.lr_decay_ratio
            # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            _lambda = lambda epoch: self.lr_decay_ratio ** (epoch // self.lr_num_epochs_decay)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lambda)
            print("=== Set learning rate decay policy for the model ===")

        else:
            self.lr_scheduler = None

        self.log('model: #params={}'.format(utils.compute_num_params(self.model, text=True)))

        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError('Loss function [{}] is not found'.format(self.loss_type))
        self.log('Loss function: [{}]'.format(self.loss_type))

        # only recover the model, but the optimizer.
        if self.load_sd:
            self.load()
        else:
            self.epoch_start = 0

        if self.multiple_gpu and self.n_gpus > 1:
            self.model = nn.parallel.DataParallel(self.model)

    def train(self):
        self.model.train()
        train_loss = utils.Averager()
        preds = []
        labels = []
        try:
            with tqdm(self.train_loader, leave=False, desc='train') as tqdm_range:
                for b_ref, b_dst, b_label in tqdm_range:
                    b, _, h, w = b_ref.shape
                    b_ref = b_ref.view(-1, 3, h, w)
                    b_dst = b_dst.view(-1, 1, h, w)

                    if self.use_gpu:
                        pred = self.model.forward(b_ref.cuda(), b_dst.cuda(), training=True)
                        # b_ref = b_ref.cuda()
                        # b_dst = b_dst.cuda()
                        b_label = b_label.float()
                    else:
                        pred = self.model.forward(b_ref, b_dst, training=True)
                        b_label = b_label.float()

                    pred = pred.view(b, -1).mean(dim=1)

                    if self.use_gpu:
                        loss = self.loss_fn(pred, b_label.cuda())
                    else:
                        loss = self.loss_fn(pred, b_label)

                    train_loss.add(loss.cpu().item())

                    reg = 0.
                    for param in self.model.parameters():
                        reg += torch.sum(torch.square(param))
                    loss += 0.0001 * reg

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pred_n = pred.detach().cpu().numpy()
                    preds.extend(pred_n)
                    labels.extend(b_label.numpy())
                    pred = None
                    loss = None
        except KeyboardInterrupt:
            tqdm_range.close()
            sys.exit(0)
        tqdm_range.close()

        if self.lr_scheduler:
            self.lr_scheduler.step()
        return train_loss.item(), preds, labels

    def qa_val(self):
        print('\n======Quality Assessment Validation======')
        self.model.eval()
        preds = []
        labels = []
        try:
            with tqdm(self.val_loader, leave=False, desc='qa_val') as tqdm_range:
                for b_ref, b_dst, b_label in tqdm_range:
                    b, _, h, w = b_ref.shape
                    b_ref = b_ref.view(-1, 3, h, w)
                    b_dst = b_dst.view(-1, 1, h, w)
                    if self.use_gpu:
                        b_ref, b_dst = b_ref.cuda(), b_dst.cuda()

                    pred = self.model.forward(b_ref, b_dst, training=True)
                    pred = pred.view(b, -1).mean(dim=1).float().cpu()

                    preds.extend(pred)
                    labels.extend(b_label)
        except KeyboardInterrupt:
            tqdm_range.close()
            sys.exit(0)
        tqdm_range.close()

        preds = np.reshape(np.array(preds), (-1, 1))
        labels = np.reshape(np.array(labels), (-1, 1))

        plcc = stats.pearsonr(preds, labels)[0][0]
        srocc = stats.spearmanr(preds, labels)[0]
        krocc = stats.stats.kendalltau(preds, labels)[0]

        self.model.train()
        return plcc, srocc, krocc

    def val(self, epoch):
        print('\n======Validation======')
        self.model.eval()
        ep = 'epoch{}'.format(epoch)
        sd = self.val_sv_pth + '/' + ep
        if not os.path.exists(sd):
            os.mkdir(sd)

        for name in os.listdir(self.val_data_pth):
            img_path = os.path.join(self.val_data_pth, name)
            img = Image.open(img_path)
            [W, H] = img.size

            if np.mod(W, 4) != 0 or np.mod(H, 4) != 0:
                W = W - np.mod(W, 4)
                H = H - np.mod(H, 4)
                img = img.crop((0, 0, W, H))
            ref_img = np.reshape(np.array(img.getdata()), [H, W, -1]).copy()

            img = self.transforms(img).unsqueeze(0)
            if torch.cuda.is_available:
                img = img.cuda()
            pred = self.model.forward(img).squeeze(0).squeeze(0).cpu().numpy()
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
                raise AssertionError(
                    'The 3th dimension should be 1 or 3, but [{}] is received'.format(ref_img.shape[2]))

            jnd_mse = np.mean(pred ** 2)
            alpha = 12 / np.sqrt(jnd_mse + 1e-10)
            dst_img = utils.bound((ref_img + alpha * _pred * ran_mat) / 255.)
            dst_mse = np.mean((dst_img - ref_img / 255.) ** 2)
            dst_psnr = int(10 * math.log10(1 / (dst_mse + 1e-10)))
            dst_name = sd + '/' + 'psnr_' + str(dst_psnr) + name
            plt.imsave(dst_name, dst_img)

        self.model.train()

    def save_checkpoint(self, epoch, epoch_save):
        if self.n_gpus > 1 and self.multiple_gpu:
            model_ = self.model.module
        else:
            model_ = self.model

        self.model_spec['name'] = 'RDN-JND'
        self.model_spec['dict'] = model_.state_dict()
        self.optimizer_spec['name'] = self.optimizer_type
        self.optimizer_spec['dict'] = self.optimizer.state_dict()

        sv_file = {
            'model': self.model_spec,
            'optimizer': self.optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(self.model_sv_pth, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(self.model_sv_pth, 'epoch-{}.pth'.format(epoch)))

    def load(self):
        if os.path.exists(os.path.join(self.model_sv_pth, 'epoch-last.pth')):
            sv_file = torch.load(os.path.join(self.model_sv_pth, 'epoch-last.pth'))
            self.model.load_state_dict(sv_file['model']['dict'])
            self.optimizer.load_state_dict(sv_file['optimizer']['dict'])
            self.epoch_start = sv_file['epoch'] + 1
            self.log('Model state dict has been recovered from [{}]'.format(
                os.path.join(self.model_sv_pth, 'epoch-last.pth')))

        else:
            raise AssertionError(
                'sv_file [{}] is not found'.format(os.path.join(self.model_sv_pth, 'epoch-last.pth')))

    def print_network(self):
        """Print out the network information."""
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print(self.model)
        print("The number of parameters of the above model is [{}] or [{:>.4f}M]".format(num_params,
                                                                                         num_params / 1e6))

    def overlap_crop_forward(self, x, shave=10, min_size=100000):
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
                pred_batch = self.model.forward(x_batch)

                pred_list.extend(pred_batch.chunk(n_GPUs, dim=0))
        else:
            pred_list = [
                self.overlap_crop_forward(patch, shave=shave, min_size=min_size) \
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

    def _forward(self, x):
        with torch.no_grad():
            pred = self.model.forward(x)
        return pred

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))

        return s, n