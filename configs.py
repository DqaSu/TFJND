# -*-coding:utf-8-*-

import argparse
from torchvision import transforms
# from utils import str2bool

img_size = 256


def get_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--load_sd', action='store_true', default=False, help='load save dictionary, default: False')
    parser.add_argument('--shuffle', action='store_false', default=True, help='shuffle when load dataset, default: True')
    parser.add_argument('--drop_last', action='store_true', default=False, help=' drop the last incomplete batch, default: False')
    parser.add_argument('--image_size', type=int, default=img_size, help='image load resolution')
    parser.add_argument('--version', type=str, default='FR-IQA', help='JND|FR-IQA')
    parser.add_argument('--init_type', type=str, default='kaiming', help='normal|xavier|kaiming|orthogonal')
    parser.add_argument('--grow_rate0', type=int, default=64)
    parser.add_argument('--k_size', type=int, default=3)
    parser.add_argument('--block_style', type=str, default='D')
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--keep_prob', type=float, default=0.1)

    # transform
    parser.add_argument('--to_tensor', action='store_false', default=True, help='transform data to torch tensor, default: True')
    parser.add_argument('--gray_scale', action='store_true', default=False, help='transform data to torch tensor, default: True')
    parser.add_argument('--rand_crop', action='store_true', default=False, help='transform data to torch tensor, default: True')
    parser.add_argument('--resize', action='store_true', default=False, help='transform resize, default: False')
    parser.add_argument('--normalize', action='store_true', default=False, help='transform data to torch tensor, default: False')
    parser.add_argument('--cent_crop', action='store_true', default=False, help='transform data to torch tensor, default: False')
    parser.add_argument('--flip_h', action='store_true', default=False, help='transform data to torch tensor, default: False')
    parser.add_argument('--flip_v', action='store_true', default=False, help='transform data to torch tensor, default: False')
    parser.add_argument('--color_jitter', action='store_true', default=False, help='transform data to torch tensor, default: False')

    # Training configuration.
    parser.add_argument('--pretrained_model', type=int, default=None)  # pretrained
    parser.add_argument('--epoch_max', type=int, default=800, help='total epochs to update the generator')
    parser.add_argument('--epoch_val', type=int, default=10, help='epochs interval to val the JND performance')
    parser.add_argument('--epoch_save', type=int, default=50, help='epochs interval to save model')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch size')  # 20
    parser.add_argument('--num_workers', type=int, default=4, help='subprocesses to use for data loading')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='setup learning rate decay schedule')
    parser.add_argument('--lr_num_epochs_decay', type=int, default=5,
                        help='LambdaLR: epoch at starting learning rate')  # half 20（50）
    parser.add_argument('--lr_decay_ratio', type=float, default=0.99,
                        help='LambdaLR: ratio of linearly decay learning rate to zero')  # half 20（50）
    parser.add_argument('--optimizer', type=str, default='adam', help='adam|rmsp|sgd')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--alpha', type=float, default=0.99, help='alpha for rmsprop optimizer')
    parser.add_argument('--gain', type=float, default=0.02)
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay for SGD and RMSprop')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD and RMSprop')
    parser.add_argument('--loss_fn', type=str, default='l2', help='l1|l2')

    # Directories.
    parser.add_argument('--data_root_dir', type=str, default='/mnt/diskplus/myw/database')
    parser.add_argument('--save_log_dir', type=str, default='./results/log')
    parser.add_argument('--save_model_dir', type=str, default='./results/models')
    parser.add_argument('--db_dict', type=str, default={'CSIQ':{'name': 'CSIQ', 'info': 'csiq_info'},
                                                        'TID':{'name': 'TID2013', 'info': 'tid2013_info'},
                                                        'LIVE':{'name': 'LIVE', 'info': 'live_info'}})
    parser.add_argument('--db_key', type=str, default='CSIQ', help='CSIQ|TID|LIVE')
    parser.add_argument('--data_enhance', action='store_true', default=False)
    parser.add_argument('--enhance_level', type=int, default=5, help='A int value from 1 to 5 is needed')
    parser.add_argument('--tvr', type=float, default=0.9, help='Ratio of train set to val set')

    # valuation
    parser.add_argument('--val_data_dir', type=str, default='./val/data')
    parser.add_argument('--val_save_dir', type=str, default='./val/results')
    parser.add_argument('--patch_size', type=int, default=176)
    parser.add_argument('--overlapping', type=int, default=32)

    # Misc
    parser.add_argument('--parallel', action='store_true', default=False, help='use noise_level-GPU for training')
    parser.add_argument('--is_print_network', action='store_true', default=False)
    parser.add_argument('--multiple_gpu', action='store_false', default=True)
    parser.add_argument('--gpu_ids', default=[0, 1, 2, 3])
    parser.add_argument('--gpu_id', type=str, default="3")
    return parser.parse_args()


def get_transform(to_tensor=True, resize=False, normalize=False, gray_scale=False,
                  rand_crop=False, cent_crop=False,
                  flip_h=False, flip_v=False, color_jitter=False):

    options = []
    # preprocess
    if resize:
        options.append(transforms.Resize(size=img_size))
    if normalize:
        options.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    if gray_scale:
        options.append(transforms.Grayscale())
    # image crop
    if rand_crop:
        options.append(transforms.RandomCrop(size=img_size))
    if cent_crop:
        options.append(transforms.CenterCrop(size=img_size))
    # image augmentation
    if flip_h:
        options.append(transforms.RandomHorizontalFlip(p=0.5))
    if flip_v:
        options.append(transforms.RandomVerticalFlip(p=0.5))
    if color_jitter:
        options.append(transforms.ColorJitter(contrast=.15))
    if to_tensor:
        options.append(transforms.ToTensor())
    transform = transforms.Compose(options)
    return transform