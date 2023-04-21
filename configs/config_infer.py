# -*-coding:utf-8-*-
import argparse


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-image_pth',
                        type=str,
                        default="/mnt/diskplus/myw/database/CSIQ/src_imgs",
                        help="Path of images to be processed, can be image path or dictionary.")
    parser.add_argument('-model_pth',
                        type=str,
                        default="/mnt/diskplus/myw/TFJND/results/csiq/epoch-last.pth",
                        help="Path of the pre-trained JND predictor")
    parser.add_argument('-device',
                        type=str,
                        default='gpu')

    parser.add_argument('--is_print_network', action='store_true', default=False)
    parser.add_argument('--multiple_gpu', action='store_false', default=True)
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='use noise_level-GPU for training')
    parser.add_argument('--gpu_ids', default=[0, 1, 2, 3],
                        help='GPU ids for multi-GPU training')
    parser.add_argument('--gpu_id', type=str, default="3",
                        help='chose gpu id for single GPU training')

    return parser.parse_args()