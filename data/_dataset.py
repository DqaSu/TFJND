import os
import torch
import random
import scipy.io as scio
from torch.utils import data
from PIL import Image, ImageFile
from torchvision import transforms


crop_size = [256, 256]


class MyDataSet(data.Dataset):
    # def __init__(self, info_file, root_dir, transform):
    def __init__(self, info):

        super(MyDataSet, self).__init__()

        self.root_dir = info['root_dir']
        self.transform = info['transform']
        self.db = info['db']
        self.ref_name = info['ref_name']
        self.dst_name = info['dst_name']
        self.label = info['label']
        self.db_size = len(info['label'])
        self.rand_crop = transforms.RandomCrop(crop_size)
        self.v_flip = transforms.RandomHorizontalFlip()
        self.h_flip = transforms.RandomVerticalFlip()
        self.patch_num = 4

    def __len__(self):
        return self.db_size

    def __getitem__(self, item):
        # Allow to truncate images with huge size
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        ref_path = os.path.join(self.root_dir, self.db, self.ref_name[item])
        dst_path = os.path.join(self.root_dir, self.db, self.dst_name[item])

        ref_img = Image.open(ref_path)
        dst_img = scio.loadmat(dst_path)['mean_square_error']
        # dst_img = Image.open(dst_path)
        label = self.label[item]

        # crop
        ref_img = self.transform(ref_img)
        dst_img = self.transform(dst_img)

        seed = random.randint(1, 10000)
        ref_patches = []
        dst_patches = []

        if self.transform is not None:
            random.seed(seed)
            for _ in range(self.patch_num):
                ref_patches.append(self.h_flip(self.v_flip(self.rand_crop(ref_img))))
            random.seed(seed)
            for _ in range(self.patch_num):
                dst_patches.append(self.h_flip(self.v_flip(self.rand_crop(dst_img))))

        ref_patches = torch.cat(tuple(ref_patches), 0)
        dst_patches = torch.cat(tuple(dst_patches), 0)
        return ref_patches, dst_patches, label