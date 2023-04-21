import os
import numpy as np
import utils
import random
import scipy.io as scio
from torchvision import transforms


class LoadInfo():
    def __init__(self, root_dir, db_dict, transform=None, enhance_level=5, db_key='CSIQ', data_enhance=False, tvr=0.9):
        self.root_dir = root_dir
        self.db_key = db_key
        self.transform = transform
        self.data_enhance = data_enhance
        self.enhance_level = enhance_level
        self.tvr = tvr

        self.db = db_dict[self.db_key]['name']
        self.info_name = db_dict[self.db_key]['info']
        self.db_size = 0

        self.ref_name = []
        self.dst_name = []
        self.label = []
        self.org_label = []
        self.load_info()
        self.info_train, self.info_val = self.divide_info_org()
        if data_enhance:
            self.info_train = self.data_aug(self.info_train)
            self.info_val = self.data_aug(self.info_val)

    def load_info(self):
        info_path = os.path.join(self.root_dir, self.db, self.info_name)
        info = scio.loadmat(info_path)

        if info['mos'] is not None:
            self.db_size = len(info['mos'])
        else:
            raise NameError('There is no key-word of mos in info.')

        if self.db == 'TID2013':
            for i in range(self.db_size):
                if info['noise_idx'][i].item() not in range(14, 18):
                    self.ref_name.append(utils.win2linux(info['ref_name'][i].item().item()))
                    self.dst_name.append(utils.win2linux(info['dst_name'][i].item().item()))
                    self.label.append(info['mos'][i].item())  # dmos  1
                    self.org_label.append(int(info['org_label'][i].item()))
                    self.db_size = len(self.label)
        else:
            for i in range(self.db_size):
                if info['noise_idx'][i].item() != 6:
                    self.ref_name.append(utils.win2linux(info['ref_name'][0][i].item()))
                    dst_name = utils.win2linux(info['dst_name'][0][i].item())
                    if self.db == 'CSIQ':
                        iname = dst_name.split('/')[-1]
                    elif self.db == 'LIVE':
                        iname = dst_name
                    else:
                        raise NotImplementedError

                    if iname[-3:] == 'png':
                        iname = iname.replace('.png', '.mat')
                    elif iname[-3:] == 'bmp':
                        iname = iname.replace('.bmp', '.mat')
                    else:
                        raise NotImplementedError
                    self.dst_name.append(os.path.join('errors', iname))

                    self.label.append(info['mos'][i].item())  # dmos  1
                    self.org_label.append(int(info['org_label'][i].item()))

            self.db_size = len(self.dst_name)

        if self.db == 'LIVE':
            for i in range(self.db_size):
                self.label[i] = self.label[i] / 100

        elif self.db == 'TID2013':
            for i in range(self.db_size):
                self.label[i] = 1 - self.label[i] / 9

        print('Dataset with size [{}] has been loaded'.format(self.db_size))

    def divide_info_org(self):
        # divide dataset according to reference image
        orgN = np.max(self.org_label)
        orgNv = int(np.ceil(orgN * (1 - self.tvr)))
        orgset = [i for i in range(1, orgN+1)]

        random.seed(1)
        random.shuffle(orgset)
        orgsett = orgset[:-orgNv]
        orgsetv = orgset[-orgNv:]
        print('Validation org:')
        print(orgsetv)
        indext = []
        indexv = []
        for i in range(len(self.org_label)):
            if self.org_label[i] in orgsett:
                indext.append(i)
            else:
                indexv.append(i)

        info_train = {}
        info_val = {}

        info_train['ref_name'] = []
        info_train['dst_name'] = []
        info_train['label'] = []
        info_train['name'] = 'train'
        info_train['root_dir'] = self.root_dir
        info_train['transform'] = self.transform
        info_train['db'] = self.db

        info_val['ref_name'] = []
        info_val['dst_name'] = []
        info_val['label'] = []
        info_val['name'] = 'val'
        info_val['root_dir'] = self.root_dir
        info_val['transform'] = self.transform
        info_val['db'] = self.db

        for i in indext:
            info_train['ref_name'].append(self.ref_name[i])
            info_train['dst_name'].append(self.dst_name[i])
            info_train['label'].append(self.label[i])

        for i in indexv:
            info_val['ref_name'].append(self.ref_name[i])
            info_val['dst_name'].append(self.dst_name[i])
            info_val['label'].append(self.label[i])

        print('A total [{}] data units for training, and [{}] data units for evaluation'.format(len(indext), len(indexv)))

        if self.data_enhance:
            info_train = self.data_aug(info_train)
        return info_train, info_val

    def divide_info(self):
        # divide dataset without considering reference image
        _info = {}
        _info['ref_name'] = self.ref_name
        _info['dst_name'] = self.dst_name
        _info['label'] = self.label
        if self.data_enhance:
            info = self.data_aug(_info)
        else:
            info = _info
        db_size = len(info['label'])
        index = [i for i in range(0, db_size)]
        random.shuffle(index)
        indext = index[:int(db_size*self.tvr)]
        indexv = index[int(db_size*self.tvr):]

        info_train = {}
        info_val = {}

        info_train['ref_name'] = []
        info_train['dst_name'] = []
        info_train['label'] = []
        info_train['name'] = 'train'
        info_train['root_dir'] = self.root_dir
        info_train['transform'] = self.transform
        info_train['db'] = self.db

        info_val['ref_name'] = []
        info_val['dst_name'] = []
        info_val['label'] = []
        info_val['name'] = 'val'
        info_val['root_dir'] = self.root_dir

        val_transform = []

        val_transform.append(transforms.ToTensor())
        val_transform = transforms.Compose(val_transform)
        info_val['transform'] = val_transform
        info_val['db'] = self.db

        for i in indext:
            info_train['ref_name'].append(info['ref_name'][i])
            info_train['dst_name'].append(info['dst_name'][i])
            info_train['label'].append(info['label'][i])

        for i in indexv:
            info_val['ref_name'].append(info['ref_name'][i])
            info_val['dst_name'].append(info['dst_name'][i])
            info_val['label'].append(info['label'][i])
        print('A total [{}] data units for training, and [{}] data units for evaluation'.format(len(indext), len(indexv)))
        return info_train, info_val

    def data_aug(self, info):
        dst_name = info['dst_name']
        label = info['label']
        orilen = len(label)

        if not isinstance(self.enhance_level, int):
            raise AssertionError(
                'A int value is needed but [{}] value is received'.format(type(self.enhance_level)))

        if self.enhance_level < 1:
            raise AssertionError(
                'A int value more than 1 is needed but [{}] is received'.format(self.enhance_level))

        for i in range(1, self.enhance_level):

            for j in range(orilen):
                nl = int(dst_name[j].split('.')[-2][-1])

                if nl == i:

                    ref_name = dst_name[j]
                    ref_index = j
                    while j in range(orilen - 1):

                        nl2 = int(dst_name[j + 1].split('.')[-2][-1])
                        if nl < nl2:
                            info['ref_name'].append(ref_name)
                            j += 1
                            info['dst_name'].append(dst_name[j])
                            info['label'].append(abs(label[j] - label[ref_index]))  #2
                            # self.label.append(label[ref_index] - label[j])
                        else:
                            break

        print('Dataset [{}] has been data_enhanced from size [{}] to size [{}]'.format(self.db, orilen, len(info['label'])))
        return info

    def get_info_train(self):
        return self.info_train

    def get_info_val(self):
        return self.info_val
