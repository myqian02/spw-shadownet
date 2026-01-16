import os
import imageio
import torch
from data import common
import numpy as np
from glob import glob
import torch.utils.data as data


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def get_content_after_last_backslash(s):
    last_backslash_index = s.rfind('/')
    if last_backslash_index == -1:
        return ""
    return s[last_backslash_index + 1:]

class ISTD(data.Dataset):
    def __init__(self, args, mode='train'):
        super(ISTD, self).__init__()
        self.args = args
        self.n_colors = args.n_colors
        self.mode = mode

        self._scan()

    def _scan(self):
        self.image_names = glob(self.args.train_data + '/train/train_C/*.png')
        return

    def __getitem__(self, idx):
        high_name = self.image_names[idx]
        img_name = get_content_after_last_backslash(high_name)
        low_name = high_name.replace('train_C', 'train_A')
        low_illumination_name = high_name.replace('train_A', 'train_A')
        high = imageio.imread(high_name, pilmode='RGB')
        low = imageio.imread(low_name, pilmode='RGB')
        low_illumination = imageio.imread(low_illumination_name)

        H, W, C = high.shape
        ix = np.random.randint(0, H - self.args.patch_size + 1)
        iy = np.random.randint(0, W - self.args.patch_size + 1)

        high_patch = high[ix:ix + self.args.patch_size, iy:iy + self.args.patch_size, :]
        low_patch = low[ix:ix + self.args.patch_size, iy:iy + self.args.patch_size, :]
        low_illumination_patch = low_illumination[ix:ix + self.args.patch_size, iy:iy + self.args.patch_size, 0]

        aug_mode = np.random.randint(0, 8)
        high_patch = common.augment_img(high_patch, aug_mode)
        low_patch = common.augment_img(low_patch, aug_mode)
        low_illumination_patch = common.augment_img(low_illumination_patch, aug_mode)

        high_patch = common.image_to_tensor(high_patch)
        low_patch = common.image_to_tensor(low_patch)
        low_illumination_patch = common.image_to_tensor(low_illumination_patch)

        return low_patch, high_patch, low_illumination_patch, img_name



    def __len__(self):
        return len(self.image_names)
