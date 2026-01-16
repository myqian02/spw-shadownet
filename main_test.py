import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import imageio.v2 as imageio
import util
import pandas as pd
import cv2
import lpips
from tqdm import tqdm
import torchvision.models as models
from thop import profile, clever_format

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Params
parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--save_dir', default='tttt/', type=str)
parser.add_argument('--test_data', default='/home/myq/projects/dataset/ISTD+/test/', type=str, help='path of test data')
parser.add_argument('--model_path', default='experiment_100_0.10_2025-08-29_noIgbn_istd+_best/models/model_best.pth', type=str)
parser.add_argument('--dataset_name', default='ISTD', type=str)
parser.add_argument('--n_colors', default=3, type=int)
parser.add_argument('--save_results', default=False, type=bool)
parser.add_argument('--w1', default=10, type=float)
parser.add_argument('--w2', default=0.1, type=float)
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 8

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

def count_params(model):
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for i in param_size:
            count_of_one_param *= i
        count += count_of_one_param
    print('Total parameters: %d' % count)


if __name__ == '__main__':

    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]

    args = parser.parse_args()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    shadow_path = os.path.join(args.test_data, 'test_A')
    gt_path = os.path.join(args.test_data, 'test_C')

    cuda = torch.cuda.is_available()
    lpips_model = lpips.LPIPS(net="alex").cuda()

    ############### prepare train data ###############
    data_time = time.time()
    image_names = []
    for root, _, fnames in sorted(os.walk(gt_path)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
                image_names.append(fname)
    image_names.sort()

    print('Finding {} test data file path'.format(len(image_names)))
    print('Reading images to memory..........')

    ###############################################################
    all_model = []

    model = torch.load(args.model_path)
    count_params(model)

    print(model.__class__)          # 输出模型类名
    model.eval()
    if cuda:
        begin_time = time.time()
        model = model.cuda()
        intensity_psnrs = []
        intensity_ssims = []
        intensity_lpips = []
        intensity_rmses = []
        intensity_niqes = []
        total_time = []

        with torch.no_grad():
            for img_name in tqdm(image_names):
                gt_image = imageio.imread(os.path.join(gt_path, img_name), pilmode='RGB')
                shadow_image = imageio.imread(os.path.join(shadow_path, img_name))
                shadow_image = util.uint2single(shadow_image)

                shadow_input = util.single2tensor4(shadow_image)
                shadow_input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(shadow_input)
                if cuda:
                    shadow_input = shadow_input.cuda()
                pred_img, _ = model(shadow_input)
                pred_img = pad_tensor_back(pred_img, pad_left, pad_right, pad_top, pad_bottom)

                gt_tensor = util.single2tensor4(gt_image/255).cuda()
                intensity_lpip = lpips_model(pred_img, gt_tensor)
                pred_img = util.tensor2single(pred_img)

                torch.cuda.synchronize()

                pred_img = util.single2uint(pred_img)
                intensity_psnr = util.calculate_psnr(pred_img, gt_image)
                intensity_ssim = util.calculate_ssim(pred_img, gt_image)
                intensity_rmse = util.calculate_rmse(pred_img, gt_image)
                intensity_niqe = util.calculate_niqe(gt_image)

                if args.save_results:
                    result_path = os.path.join(args.save_dir, 'results', args.dataset_name, args.model_path[-14:])
                    makedir(result_path)
                    imageio.imsave(os.path.join(result_path, img_name[:-4] + '.png'), pred_img)

        print('Intensity PSNR: %04f' % (np.mean(intensity_psnrs)))
        print('Intensity SSIM: %04f' % (np.mean(intensity_ssims)))
        print('Intensity RMSE: %04f' % (np.mean(intensity_rmses)))
        print('Intensity LPIPS: %04f' % (sum(intensity_lpips)/len(image_names)))
        print('Intensity NIQES: %04f' % (sum(intensity_niqes)/len(image_names)))
        print('Everage Time Per Image: %04f' % (np.mean(total_time)))
        print('Frames Per Second: %04f' % (1/np.mean(total_time)))
        print('Elapsed Time: %04f' % (time.time() - begin_time))
        print('Image Counts: %04d' % (len(intensity_psnrs)))



