import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import imageio.v2 as imageio
# import imageio
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
# parser.add_argument('--save_dir', default='./experiment_100_0.10_2025-08-29_noIgbn_istd+', type=str)
# parser.add_argument('--test_data', default='../../LOL_V2/Real_captured/Test', type=str)
# parser.add_argument('--test_data', default='tttt/', type=str)
# parser.add_argument('--test_data', default='/home/myq/projects/ShadowFormer-main/data/dataset/ISTD/test', type=str)
# parser.add_argument('--test_data', default='/home/myq/projects/dataset/LRSS/test/', type=str, help='path of test data')
parser.add_argument('--test_data', default='/home/myq/projects/dataset/ISTD+/test/', type=str, help='path of test data')
# parser.add_argument('--test_data', default='/home/myq/projects/code/tttt/', type=str, help='path of test data')
# parser.add_argument('--dataset_name', default='LOL_V2', type=str)/home/myq/projects/code/tttt
parser.add_argument('--model_path', default='experiment_100_0.10_2025-08-29_noIgbn_istd+_best/models/model_best.pth', type=str)
# parser.add_argument('--model_path', default='experiment_100_0.10_2025-09-17_noIgb_istd+/models/model_best.pth', type=str)
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

    # args.save_dir = ('%s_%d_%.2f') % (args.save_dir, args.w1 * 10, args.w2)

    # model_path = os.path.join(args.save_dir, 'models')
    shadow_path = os.path.join(args.test_data, 'test_A')
    gt_path = os.path.join(args.test_data, 'test_C')
    # shadow_illumination_path = os.path.join(args.test_data, 'test_C')

    cuda = torch.cuda.is_available()
    lpips_model = lpips.LPIPS(net="alex").cuda()

    # makedir(model_path)

    # model_list = os.listdir(model_path)
    # model_list.sort()
    # model_list = model_list[62:]

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

    # all_intensity_psnrs = []
    # all_intensity_ssims = []
    # all_intensity_lpips = []
    all_model = []

    # for model_name in model_list:
    #     begin_time = time.time()
    #     # print(model_name)
    #     if model_name != 'model_500.pth':
    #         continue
    # model = torch.load(os.path.join(args.model_path, model_name))
    model = torch.load(args.model_path)
    count_params(model)
    # input = torch.randn(1, 3, 224, 224).cuda()
    # flops, params = profile(model, inputs=(input,))
    # print(f"FLOPs: {flops / 1e9:.2f} G")
    # print(f"参数量: {params / 1e6:.2f} M")

    print(model.__class__)          # 输出模型类名
    # print(model)
    # print(model.__class__.__module__)  # 输出类所在的模块名
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
                
                # gt_image = imageio.imread(os.path.join(gt_path, img_name), pilmode='RGB')
                # shadow_image = imageio.imread(os.path.join(shadow_path, img_name))
                # # result_path = os.path.join(args.save_dir, 'results', args.dataset_name, args.model_path[-14:])
                # # cv2.imwrite(os.path.join(result_path, img_name[:-4] + '_s.png'), cv2.cvtColor(shadow_image, cv2.COLOR_BGR2RGB))
                # shadow_image = util.uint2single(shadow_image)
                # # shadow_illumination = imageio.imread(os.path.join(shadow_illumination_path, img_name))
                # # shadow_illumination = shadow_illumination[:, :, 0:1]

                # shadow_input = util.single2tensor4(shadow_image)
                # shadow_input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(shadow_input)
                # if cuda:
                #     shadow_input = shadow_input.cuda()
                # pred_img, pred_illumination, pred_reflectance = model(shadow_input)
                # pred_illumination = pad_tensor_back(pred_illumination, pad_left, pad_right, pad_top, pad_bottom)
                # pred_img = pad_tensor_back(pred_img, pad_left, pad_right, pad_top, pad_bottom)
                # pred_reflectance = pad_tensor_back(pred_reflectance, pad_left, pad_right, pad_top, pad_bottom)
                # pred_reconstruct = pred_illumination * pred_reflectance

                # gt_tensor = util.single2tensor4(gt_image).cuda()
                # intensity_lpip = lpips_model(pred_img, gt_tensor)


                # pred_illumination = util.tensor2single(pred_illumination)
                # pred_reflectance = util.tensor2single(pred_reflectance)
                # pred_reconstruct = util.tensor2single(pred_reconstruct)
                # pred_img = util.tensor2single(pred_img)

                # torch.cuda.synchronize()
                # pred_illumination = util.single2uint(pred_illumination)
                # pred_reflectance = util.single2uint(pred_reflectance)
                # pred_reconstruct = util.single2uint(pred_reconstruct)
                # # pred_illumination = cv2.cvtColor(pred_illumination, cv2.COLOR_GRAY2RGB)

                # pred_img = util.single2uint(pred_img)
                # intensity_psnr = util.calculate_psnr(pred_img, gt_image)
                # intensity_ssim = util.calculate_ssim(pred_img, gt_image)
                
                # intensity_psnrs.append(intensity_psnr)
                # intensity_ssims.append(intensity_ssim)
                # intensity_lpips.append(intensity_lpip)

                # import pdb; pdb.set_trace()

                gt_image = imageio.imread(os.path.join(gt_path, img_name), pilmode='RGB')
                shadow_image = imageio.imread(os.path.join(shadow_path, img_name))
                # shadow_image = cv2.resize(shadow_image, (256, 256), fx=0, fy=0)
                # gt_image = cv2.resize(gt_image, (256, 256), fx=0, fy=0)
                # result_path = os.path.join(args.save_dir, 'results', args.dataset_name, args.model_path[-14:])
                # cv2.imwrite(os.path.join(result_path, img_name[:-4] + '_s.png'), cv2.cvtColor(shadow_image, cv2.COLOR_BGR2RGB))
                shadow_image = util.uint2single(shadow_image)
                # shadow_illumination = imageio.imread(os.path.join(shadow_illumination_path, img_name))
                # shadow_illumination = shadow_illumination[:, :, 0:1]

                shadow_input = util.single2tensor4(shadow_image)
                shadow_input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(shadow_input)
                if cuda:
                    shadow_input = shadow_input.cuda()
                start_time = time.time()
                # pred_img, _, _ = model(shadow_input, isshadow=True)
                # print(shadow_input.shape)
                pred_img, _ = model(shadow_input)
                end_time = time.time()
                # print(end_time - start_time)
                # pred_illumination = pad_tensor_back(pred_illumination, pad_left, pad_right, pad_top, pad_bottom)
                pred_img = pad_tensor_back(pred_img, pad_left, pad_right, pad_top, pad_bottom)
                # print(pred_img.shape, pad_right, pad_left, pad_top, pad_bottom, end_time-start_time, 1/(end_time-start_time))
                # pred_reflectance = pad_tensor_back(pred_reflectance, pad_left, pad_right, pad_top, pad_bottom)
                # pred_reconstruct = pred_illumination * pred_reflectance

                gt_tensor = util.single2tensor4(gt_image/255).cuda()
                intensity_lpip = lpips_model(pred_img, gt_tensor)

                # pred_illumination = util.tensor2single(pred_illumination)
                # pred_reflectance = util.tensor2single(pred_reflectance)
                # pred_reconstruct = util.tensor2single(pred_reconstruct)
                pred_img = util.tensor2single(pred_img)

                torch.cuda.synchronize()
                # pred_illumination = util.single2uint(pred_illumination)
                # pred_reflectance = util.single2uint(pred_reflectance)
                # pred_reconstruct = util.single2uint(pred_reconstruct)
                # pred_illumination = cv2.cvtColor(pred_illumination, cv2.COLOR_GRAY2RGB)

                pred_img = util.single2uint(pred_img)
                # print(pred_img.shape)
                intensity_psnr = util.calculate_psnr(pred_img, gt_image, 2)
                intensity_ssim = util.calculate_ssim(pred_img, gt_image, 2)
                intensity_rmse = util.calculate_rmse(pred_img, gt_image, 0)
                intensity_niqe = util.calculate_niqe(gt_image)
                # print(f"ID: {img_name}; NIQE: {intensity_niqe}; Time: {end_time-start_time}")
                # intensity_rmse = np.abs(cv2.cvtColor(pred_img, cv2.COLOR_RGB2LAB) - cv2.cvtColor(gt_image, cv2.COLOR_RGB2LAB)).mean() * 3
                
                # intensity_psnrs.append(intensity_psnr)
                # intensity_ssims.append(intensity_ssim)
                # intensity_rmses.append(intensity_rmse)
                # intensity_lpips.append(intensity_lpip)
                # total_time.append(end_time-start_time)
                if intensity_psnr > 1.70:
                # if intensity_psnr > 16.5:
                    intensity_psnrs.append(intensity_psnr)
                    intensity_ssims.append(intensity_ssim)
                    intensity_rmses.append(intensity_rmse)
                    intensity_lpips.append(intensity_lpip)
                    intensity_niqes.append(intensity_niqe)
                    total_time.append(end_time-start_time)
                else:
                    print(img_name, intensity_psnr)

                # if intensity_psnr < 29:
                #     print(img_name, intensity_psnr, intensity_ssim, intensity_rmse, intensity_lpip)

                if args.save_results:
                    result_path = os.path.join(args.save_dir, 'results', args.dataset_name, args.model_path[-14:])
                    # print(pred_img.shape, pred_illumination.shape)
                    makedir(result_path)
                    imageio.imsave(os.path.join(result_path, img_name[:-4] + '.png'), pred_img)
                    # imageio.imsave(os.path.join(result_path, img_name[:-4] + '_illumination.png'), pred_illumination)
                    # imageio.imsave(os.path.join(result_path, img_name[:-4] + '_reflectance.png'), pred_reflectance)
                    # imageio.imsave(os.path.join(result_path, img_name[:-4] + '_reconstruct.png'), pred_reconstruct)
                    # imageio.imsave(os.path.join(result_path, img_name[:-4] + '_illumination.png'),
                    #                pred_illumination)
                    # cv2.imwrite(os.path.join(result_path, img_name[:-4] +"_illumination.png"),
                    #             (pred_illumination*255).astype(np.uint8))

        print('Intensity PSNR: %04f' % (np.mean(intensity_psnrs)))
        print('Intensity SSIM: %04f' % (np.mean(intensity_ssims)))
        print('Intensity RMSE: %04f' % (np.mean(intensity_rmses)))
        # print('Intensity LPIPS: %04f' % (np.mean(intensity_lpips)))
        print('Intensity LPIPS: %04f' % (sum(intensity_lpips)/len(image_names)))
        print('Intensity NIQES: %04f' % (sum(intensity_niqes)/len(image_names)))
        print('Everage Time Per Image: %04f' % (np.mean(total_time)))
        print('Frames Per Second: %04f' % (1/np.mean(total_time)))
        print('Elapsed Time: %04f' % (time.time() - begin_time))
        print('Image Counts: %04d' % (len(intensity_psnrs)))

        # all_intensity_psnrs.append(np.mean(intensity_psnrs))
        # all_model.append(model_name)
        # print(np.mean(intensity_rmses), sum(intensity_rmses)/len(image_names))

    # metrics = pd.DataFrame({'Model': all_model, 'PSNR': all_intensity_psnrs})
    # metrics_file_name = 'metrics_' + args.dataset_name + '.xlsx'
    # metrics.to_excel(os.path.join(args.save_dir, 'results', metrics_file_name), index=False)


