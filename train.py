import argparse
import re
import os, glob, datetime, time
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio
import util
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
from torchvision.utils import save_image
from data import TrainData
# from model_noIgb_nosm_ori import shemove
from model_noIgb2 import shemove
from losses import SSIM, MS_SSIM, TVLoss
import matplotlib.pyplot as plt

# Params
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', default=200, type=int, help='number of train epoches')
# parser.add_argument('--epoch', default=200, type=int, help='number of train epoches')
parser.add_argument('--seed', default=813, type=int)
parser.add_argument('--n_feats', default=32, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--save_dir', default='./experiment', type=str)
parser.add_argument('--save_train', default=False, type=bool)
parser.add_argument('--w1', default=10, type=float)
parser.add_argument('--w2', default=0.1, type=float)

# Data
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--train_data', default='/home/myq/projects/dataset/ISTD+/', type=str, help='path of train data')
parser.add_argument('--test_data', default='/home/myq/projects/dataset/ISTD+/test/', type=str, help='path of test data')
parser.add_argument('--pre_trained', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--dataset', default='ISTD', type=str, help='dataset name')
parser.add_argument('--n_colors', default=3, type=int)
parser.add_argument('--patch_size', default=320, type=int)
parser.add_argument('--divide', default=16, type=int)
parser.add_argument('--restart', default=True, type=bool)
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--datas', default='istd+', type=str, help='dataset')

# Optim
parser.add_argument('--lr', default=0.00025, type=float, help='initial learning rate for AdamW')
parser.add_argument('--weight_decay', default=0.02, type=float)
parser.add_argument('--clip_grad_norm', default=2.5, type=float)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

cuda = torch.cuda.is_available()
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
args.save_dir = ('%s_%d_%.2f_%s_noIgb_%s')%(args.save_dir, args.w1*10, args.w2, current_date, args.datas)
model_save_dir = os.path.join(args.save_dir, 'models')
optim_save_dir = os.path.join(args.save_dir, 'optim')
results_save_dir = os.path.join(args.save_dir, 'results')
training_save_dir = os.path.join(args.save_dir, 'training')
shadow_path = os.path.join(args.test_data, 'test_A')
gt_path = os.path.join(args.test_data, 'test_C')
metric_save_dir = os.path.join(args.save_dir, 'metric')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

image_names = []
for root, _, fnames in sorted(os.walk(gt_path)):
    for fname in fnames:
        if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
            image_names.append(fname)
image_names.sort()

print('Finding {} test data file path'.format(len(image_names)))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(model_save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def count_params(model):
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for i in param_size:
            count_of_one_param *= i
        count += count_of_one_param
    print('Total parameters: %d' % count)

def compute_reflectance(image, illumination, epsilon=1e-6):
    if not image.is_cuda or not illumination.is_cuda:
        raise ValueError("Both image and illumination must be on CUDA.")
    reflectance = torch.clamp(reflectance, min=0, max=1)
    
    return reflectance

def compute_illumination(x, epsilon=1e-6):
    if not x.is_cuda:
        raise ValueError("Both image and illumination must be on CUDA.")
    
    Gray = (0.299*x[:, 0:1, :, :]+0.587*x[:, 1:2, :, :]+0.114*x[:, 2:, :, :])
    
    return Gray

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def pad_tensor(input, divide):
    height_org, width_org = input.shape[2], input.shape[3]
    # divide = 8

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

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16]  # 使用前16层
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        # loss = F.mse_loss(output_features, target_features)
        loss = torch.nn.functional.mse_loss(output_features, target_features)
        return loss

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    makedir(model_save_dir)
    makedir(optim_save_dir)
    makedir(results_save_dir)
    makedir(training_save_dir)

    print('===> Building model')
    model = shemove(n_colors=args.n_colors, n_feats=args.n_feats)
    count_params(model)

    ###############################################################

    if args.restart:
        initial_epoch = 0
    else:
        initial_epoch = findLastCheckpoint(save_dir=model_save_dir)  # load the last model in matconvnet style
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(model_save_dir, 'model_%03d.pth' % initial_epoch))
    print("The initial epoch is :", initial_epoch)
    if args.pre_trained != None:
        model = torch.load(args.pre_trained)

    model.train()
    print("Model Name:", model.__class__) 

    criterion = nn.MSELoss()
    criterion_ssim_E = MS_SSIM(data_range=1, channel=3)
    criterion_ssim_I = MS_SSIM(data_range=1, channel=1)
    criterion_ssim = SSIM(data_range=1, channel=3)
    criterion_grad = TVLoss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_ssim_E = criterion_ssim_E.cuda()
        criterion_ssim_I = criterion_ssim_I.cuda()
        criterion_grad = criterion_grad.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-7)
    TrainLoader = TrainData(args).get_loader()
    perceptual_loss = PerceptualLoss().cuda()
    
    lrd = args.lr / ((args.epoch*1//2))
    global_psnr = 0.0
    global_ssim = 0.0
    global_rmse = 0.0
    best_epoch = 0
    psnr_list = []
    ssim_list = []
    rmse_list = []
    lr_list = []
    lr_epoch = args.lr
    lr_end = 1e-8
    for epoch in range(initial_epoch, args.epoch):
        lr_epoch = optimizer.param_groups[0]['lr']
        lr_list.append(lr_epoch)
        time_begin = time.time()

        epoch_total_loss = 0
        epoch_intensity_loss = 0
        epoch_perceptual_loss = 0

        start_time = time.time()
        print("The number of training data is:", len(TrainLoader) * args.batch_size)

        for n_count, (shadow, gt, shadow_illumination, img_name) in enumerate(TrainLoader):
            optimizer.zero_grad()
            if cuda:
                shadow, gt = shadow.cuda(), gt.cuda()

            pred_img = model(shadow)

            if args.save_train:
                training_path = os.path.join(training_save_dir,  img_name[0][:-4] + "_output.png")
                save_image(pred_img, training_path, nrow=args.batch_size//2)
            
            epoch_total_loss += total_loss.item()
            epoch_intensity_loss += intensity_loss.item()
            epoch_perceptual_loss += perception_loss.item()
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        elapsed_time = time.time() - start_time
        log('epoch = %4d , total_loss = %.6f , time = %4.2f s, w1=%d, w2=%.2f' % (epoch + 1, epoch_total_loss, elapsed_time, args.w1*10, args.w2))
        log('learning_rate = %.4f, intensity_loss = %.4f , perception_loss = %.4f' % (optimizer.param_groups[0]['lr'], epoch_intensity_loss, epoch_perceptual_loss))

        model.eval()

        intensity_psnrs = []
        intensity_ssims = []
        intensity_rmses = []

        with torch.no_grad():
            for img_name in image_names:                   
                gt_image = imageio.imread(os.path.join(gt_path, img_name), pilmode='RGB')
                shadow_image = imageio.imread(os.path.join(shadow_path, img_name))
                shadow_image = util.uint2single(shadow_image)

                shadow_input = util.single2tensor4(shadow_image)
                shadow_input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(shadow_input, args.divide)
                shadow_input = shadow_input.cuda()

                pred_img = model(shadow_input)
                pred_img = pad_tensor_back(pred_img, pad_left, pad_right, pad_top, pad_bottom)
                pred_img = util.tensor2single(pred_img)

                torch.cuda.synchronize()
                pred_img = util.single2uint(pred_img)
                intensity_psnr = util.calculate_psnr(pred_img, gt_image)
                intensity_ssim = util.calculate_ssim(pred_img, gt_image)
                intensity_rmse = util.calculate_rmse(pred_img, gt_image)
                
                intensity_psnrs.append(intensity_psnr)
                intensity_ssims.append(intensity_ssim)
                intensity_rmses.append(intensity_rmse)

        eval_psnr = sum(intensity_psnrs)/len(image_names)
        eval_ssim = sum(intensity_ssims)/len(image_names)
        eval_rmse = sum(intensity_rmses)/len(image_names)
        psnr_list.append(eval_psnr)
        ssim_list.append(eval_ssim)
        rmse_list.append(eval_rmse)
        log('Current Test Results: PSNR: %04f , SSIM: %04f' % (eval_psnr, eval_ssim))
        
        if eval_psnr > global_psnr:
            global_psnr = eval_psnr
            global_ssim = eval_ssim
            global_rmse = eval_rmse
            torch.save(model, os.path.join(model_save_dir, 'model_best.pth'))
            best_epoch = epoch + 1
        log('Current Best Results at Epoch %03d: PSNR: %04f , SSIM: %04f , RMSE: %04f' % (best_epoch, global_psnr, global_ssim, global_rmse))

        if (epoch + 1)  % 100 == 0:
            torch.save(model, os.path.join(model_save_dir, 'model_%03d.pth' % (epoch + 1)))

        scheduler.step()

    log('Global Best Results at Epoch %03d: PSNR: %04f , SSIM: %04f , RMSE: %04f' % (best_epoch, global_psnr, global_ssim, global_rmse))
