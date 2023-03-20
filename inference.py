import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import SimpleITK as sitk
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim
import time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    cnt, cnt3d = 0, 0
    EPETimg = np.zeros([128, 128, 128])
    SPETimg = np.zeros([128, 128, 128])
    IPETimg = np.zeros([128, 128, 128])
    RSimg = np.zeros([128, 128, 128])
    total_psnr, total_ssim, total_nmse = [], [], []
    time_start = time.time()
    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals(need_LR=False)
        image_s = np.squeeze(visuals['HR'].cpu().detach().numpy())
        res = np.squeeze(visuals['SR'].cpu().detach().numpy())
        IP = np.squeeze(visuals['IP'].cpu().detach().numpy())
        EPETimg[cnt, :, :] = res
        IPETimg[cnt, :, :] = IP
        SPETimg[cnt, :, :] = image_s
        cnt += 1

        if cnt == 128:
            time_end = time.time()
            print('time cost', time_end - time_start, 's')
            time_start = time.time()
            cnt = 0
            cnt3d += 1
            RSimg = EPETimg
            # above = np.where(EPETimg<0)
            # EPETimg[above] = 0
            EPETimg = EPETimg +IPETimg
            # above = np.where(EPETimg < 0)
            # EPETimg[above] = 0
            chann, weight, height = EPETimg.shape
            for c in range(chann):  # 遍历高
                for w in range(weight):  # 遍历宽
                    for h in range(height):
                        if EPETimg[c][w][h] <= 0.0:
                            EPETimg[c][w][h] = 0
            y = np.nonzero(SPETimg)  # 取非黑色部分
            SPETimg_1 = SPETimg[y]
            EPETimg_1 = EPETimg[y]
            IPETimg_1 = IPETimg[y]
            print(EPETimg.shape)
            ip_psnr = psnr(IPETimg_1, SPETimg_1, data_range=1)
            cur_psnr = psnr(EPETimg_1, SPETimg_1, data_range=1)
            cur_ssim = ssim(EPETimg, SPETimg, multi_channel=1)
            cur_nmse = nmse(EPETimg, SPETimg) ** 2
            print('IP_PSNR: {:6f} PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(ip_psnr,cur_psnr, cur_ssim, cur_nmse))
            total_psnr.append(cur_psnr)
            total_ssim.append(cur_ssim)
            total_nmse.append(cur_nmse)
            Metrics.save_img(EPETimg, '{}/{}_{}_result.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(RSimg, '{}/{}_{}_rs.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(IPETimg,'{}/{}_{}_IP.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(SPETimg, '{}/{}_{}_hr.img'.format(result_path, current_step, cnt3d))
        # hr_img = Metrics.tensor2img(visuals['HR'],(0,1)) # uint8
        # fake_img = Metrics.tensor2img(visuals['INF'],(0,1)) # uint8
        #
        # sr_img_mode = 'grid'
        # if sr_img_mode == 'single':
        #     # single img series
        #     sr_img = visuals['SR']  # uint8
        #     sample_num = sr_img.shape[0]
        #     for iter in range(0, sample_num):
        #         Metrics.save_img(
        #             Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.img'.format(result_path, current_step, idx, iter))
        # else:
        #     # grid img
        #     sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        #     Metrics.save_img(
        #         sr_img, '{}/{}_{}_sr_process.img'.format(result_path, current_step, idx))
        #     Metrics.save_img(
        #         Metrics.tensor2img(visuals['SR'][-1],(0,1)), '{}/{}_{}_rs.img'.format(result_path, current_step, idx))
        #     Metrics.save_img(
        #         Metrics.tensor2img(visuals['IP'],(0,1)), '{}/{}_{}_IP.img'.format(result_path, current_step, idx))
        #     Metrics.save_img(
        #         (Metrics.tensor2img(visuals['SR'][-1],(0,1))+Metrics.tensor2img(visuals['IP'],(0,1))), '{}/{}_{}_final.img'.format(result_path, current_step, idx))
        #
        # Metrics.save_img(
        #     hr_img, '{}/{}_{}_hr.img'.format(result_path, current_step, idx))
        # Metrics.save_img(
        #     fake_img, '{}/{}_{}_inf.img'.format(result_path, current_step, idx))
    avg_psnr = np.mean(total_psnr)
    avg_ssim = np.mean(total_ssim)
    avg_nmse = np.mean(total_nmse)
    # print(': Avg. PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(avg_psnr, avg_ssim, avg_nmse))
    if wandb_logger and opt['log_infer']:
        wandb_logger.log_eval_table(commit=True)
