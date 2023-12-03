
import os
from medpy.io import load
import SimpleITK as sitk
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nmse
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    
    # path is your result root
    path = './new_test'
    hr_path = []
    result_path = []
    for root, dirs, files in sorted(os.walk(path)):
        for file in files:
            if file.endswith('hr.img'):
                hr_path.append(os.path.join(root, file))
            if file.endswith('result.img'):
                result_path.append(os.path.join(root, file))

    total_psnr = []
    total_ssim = []
    total_nmse = []
    for i in range(len(hr_path)):
        print(result_path[i])
        SPETimg,_ = load(result_path[i])
        EPETimg,_ = load(hr_path[i])
        chann, weight, height = EPETimg.shape
        for c in range(chann): 
            for w in range(weight):  
                for h in range(height):
                    if EPETimg[c][w][h] <= 0.05:
                        EPETimg[c][w][h] = 0
                        SPETimg[c][w][h] = 0
        y = np.nonzero(EPETimg)
        im1_1 = SPETimg[y]
        im2_1 = EPETimg[y]

        dr = np.max([im1_1.max(), im2_1.max()]) - np.min([im1_1.min(), im2_1.min()])
        cur_psnr = psnr(im1_1, im2_1, data_range=dr)
        cur_ssim = ssim(SPETimg, EPETimg, multi_channel=1)
        cur_nmse = nmse(im1_1, im2_1) ** 2
        print('PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(cur_psnr, cur_ssim, cur_nmse))

        total_psnr.append(cur_psnr)
        total_ssim.append(cur_ssim)
        total_nmse.append(cur_nmse)
    avg_psnr = np.mean(total_psnr)
    avg_ssim = np.mean(total_ssim)
    avg_nmse = np.mean(total_nmse)
    print('Avg. PSNR: {:6f} SSIM: {:6f} NMSE: {:6f}'.format(avg_psnr, avg_ssim, avg_nmse))
