import itertools
import pickle

import cv2
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import os
import time
from functools import partial
from multiprocessing import Process, Pipe, Pool
import torchvision.transforms as F
from PIL import Image
from skimage import color
from pywt import dwt2, idwt2
from tqdm import tqdm
from bm3d import bm3d
import cv2
import matplotlib.pyplot as plt


noise_path = ["../gaussian_noise","../impulse_noise","../shot_noise"]
# noise_level = [1,2,3,4,5]
noise_level = [5]
processor_num = 4
clean_path = "../val"
ckp = False

def cal_psnr(clean_image, denoise_image):
    return peak_signal_noise_ratio(clean_image,denoise_image)

def medianBlur(noise_image):
    denoised = cv2.medianBlur(noise_image,3)
    # denoised = cv2.blur(noise_image,(3,3))
    # denoised = cv2.GaussianBlur(noise_image,(3,3),0)
    # denoised = cv2.bilateralFilter(noise_image,5,20,20)
    return denoised

def nl_denoising(noise_image, h = 25, templateWS = 7, searchWS = 21):
    denoised_img = cv2.fastNlMeansDenoisingColored(noise_image, None, h, h, templateWS, searchWS)
    # denoised_img = cv2.fastNlMeansDenoisingColored(noise_image)
    return denoised_img

def GaussianBlur(noise_image):
    # denoised = cv2.medianBlur(noise_image,3)
    # denoised = cv2.blur(noise_image,(3,3))
    denoised = cv2.GaussianBlur(noise_image,(3,3),0)
    # denoised = cv2.bilateralFilter(noise_image,5,20,20)
    return denoised

def bilateralFilter(noise_image):
    # denoised = cv2.medianBlur(noise_image,3)
    # denoised = cv2.blur(noise_image,(3,3))
    # denoised = cv2.GaussianBlur(noise_image,(3,3),0)
    denoised = cv2.bilateralFilter(noise_image,5,20,20)
    return denoised

def draw_imggrid(imgs, rows, cols, x_labels):
    fig, axs = plt.subplots(rows, cols)
    for i, (img, lb) in enumerate(zip(imgs,x_labels)):
        row = i / cols
        col = i % cols
        axs[row][col].imshow(img)
        axs[row][col].set_axis_off()
        axs.set_xlabel(lb)
    fig.show()


def draw_denoising(image_name,clean_folder,noise_folder,clean_t,noise_t, severity=1):
    tmp_res = []

    cur_clean_image_path = os.path.join(clean_folder,image_name)
    cur_clean_image = Image.open(cur_clean_image_path)
    cur_clean_image=np.array(clean_t(cur_clean_image))
    cur_clean_image=cv2.cvtColor(cur_clean_image,cv2.COLOR_RGB2BGR)

    cur_noise_image_path = os.path.join(noise_folder, image_name)
    cur_noise_image = Image.open(cur_noise_image_path)
    cur_noise_image = np.array(noise_t(cur_noise_image))
    cur_noise_image=cv2.cvtColor(cur_noise_image,cv2.COLOR_RGB2BGR)

    denoised_image_mean = cv2.blur(cur_noise_image,(3,3))
    denoised_image_median = medianBlur(cur_noise_image)
    denoised_imag_gaussian = GaussianBlur(cur_noise_image)
    denoised_image_bil = bilateralFilter(cur_noise_image)
    sigma_psd = [.08, .12, .18, .26,  .38][severity - 1] * 255
    denoised_image_bm3d = bm3d(cur_noise_image, sigma_psd=sigma_psd)
    denoised_image_nl = nl_denoising(cur_noise_image, h=25)

    denoised_image_variational = np.zeros_like(cur_clean_image)
    denoised_image_wavelet = np.zeros_like(cur_clean_image)

    imgs = [cur_clean_image, cur_noise_image, 
        denoised_image_mean,denoised_image_median,denoised_imag_gaussian,
        denoised_image_bil, denoised_image_variational, 
        denoised_image_wavelet, denoised_image_nl, denoised_image_bm3d]
    
    x_labels = [
        "Clean", "Gaussian Noise", "Mean", "Median", 
        "Gaussian", "Bilateral", "Variational", "Wavelet", "NLMean", "BM3D"
    ]

    draw_imggrid(imgs, 2, 5, x_labels)

    # img_row1 = np.hstack([cur_clean_image, cur_noise_image, 
    #     denoised_image_mean,denoised_image_median,denoised_imag_gaussian])
        
    # img_row2 = np.hstack([denoised_image_bil, denoised_image_variational, 
    #     denoised_image_wavelet, denoised_image_nl, denoised_image_bm3d])
    

    # img_pair_all = Image.fromarray(np.uint8(img_pair_all))
    # img_pair_all.show()


def test_and_debug(noise_type=0, noise_level = '5', image_num = 0):
    i = noise_type
    severity = noise_level
    clean_t = F.Compose([
        F.Resize(256),
        F.CenterCrop(224),
        F.Resize(256)
    ])
    noise_t = F.Compose([
        F.Resize(256),
    ])
    cate_list = sorted(os.listdir(os.path.join(noise_path[i], severity)))
    noise_images = sorted(os.listdir(os.path.join(noise_path[i], severity, cate_list[0])))
    noise_folder = os.path.join(noise_path[i], severity, cate_list[0])
    clean_folder = os.path.join(clean_path,cate_list[0])
    draw_denoising(noise_images[image_num],clean_folder,noise_folder,clean_t,noise_t, severity=int(severity))

def main():
    test_and_debug(image_num=10)

if __name__=="__main__":
    main()