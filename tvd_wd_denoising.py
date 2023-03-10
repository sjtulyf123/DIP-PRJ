import itertools

import cv2
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import os
import time
from functools import partial
from multiprocessing import Process, Pipe, Pool
import torchvision.transforms as F
import matplotlib.pyplot as plt
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_wavelet

noise_path = ["../gaussian_noise","../impulse_noise","../shot_noise"]
noise_level = [5]
clean_path = "../val"
ckp = False

def cal_psnr(clean_image, denoise_image):
    if clean_image.dtype != denoise_image.dtype:
        print(clean_image.dtype, denoise_image.dtype)
    return peak_signal_noise_ratio(clean_image,denoise_image)

def filtering(noise_image, weight=1.0):
    # denoised = denoise_tv_chambolle(noise_image, weight=weight, multichannel=True)
    denoised = denoise_wavelet(noise_image, multichannel=True)
    denoised = (denoised*255).astype("uint8")
    return denoised

def process_one_image_pair(image_name,clean_folder, noise_folder,clean_t,noise_t):
    cur_clean_image_path = os.path.join(clean_folder, image_name)
    cur_clean_image = Image.open(cur_clean_image_path)
    cur_clean_image = np.array(clean_t(cur_clean_image))
    cur_clean_image = cv2.cvtColor(cur_clean_image,cv2.COLOR_RGB2BGR)

    cur_noise_image_path = os.path.join(noise_folder, image_name)
    cur_noise_image = Image.open(cur_noise_image_path)
    cur_noise_image = np.array(noise_t(cur_noise_image))
    cur_noise_image=cv2.cvtColor(cur_noise_image,cv2.COLOR_RGB2BGR)
    denoised_image = filtering(cur_noise_image, weight=0.7)
    tmp_psnr = cal_psnr(cur_clean_image,denoised_image)
    return tmp_psnr

def main():
    # clean_cate_list = sorted(os.listdir(clean_path))
    res = {}
    final = {}

    # check whether one noise image corresponds to one original image
    if ckp:
        for one_noise_type in noise_path:
            for one_level in noise_level:
                cate_list = sorted(os.listdir(os.path.join(one_noise_type,str(one_level))))
                for one_cate in cate_list:
                    noise_images = np.array(sorted(os.listdir(os.path.join(one_noise_type,str(one_level),one_cate))))
                    clean_images = np.array(sorted(os.listdir(os.path.join(clean_path,one_cate))))
                    if (noise_images==clean_images).all():
                        continue
                    else:
                        raise ValueError
        print("success")
                
    # process data and calculate metric
    clean_t = F.Compose([
        F.Resize(256),
        F.CenterCrop(224),
        F.Resize(256)
    ])
    noise_t = F.Compose([
        F.Resize(256),
    ])

    parallel_worker = Pool(processes=400)
    for one_noise_type in noise_path:
        res[one_noise_type] = {}
        for one_level in noise_level:
            res[one_noise_type][one_level] = []
            cate_list = sorted(os.listdir(os.path.join(one_noise_type, str(one_level))))
            start_time = time.time()
            for one_cate in cate_list:
                noise_images = sorted(os.listdir(os.path.join(one_noise_type, str(one_level), one_cate)))
                partial_job = partial(process_one_image_pair, clean_folder=os.path.join(clean_path,one_cate), noise_folder=os.path.join(one_noise_type,str(one_level),one_cate),clean_t=clean_t,noise_t=noise_t)
                batch_res = parallel_worker.map(partial_job, noise_images)
                tmp_metric = np.mean(np.array(batch_res))
                res[one_noise_type][one_level].append(tmp_metric)
            end_time = time.time()
            print("process time: ", end_time-start_time)
    for one_noise_type in res.keys():
        final[one_noise_type]={}
        for one_level in res[one_noise_type].keys():
            final[one_noise_type][one_level] = np.mean(res[one_noise_type][one_level])
    parallel_worker.close()
    parallel_worker.join()
    print(final)

if __name__=="__main__":
    main()