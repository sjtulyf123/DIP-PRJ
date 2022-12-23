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
from bm3d_denoising import bm3d
import cv2

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

def bm3d_dn(image, sigma_psd = 0.08 * 255):
    image_ycbcr = color.rgb2ycbcr(image)

    # Denoise the Y channel using BM3D
    denoised_y = bm3d(image_ycbcr[:, :, 0], sigma_psd = sigma_psd)

    # Reconstruct the denoised image in the RGB color space
    denoised_image = color.ycbcr2rgb(np.dstack((denoised_y, image_ycbcr[:, :, 1], image_ycbcr[:, :, 2])))
    return denoised_image

def wavelet(image):
    image_ycbcr = color.rgb2ycbcr(image)

    # Denoise the Y channel using wavelet denoising
    coeffs = dwt2(image_ycbcr[:, :, 0], 'db1')
    cA, (cH, cV, cD) = coeffs

    # Threshold the coefficients to suppress the noise
    threshold = np.median(np.abs(cH)) / 0.01
    cH[np.abs(cH) < threshold] = 0
    cV[np.abs(cV) < threshold] = 0
    cD[np.abs(cD) < threshold] = 0
    # cH[:] = 0
    # cV[:] = 0
    # cD[:] = 0

    # Reconstruct the denoised image
    denoised_y = idwt2((cA, (cH, cV, cD)), 'db1')

    # Reconstruct the denoised image in the RGB color space
    denoised_image = color.ycbcr2rgb(np.dstack((denoised_y, image_ycbcr[:, :, 1], image_ycbcr[:, :, 2])))
    return denoised_image

def process_one_image_pair(image_name,clean_folder,noise_folder,clean_t,noise_t, severity=1):

    cur_clean_image_path = os.path.join(clean_folder,image_name)
    cur_clean_image = Image.open(cur_clean_image_path)
    cur_clean_image=np.array(clean_t(cur_clean_image))
    cur_clean_image=cv2.cvtColor(cur_clean_image,cv2.COLOR_RGB2BGR)

    cur_noise_image_path = os.path.join(noise_folder, image_name)
    cur_noise_image = Image.open(cur_noise_image_path)
    cur_noise_image = np.array(noise_t(cur_noise_image))
    cur_noise_image=cv2.cvtColor(cur_noise_image,cv2.COLOR_RGB2BGR)
    # bgr1 = cv2.split(cur_clean_image)
    # bgr2 = cv2.split(cur_noise_image)
    # for one_channel in range(3):
    #     denoised_image = filtering(bgr2[one_channel])
    #     tmp_psnr = cal_psnr(bgr1[one_channel], denoised_image)
    #     tmp_res.append(tmp_psnr)
    # return np.mean(tmp_res)
    # denoised_image00 = medianBlur(cur_noise_image)
    # denoised_image01 = GaussianBlur(cur_noise_image)
    # denoised_image02 = bilateralFilter(cur_noise_image)
    # denoised_image2 = wavelet(cur_noise_image)
    sigma_psd = [.08, .12, .18, .26,  .38][severity - 1] * 255
    denoised_image = bm3d(cur_noise_image, sigma_psd=sigma_psd).astype(np.uint8)
    # denoised_image = nl_denoising(cur_noise_image)

    tmp_psnr = cal_psnr(cur_clean_image,denoised_image)

    return tmp_psnr


def process_one_image_pair_debug(image_name,clean_folder,noise_folder,clean_t,noise_t, severity=1):
    tmp_res = []

    cur_clean_image_path = os.path.join(clean_folder,image_name)
    cur_clean_image = Image.open(cur_clean_image_path)
    cur_clean_image=np.array(clean_t(cur_clean_image))
    cur_clean_image=cv2.cvtColor(cur_clean_image,cv2.COLOR_RGB2BGR)

    cur_noise_image_path = os.path.join(noise_folder, image_name)
    cur_noise_image = Image.open(cur_noise_image_path)
    cur_noise_image = np.array(noise_t(cur_noise_image))
    cur_noise_image=cv2.cvtColor(cur_noise_image,cv2.COLOR_RGB2BGR)
    # bgr1 = cv2.split(cur_clean_image)
    # bgr2 = cv2.split(cur_noise_image)
    # for one_channel in range(3):
    #     denoised_image = filtering(bgr2[one_channel])
    #     tmp_psnr = cal_psnr(bgr1[one_channel], denoised_image)
    #     tmp_res.append(tmp_psnr)
    # return np.mean(tmp_res)
    denoised_image00 = medianBlur(cur_noise_image)
    denoised_image01 = GaussianBlur(cur_noise_image)
    denoised_image02 = bilateralFilter(cur_noise_image)
    denoised_image2 = wavelet(cur_noise_image)
    sigma_psd = [.08, .12, .18, .26,  .38][severity - 1] * 255
    denoised_image3 = bm3d(cur_noise_image, sigma_psd=sigma_psd)
    denoised_image4 = nl_denoising(cur_noise_image, h=25)
    # denoised_image3 = bm3d_dn(cur_noise_image, sigma_psd=sigma_psd)
    # denoised_image = bm3d(cur_noise_image, sigma_psd = np.mean(np.abs(cur_noise_image - np.median(cur_noise_image))) / 0.6745)
    tmp_psnr00 = cal_psnr(cur_clean_image,denoised_image00)
    img_pair00 = np.hstack([cur_clean_image, cur_noise_image, denoised_image00])
    # cv2.imshow("medianBlur", img_pair00)
    img_pair00 = Image.fromarray(np.uint8(img_pair00))
    img_pair00.show()
    tmp_psnr01 = cal_psnr(cur_clean_image,denoised_image01)
    img_pair01 = np.hstack([cur_clean_image, cur_noise_image, denoised_image01])
    img_pair01 = Image.fromarray(np.uint8(img_pair01))
    img_pair01.show()
    tmp_psnr02 = cal_psnr(cur_clean_image,denoised_image02)
    img_pair02 = np.hstack([cur_clean_image, cur_noise_image, denoised_image02])
    img_pair02 = Image.fromarray(np.uint8(img_pair02))
    img_pair02.show()
    tmp_psnr2 = cal_psnr(cur_clean_image / 256,denoised_image2)
    img_pair2 = np.hstack([cur_clean_image, cur_noise_image, denoised_image2 * 256])
    img_pair2 = Image.fromarray(np.uint8(img_pair2))
    img_pair2.show()
    tmp_psnr3 = cal_psnr(cur_clean_image / 256,denoised_image3 / 256)
    img_pair3 = np.hstack([cur_clean_image, cur_noise_image, denoised_image3])
    img_pair3 = Image.fromarray(np.uint8(img_pair3))
    img_pair3.show()
    tmp_psnr4 = cal_psnr(cur_clean_image, denoised_image4)
    img_pair4 = np.hstack([cur_clean_image, cur_noise_image, denoised_image4])
    img_pair4 = Image.fromarray(np.uint8(img_pair4))
    img_pair4.show()
    # img_pair5 = np.hstack([cur_clean_image, cur_noise_image])
    # cv2.imshow("medianBlur", img_pair5)
    return tmp_psnr3

def test_and_debug(noise_type=1, noise_level = '5', image_num = 0):
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
    process_one_image_pair_debug(noise_images[image_num],clean_folder,noise_folder,clean_t,noise_t, severity=int(severity))

def main():
    # clean_cate_list = sorted(os.listdir(clean_path))
    res = {}
    final = {}
    denoising_type = "bm3d"


    # check whether one noise image corresponds to one original image
    # if ckp:
    #     for one_noise_type in noise_path:
    #         for one_level in noise_level:
    #             cate_list = sorted(os.listdir(os.path.join(one_noise_type,str(one_level))))
    #             for one_cate in cate_list:
    #                 noise_images = np.array(sorted(os.listdir(os.path.join(one_noise_type,str(one_level),one_cate))))
    #                 clean_images = np.array(sorted(os.listdir(os.path.join(clean_path,one_cate))))
    #                 if (noise_images==clean_images).all():
    #                     continue
    #                 else:
    #                     raise ValueError
    #     print("success")

    # process data and calculate metric
    clean_t = F.Compose([
        F.Resize(256),
        F.CenterCrop(224),
        F.Resize(256)
    ])
    noise_t = F.Compose([
        F.Resize(256),
    ])

    # test_and_debug(image_num=10)

    parallel_worker = Pool(processes=processor_num)
    for one_noise_type in noise_path:
        print(f"Processing {one_noise_type}")
        res[one_noise_type] = {}
        for one_level in noise_level:
            print(f"Noise Level {one_level}")
            res[one_noise_type][one_level] = []
            cate_list = sorted(os.listdir(os.path.join(one_noise_type, str(one_level))))
            # start_time = time.time()
            for one_cate in tqdm(cate_list):
                noise_images = sorted(os.listdir(os.path.join(one_noise_type, str(one_level), one_cate)))
                partial_job = partial(process_one_image_pair, clean_folder=os.path.join(clean_path,one_cate), noise_folder=os.path.join(one_noise_type,str(one_level),one_cate),clean_t=clean_t,noise_t=noise_t, severity=one_level)
                # partial_job = process_one_image_pair(clean_folder=os.path.join(clean_path,one_cate), noise_folder=os.path.join(one_noise_type,str(one_level),one_cate),clean_t=clean_t,noise_t=noise_t)
                batch_res = parallel_worker.map(partial_job, noise_images)
                # batch_res = []
                # for noise_image in tqdm(noise_images):
                    # mtc = process_one_image_pair(image_name=noise_image,clean_folder=os.path.join(clean_path,one_cate), noise_folder=os.path.join(one_noise_type,str(one_level),one_cate),clean_t=clean_t,noise_t=noise_t, severity=one_level)
                    # batch_res.append(mtc)
                tmp_metric = np.mean(batch_res)
                res[one_noise_type][one_level].append(tmp_metric)
                # if k % 10 == 0:
                #     print(f"{k}")
                print(f"PSNR: {tmp_metric}")
                with open(f"psnr_results_{denoising_type}.pkl", "wb") as f:
                    pickle.dump(res, f)
            # end_time = time.time()
            # print("process time: ", end_time-start_time)

    for one_noise_type in res.keys():
        final[one_noise_type]={}
        for one_level in res[one_noise_type].keys():
            final[one_noise_type][one_level] = np.mean(res[one_noise_type][one_level])
    parallel_worker.close()
    parallel_worker.join()
    print(f"ALL PSNR", final)

if __name__=="__main__":
    main()