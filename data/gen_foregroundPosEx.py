import sys
import torch
import torchvision.models as models
from torchvision import transforms as T

import os
from PIL import Image, ImageFilter 
import numpy as np
import scipy as sp
import scipy.signal
from shutil import copyfile
import cv2 as cv
from third_party.data.image_folder import make_dataset


def prep_data(basedir, index):
    rgb_paths = sorted(make_dataset(os.path.join(basedir, 'rgb')))
    # mask_paths = sorted(make_dataset(os.path.join(basedir, 'l2_fake_real_comp_mask')))
    mask_paths = sorted(make_dataset(os.path.join(basedir, 'mask_nocushionmask/02/')))
    gt = np.asarray(Image.open(rgb_paths[index]).convert('RGBA'))
    mask = np.asarray(Image.open(mask_paths[index]).convert('L'))
    if abs(mask).sum() == 0:
        return None, None
    mask[mask != 0] = 1
        # Optionally erode to be conservative
    mask = cv.erode(mask, kernel=np.ones((5, 5)), iterations=1)
    mask = np.expand_dims(mask, -1)
    cube = np.where(mask != 0)
    up, down = cube[0].min(), cube[0].max()
    left, right = cube[1].min(), cube[1].max()
    fg = gt * mask
    return [left, right, up, down], fg

def add_reflection(img, surface=140, alpha_range=[0, 0.75]):
    alpha = np.random.uniform(alpha_range[0], high=alpha_range[1])
    print(alpha)
    h, w, _ = img.shape
    start = abs(h - 2*surface)
    img[surface:] = alpha * img[start:surface].copy()[::-1]
    return img

# def add_blur(img, sigma_range=[10, 20]):
#     sigma = 0
#     while sigma % 2 == 0:
#         # GaussianBlur only accepts odd kernel size
#         sigma = np.random.randint(sigma_range[0], high=sigma_range[1])
#     img = cv.GaussianBlur(img, (sigma, sigma), sigma/4 , borderType = cv.BORDER_REPLICATE)
#     return img

def add_blur_1(img, sigma_range=[0.2, 1], kernel=5):
    img = img.astype("int16")
    std = np.random.uniform(sigma_range[0], high=sigma_range[1])
    blur_img = cv.GaussianBlur(img, (kernel, kernel), std, borderType = cv.BORDER_REPLICATE)
    blur_img = ceil_floor_image(blur_img)
    return blur_img

def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

def add_noise(img, std_range=[0, 20], mean=0):
    std = np.random.randint(std_range[0], high=std_range[1])
    print('std', std)
    gaussian_noise = np.random.normal(mean, std, img.shape)
    img = img.astype("int16")
    noise_img = img + gaussian_noise
    noise_img = ceil_floor_image(noise_img)
    return noise_img

def flip(img):
    p = np.random.uniform()
    if p<0.5:
        img = img[:, ::-1]
    else:
        img = img[::-1, :]
    return img

def rotate(img):
    deg = np.random.randint(0, 360)
    img = sp.ndimage.rotate(img, deg, reshape=False)
    return img

def gen_pos_ex_fg(basedir, ind_low, ind_high, add_rot, add_flip, add_blurr_or_noise, add_gaussian_noise, \
    num=5000, blur_kwargs=None, noise_kwargs=None, folder_suffix=''):
    """
    ind high exclusive
    """
    save_dir = os.path.join(basedir, 'dis_real_l2', '_'.join([str(add_rot) + 'rot', str(add_flip) + 'flip', \
        str((1-add_gaussian_noise)*add_blurr_or_noise) + 'blursigma' + str(blur_kwargs['sigma_range'][0]) + str(blur_kwargs['sigma_range'][0])+'k'+str(blur_kwargs['kernel']),\
             str(add_gaussian_noise*add_blurr_or_noise)+'gaussian_noise_std'+str(noise_kwargs['std_range'][0])+str(noise_kwargs['std_range'][1])+'mean'+str(noise_kwargs['mean']), folder_suffix]))
    os.makedirs(save_dir)
    for n in range(0, num):
        print(n)
        fg = None
        while fg is None:
            ind = np.random.randint(0, high=ind_high-ind_low+1)
            boundaries, fg = prep_data(basedir, ind)
        
        decision = np.random.uniform(size=5)
        print(decision)
        # if decision[4] < 0.5:
        #     scale = np.random.uniform(0.2, 1.2)
        #     print('scale', scale)
        #     scaled = scale * fg[:,:,:3].astype('int')
        #     fg[:,:,:3] = np.clip(scaled, 0, 255).astype('uint8')
        if decision[0] < add_rot:
            print('rot')
            fg = rotate(fg)
        if decision[1] < add_flip:
            print('flip')
            fg = flip(fg)

        h, w, _ = fg.shape
        grey = np.ones((h, w, 4))*255
        grey[:,:,0]=0
        grey[:,:,1]=255
        grey[:,:,2]=0
#         grey[:,:,:3]=np.random.randint(0, high=80)
        canvas = Image.fromarray(grey.astype('uint8'))
        canvas_np = np.asarray(canvas)
        for j in range(h):
            for k in range(w):
                if fg[j, k, -1] == 0:
                    fg[j, k] = canvas_np[j,k]
            
        if decision[2] < add_blurr_or_noise:
            if decision[3] < add_gaussian_noise:
                print('gaussian noise')
                fg = add_noise(fg, **noise_kwargs)
            # blur and noise are exclusive
            else:
                print('blurr, using add_blur_1')
                fg = add_blur_1(fg, **blur_kwargs)
        fg_img = Image.fromarray(fg).convert('RGB')
        fg_img.save(os.path.join(save_dir, '_'.join([str(n), 'from', str(ind+ind_low), 'fg', folder_suffix])+'.png'))
        
        
        
if __name__ == '__main__':
    datadir = 'datasets/DVM_womanfall'
    video_start_ind = 0
    video_end_ind = 99

    # The probability of applying each augmentation during the generation of each positive example
    add_rot = 0 #0.5
    add_flip = 0 #0.5
    # reflec_kwargs= {
    #     'alpha_range': [0.1, 0.7], 
    #     'surface': 140
    # }

    add_blurr_or_noise = 0.5
    blur_kwargs= {'sigma_range':[0.2, 1], 'kernel': 5}
    add_gaussian_noise = 0.5
    gaussian_noise_kwargs= {'std_range':[2, 7], 'mean':0}                           

    gen_pos_ex_fg(datadir, video_start_ind, video_end_ind, add_rot, add_flip, \
                  add_blurr_or_noise, add_gaussian_noise, num=2500, blur_kwargs=blur_kwargs, \
                    noise_kwargs=gaussian_noise_kwargs, folder_suffix='erode5')

        