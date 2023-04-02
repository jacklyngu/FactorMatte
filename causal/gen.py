import cv2 as cv
import torch
import torchvision.models as models
from torchvision import transforms as T

import os
from PIL import Image, ImageFilter 
import numpy as np
import scipy.ndimage

def prep_data(index):
    name = ('00'+str(index))[-4:]+'.png'
    gt = np.asarray(Image.open('../datasets/cushion_birdeye_texturecolor_suzanne_nocushionmask_3layer/rgb/'+name).convert('RGBA'))
    mask = np.asarray(Image.open('../datasets/cushion_birdeye_texturecolor_suzanne_nocushionmask_3layer/mask/01/seg'+name))
    mask[mask!=0] = 1
    mask = np.expand_dims(mask,-1)
    cube = np.where(mask!=0)
    up, down = cube[0].min(), cube[0].max()
    left, right = cube[1].min(), cube[1].max()
    fg = gt * mask
    return [left, right, up, down], fg

def one_warp(fg, boundaries):
    left, right, up, down = boundaries
    src_pts = np.array([[left,up],[right,up], [right,down], [left,down]])
    h = down - up
    w = right - left
    left1 = left + np.random.uniform(-0.5*w, 0.5*w)
    left2 = left + np.random.uniform(-0.5*w, 0.5*w)
    right1 = right + np.random.uniform(-0.5*w, 0.5*w)
    right2 = right + np.random.uniform(-0.5*w, 0.5*w)
    up1 = up + np.random.uniform(-0.5*h, 0.5*h)
    up2 = up + np.random.uniform(-0.5*h, 0.5*h)
    down1 = down + np.random.uniform(-0.5*h, 0.5*h)
    down2 = down + np.random.uniform(-0.5*h, 0.5*h)
    leftstart = np.random.uniform(-0.4 * (448-w), 0.4 * (448-w))
    upstart = np.random.uniform(-0.4 * (256-h), 0.4 * (256-h))
#     print(leftstart,upstart)
    dst_pts = np.array([[left1+leftstart,up1+upstart],[right1+leftstart,up2+upstart],[right2+leftstart,down1+upstart], [left2+leftstart,down2+upstart]])
    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    out = cv.warpPerspective(fg, M, (448, 256), flags=cv.INTER_LINEAR)
    return out
    
def gen_fg():
    save_dir = 'classifier_dataset/cushion_birdeye_texturecolor_suzanne/test/fg'
    for n in range(1000):
        print(n)
        ind = np.random.randint(80, high=221)
        boundaries, fg = prep_data(ind)
        num = np.random.randint(0, high=6)
        grey = np.ones((256, 448,4))*255
        grey[:,:,:3]=np.random.randint(0, high=256)
        canvas = Image.fromarray(grey.astype('uint8'))

        for i in range(num):
            out = one_warp(fg, boundaries)
            alpha = np.random.rand()
            out[:,:,-1]=(out[:,:,-1]*alpha).astype('uint8')
            canvas = Image.alpha_composite(canvas, Image.fromarray(out))
        blur = np.random.rand()
        if blur >0.5:
            r = np.random.uniform(low=0, high=5.5)
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius = r))
        canvas_np = np.asarray(canvas)
        for j in range(256):
            for k in range(448):
                if fg[j,k,-1]==0:
                    fg[j,k]=canvas_np[j,k]
        fg_img = Image.fromarray(fg).convert('RGB')
        fg_img.save(os.path.join(save_dir, str(n)+'_from'+str(ind)+'_test.png'))
        
        
def bg_warp(bg, boundaries):
    left, right, up, down = boundaries
    src_pts = np.array([[left,up],[right,up], [right,down], [left,down]])
    h = 256
    w = 448
    left1 = left + np.random.uniform(-0.15*w, 0.15*w)
    left2 = left + np.random.uniform(-0.15*w, 0.15*w)
    right1 = right + np.random.uniform(-0.15*w, 0.15*w)
    right2 = right + np.random.uniform(-0.15*w, 0.15*w)
    up1 = up + np.random.uniform(-0.15*h, 0.15*h)
    up2 = up + np.random.uniform(-0.15*h, 0.15*h)
    down1 = down + np.random.uniform(-0.15*h, 0.15*h)
    down2 = down + np.random.uniform(-0.15*h, 0.15*h)

    dst_pts = np.array([[left1,up1],[right1,up2],[right2,down1], [left2,down2]])
    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    out = cv.warpPerspective(bg, M, (448, 256), borderMode=cv.BORDER_WRAP, flags=cv.INTER_LINEAR) #[up:down, left:right]
    return out


save_dir = 'classifier_dataset/cushion_birdeye_texturecolor_suzanne/train/bg/'
gt = np.asarray(Image.open('../datasets/cushion_birdeye_texturecolor_suzanne_nocushionmask_3layer/bg_gt.png').convert('RGBA'))
left = 115
right = 302
up = 33
down = 200
boundaries = [left, right, up, down]
for n in range(5000):
    print(n)
    ind = np.random.randint(80, high=221)
    out = bg_warp(gt, boundaries)
    alpha = np.random.uniform(0.85, high=1)
    out[:,:,-1]=(out[:,:,-1]*alpha).astype('uint8')
    canvas = Image.fromarray(out).convert('RGB')
    blur = np.random.rand()
    if blur >0.5:
        r = np.random.uniform(low=0, high=5.5)
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius = r))
    canvas.save(os.path.join(save_dir, str(n)+'_from'+str(ind)+'_bg_train.png'))