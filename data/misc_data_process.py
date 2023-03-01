import os
from PIL import Image
import numpy as np
import shutil


def gen_black_l1mask(p):
    """
    Assuming there's only 1 foreground object, copy its mask/01 folder 
    to mask_nocushionmask/02.
    Then generate black images of the same size and same name as those in 
    mask_nocushionmask/02 and put into mask_nocushionmask/01, which is used
    as the initialization of masks for the residual layer.
    
    The composition order is homography background, residual, then foreground.
    So the residual layer has index 1 and foreground layer's index changes to 2.
    
    TODO: the name "nocushionmask" is outdated and has no particular meaning now.

    Args:
        p (_type_): _description_
    """
    os.mkdirs(os.path.join(p, 'mask_nocushionmask/01'), exists_ok=False)
    shutil.copytree(os.path.join(p, 'mask/01'), os.path.join(p, 'mask_nocushionmask/02'))
    for f in os.listdir(os.path.join(p, 'mask_nocushionmask/02')):
        if 'png' in f:
            print(f)
            img = Image.open(os.path.join(p, 'mask_nocushionmask/02', f))
            zeros = np.zeros_like(np.array(img)).astype('uint8')
            zeros_img = Image.fromarray(zeros)
            zeros_img.save(os.path.join(p, 'mask_nocushionmask/01', f))

def real_video_rgba_a(source_dir, dest_dir):
    """
    Given RGBA images in source_dir, extract the Alpha channel and store in dest_dir.
    Used after Stage 1 if you want to manually clean up some predicted alphas.
    """
    os.mkdir(dest_dir)
    for f in os.listdir(source_dir):
        if '.png' in f:
            print(f)
            img_a = np.asarray(Image.open(os.path.join(source_dir, f)))[:,:,-1]
            Image.fromarray(img_a).save(os.path.join(dest_dir, f))
            

if __name__ == '__main__':
    datadir = 'datasets/DVM_womanfall'
    gen_black_l1mask(datadir)
    #     real_video_rgba_a('results/lucia_3layer_v4_rgbwarp1e-1_alphawarp1e-1_flowrecon1e-2/test_600_/images/rgba_l1/', 'datasets/lucia/dis_gt_alpha_stage2_res')