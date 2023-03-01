# FactorMatte
## Environment
`conda create -n factormatte python=3.9 anaconda`
<br /> 
`conda activate factormatte`
<br /> 
Use conda or pip to install requirements.txt

## Example Video
### Download Dataset and put into the datasets/ folder
https://drive.google.com/file/d/1-nZ9VA8bqRvll_4HEPGOxihIJ4o8y0kY/view?usp=sharing

### Stage 1
`python train_GAN.py --name sand_car_3layer_v4_rgbwarp1e-1_alphawarp1e-1_flowrecon1e-2 --stage 1 --dataset_mode omnimatte_GANCGANFlip148 --model omnimatte_GANFlip --dataroot ./datasets/sand_car --height 192 --width 288  --save_by_epoch --prob_masks --lambda_rgb_warp 1e-1 --lambda_alpha_warp 1e-1 --model_v 4 --residual_noise --strides 0,0,0 --num_Ds 0,0,0 --n_layers 0,0,0 --display_ind 63 --pos_ex_dirs , --batch_size 16 --n_epochs 1200 --bg_noise --gpu_ids 1,0 --lambda_recon_flow 1e-2`

<span style="color:blue">Copy the trained weights to the next stage's training folder:</span> `cp 1110_checkpoints/sand_car_3layer_v4_rgbwarp1e-1_alphawarp1e-1_flowrecon1e-2/*1200* 1110_checkpoints/sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_noninter_flowmask_flowrecon1e-2`

<span style="color:blue">Run test to generate the background image:</span> `python test.py --name sand_car_3layer_v4_rgbwarp1e-1_alphawarp1e-1_flowrecon1e-2 --dataset_mode omnimatte_GANCGANFlip148 --model omnimatte_GANFlip --dataroot ./datasets/DVM_manstatic --prob_masks --model_v 4 --residual_noise --strides 0,0,0 --num_Ds 0,0,0 --n_layers 0,0,0 --pos_ex_dirs , --epoch 1200 --stage 1 --gpu_ids 0 --start_ind 0 --width 512 --height 288`

And put it in the data folder, it'll be used for the following stages. `cp results/sand_car_3layer_v4_rgbwarp1e-1_alphawarp1e-1_flowrecon1e-2/test_1200_/panorama.png datasets/sand_car/bg_gt.png`

### Stage 2
`python train_GAN.py --name sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_noninter_flowmask_flowrecon1e-2 --init_flowmask --lambda_recon_flow 1e-2 --dataset_mode factormatte_GANCGANFlip148 --model factormatte_GANFlip --dataroot ./datasets/sand_car --save_by_epoch --prob_masks --lambda_rgb_warp 1e-1 --lambda_alpha_warp 1e-1 --residual_noise --strides 0,2,2 --num_Ds 0,1,3 --n_layers 0,3,3 --start_ind 0 --noninter_only --width 288 --height 192 --discriminator_transform randomcrop --pos_ex_dirs 0uniform_0gaussian_dark_0flip_0elastic_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_rawframes,0rot_0flip_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_ --gpu_ids 0 --n_epochs 2400 --continue_train --epoch 1200 --stage 2 --display_ind 15`

<span style="color:blue">Copy the trained weights to the next stage's training folder:</span> `cp 1110_checkpoints/sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_noninter_flowmask_flowrecon1e-2/*2400* 1110_checkpoints/sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_l2arecon1e-1dilate_recon2_148_othersretro_stage22000cont_flowrecon1e-1`


### Stage 3
`python train_GAN.py --name sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_l2arecon1e-1dilate_recon2_148_othersretro_stage22000cont_flowrecon1e-1 --lambda_recon 2 --lambda_recon_flow 1e-1 --dataset_mode factormatte_GANCGANFlip148 --model factormatte_GANFlip --dataroot ./datasets/sand_car --save_by_epoch --prob_masks --lambda_rgb_warp 1e-1 --lambda_alpha_warp 1e-1 --residual_noise --strides 0,2,2 --num_Ds 0,1,3 --display_ind 63 --init_flowmask --lambda_recon_3 1e-1 --start_ind 0 --discriminator_transform randomcrop --steps 148 --pos_ex_dirs 0uniform_0gaussian_dark_0flip_0elastic_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_rawframes,0rot_0flip_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_ --stage 3` --height 192 --width 288 --n_epochs 3200 --gpu_ids 0 --continue_train --epoch 2400 --overwrite_lambdas

### Pretrained Weights
For convenience, you can also download the weights of any stage for this dataset and start training from there on.

Stage 1 weights: https://drive.google.com/drive/folders/1ERZQNM8nT2Xw9J2yzFzp3QoyxHFEZ7B4?usp=sharing

Stage 2 weights: https://drive.google.com/drive/folders/1boJJ8DwPZxk9hzxUa-4nLW0vVPhXdhPL?usp=sharing

Stage 3 weights: https://drive.google.com/drive/folders/1eDHuIsoON_ou_50sZ7nT4D3luiGxVvyx?usp=sharing

### Generate Results
`python test.py --name sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_l2arecon1e-1dilate_recon2_148_othersretro_stage22000cont_flowrecon1e-1 --dataset_mode factormatte_GANCGANFlip148 --model factormatte_GANFlip --dataroot ./datasets/sand_car --gpu_ids 0 --prob_masks --residual_noise --pos_ex_dirs , --epoch 3200 --stage 3 --width 288 --height 192 --init_flowmask --test_suffix texts_to_put_after_fixed_folder_name`


## Custom Dataset
To train on your custom video, please prepare it as follows: (Assume all file names are [xxxxx].png, e.g. 00001.png, 00100.png, 10001.png)
1. Extract all RGB frames and put them in "rgb" folder.
2. Arrange corresponding binary masks in the same order and put them in `mask/01` folder.
3. run `data/misc_data_process.py` to copy `mask/01` to `mask_nocushionmask/02`, and generate `mask_nocushionmask/01`. Please refer to the doc in data/misc_data_process.py for details. (Redundant, TODO: generate this on the fly.)
4. Estimate the homography between every two consecutive frames and flatten each matrix following the template of data/homographies.txt 
We provide a script in data/keypoint_homo_short.ipynb. It'll generate a file homographies_raw.txt. To get the final homographies.txt, run 
`python datasets/homography.py  --homography_path ./datasets/[your_folder_name]/homographies_raw.txt --width [W] --height [H]`

5. Flow estimation by RAFT:
`python video_completion.py --path datasets/[your_folder_name]/rgb --model weight/raft-things.pth --step 1`
<br /> 
`python video_completion.py --path datasets/[your_folder_name]/rgb --model weight/raft-things.pth --step 4`
<br /> 
`python video_completion.py --path datasets/[your_folder_name]/rgb --model weight/raft-things.pth --step 8`
<br /> 
(As mentioned in section 7, we use multiple time scales (1, 4, 8) to reinforce consistency.)
<br /> 
`mv RAFT_result/datasets[your_folder_name]rgb/*flow* datasets/[your_folder_name]`

6. Confidence estimate for flows:
`python datasets/confidence.py --dataroot ./datasets/[your_folder_name] --step 1`
<br /> 
`python datasets/confidence.py --dataroot ./datasets/[your_folder_name] --step 4`
<br /> 
`python datasets/confidence.py --dataroot ./datasets/[your_folder_name] --step 8`

7. Find the simpler frames if you want to use the tricks in Section 7. Separate the frame indices as in `data/noninteraction_ind.txt`. If there's no such frames, simply write "0, 1" in that file.

8. After Stage 1, run `python gen_foregroundPosEx.py` to generate positive examples for the foreground. Run `data/gen_backgroundPosEx.ipynb` to generate positive examples for the background.

9. In short, there should be these folders in data/[your_folder_name]:
forward_flow_step1, forward_flow_step4, forward_flow_step8
<br /> 
backward_flow_step1, backward_flow_step4, backward_flow_step8
<br /> 
confidence_step1, confidence_step4, confidence_step8
<br /> 
homographies.txt
<br /> 
mask_nocushionmask (2 subfolders: "01", "02")
<br /> 
mask (1 subfolder containing the segmentaion mask of the foreground object: "01")
<br /> 
noninteraction_ind
<br /> 
zbar.pth (Automatically generated to make sure the model starts with a fixed random noise.)
<br /> 
dis_real_l1, dis_real_l2 (Generated after running Stage 1.)

