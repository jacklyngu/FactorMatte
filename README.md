# FactorMatte
## Environment
conda create -n factormatte python=3.9 anaconda

conda activate factormatte

Use conda or pip to install requirements.txt

## Example Video
### Download Dataset and put into the datasets/ folder
https://drive.google.com/file/d/1-nZ9VA8bqRvll_4HEPGOxihIJ4o8y0kY/view?usp=sharing

### Stage 1
python train_GAN.py --name sand_car_3layer_v4_rgbwarp1e-1_alphawarp1e-1_flowrecon1e-2 --stage 1 --dataset_mode omnimatte_GANCGANFlip148 --model omnimatte_GANFlip --dataroot ./datasets/sand_car --height 192 --width 288  --save_by_epoch --prob_masks --lambda_rgb_warp 1e-1 --lambda_alpha_warp 1e-1 --model_v 4 --residual_noise --strides 0,0,0 --num_Ds 0,0,0 --n_layers 0,0,0 --display_ind 63 --pos_ex_dirs , --batch_size 16 --n_epochs 1200 --bg_noise --gpu_ids 1,0 --lambda_recon_flow 1e-2

<span style="color:blue">Copy the trained weights to the next stage's training folder:</span> cp 1110_checkpoints/sand_car_3layer_v4_rgbwarp1e-1_alphawarp1e-1_flowrecon1e-2/*1200* 1110_checkpoints/sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_noninter_flowmask_flowrecon1e-2

### Stage 2
python train_GAN.py --name sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_noninter_flowmask_flowrecon1e-2 --init_flowmask --lambda_recon_flow 1e-2 --dataset_mode factormatte_GANCGANFlip148 --model factormatte_GANFlip --dataroot ./datasets/sand_car --save_by_epoch --prob_masks --lambda_rgb_warp 1e-1 --lambda_alpha_warp 1e-1 --residual_noise --strides 0,2,2 --num_Ds 0,1,3 --n_layers 0,3,3 --start_ind 0 --noninter_only --width 288 --height 192 --discriminator_transform randomcrop --pos_ex_dirs 0uniform_0gaussian_dark_0flip_0elastic_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_rawframes,0rot_0flip_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_ --gpu_ids 0 --n_epochs 2400 --continue_train --epoch 1200 --stage 2 --display_ind 15

<span style="color:blue">Copy the trained weights to the next stage's training folder:</span> cp 1110_checkpoints/sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_noninter_flowmask_flowrecon1e-2/*2400* 1110_checkpoints/sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_l2arecon1e-1dilate_recon2_148_othersretro_stage22000cont_flowrecon1e-1


### Stage 3
python train_GAN.py --name sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_l2arecon1e-1dilate_recon2_148_othersretro_stage22000cont_flowrecon1e-1 --lambda_recon 2 --lambda_recon_flow 1e-1 --dataset_mode factormatte_GANCGANFlip148 --model factormatte_GANFlip --dataroot ./datasets/sand_car --save_by_epoch --prob_masks --lambda_rgb_warp 1e-1 --lambda_alpha_warp 1e-1 --residual_noise --strides 0,2,2 --num_Ds 0,1,3 --display_ind 63 --init_flowmask --lambda_recon_3 1e-1 --start_ind 0 --discriminator_transform randomcrop --steps 148 --pos_ex_dirs 0uniform_0gaussian_dark_0flip_0elastic_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_rawframes,0rot_0flip_0.25blursigma0.20.2k5_0.25gaussian_noise_std27mean0_ --stage 3 --height 192 --width 288 --n_epochs 3200 --gpu_ids 0 --continue_train --epoch 2400 --overwrite_lambdas

### Pretrained Weights
For convenience, you can also download the weights of any stage for this dataset and start training from there on.

Stage 1 weights: https://drive.google.com/drive/folders/1ERZQNM8nT2Xw9J2yzFzp3QoyxHFEZ7B4?usp=sharing

Stage 2 weights: https://drive.google.com/drive/folders/1boJJ8DwPZxk9hzxUa-4nLW0vVPhXdhPL?usp=sharing

Stage 3 weights: https://drive.google.com/drive/folders/1eDHuIsoON_ou_50sZ7nT4D3luiGxVvyx?usp=sharing

### Generate Results
python test.py --name sand_car_3layer_13GAN1e-3_strides22crop_D1_v4_rgbwarp1e-1_alphawarp1e-1_l2arecon1e-1dilate_recon2_148_othersretro_stage22000cont_flowrecon1e-1 --dataset_mode factormatte_GANCGANFlip148 --model factormatte_GANFlip --dataroot ./datasets/sand_car --gpu_ids 0 --prob_masks --residual_noise --pos_ex_dirs , --epoch 3200 --stage 3 --width 288 --height 192 --init_flowmask --test_suffix texts_to_put_after_fixed_folder_name


## Custom Dataset
TODO
