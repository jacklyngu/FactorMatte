# Copyright 2021 Erika Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from third_party.models.base_model import BaseModel
from third_party.models.networks_lnr import MaskLoss, cal_alpha_reg, MaskLoss_Noreduce

# from causal.discriminator_dataset import RGBA2RGB_diff
from PIL import Image
from third_party.util.util import tensor2im
from . import networks

import numpy as np
import utils
import os
from causal.discriminator import *

# torch.autograd.set_detect_anomaly(True)


class FactormatteGANFlipModel(BaseModel):
    """This class implements the layered neural rendering model for decomposing a video into layers."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode="factormatte_GAN")
        if is_train:
            parser.add_argument(
                "--lambda_plausible_layers",
                type=str,
                default="0,1e-3,1e-3",
                help="punish inplausible outputs of factormatte, lambda for each layer starting from 1, separated by comma no space.",
            )
            parser.add_argument(
                "--lambda_D_layers",
                type=str,
                default="0,1e-3,1e-3",
                help="for training the discriminator, lambda for each layer starting from 1, separated by comma no space.",
            )
            parser.add_argument(
                "--lambda_adj_reg",
                type=float,
                default=0.001,
                help="regularizer for cam adjustment",
            )
            parser.add_argument(
                "--lambda_recon_flow",
                type=float,
                default=1.0,
                help="flow recon loss weight",
            )
            parser.add_argument(
                "--lambda_recon_warp",
                type=float,
                default=0.0,
                help="warped recon loss weight",
            )
            parser.add_argument(
                "--lambda_alpha_warp",
                type=float,
                default=0.005,
                help="alpha warping  loss weight",
            )
            parser.add_argument(
                "--lambda_alpha_l1",
                type=float,
                default=0.01,
                help="alpha L1 sparsity loss weight",
            )
            parser.add_argument(
                "--lambda_alpha_l0",
                type=float,
                default=0.005,
                help="alpha L0 sparsity loss weight",
            )
            parser.add_argument(
                "--alpha_l1_rolloff_epoch",
                type=int,
                default=200,
                help="turn off L1 alpha sparsity loss weight after this epoch",
            )
            parser.add_argument(
                "--lambda_mask",
                type=float,
                default=50,
                help="layer matting loss weight",
            )
            parser.add_argument(
                "--mask_thresh",
                type=float,
                default=0.02,
                help="turn off masking loss when error falls below this value",
            )
            parser.add_argument(
                "--mask_loss_rolloff_epoch",
                type=int,
                default=-1,
                help="decrease masking loss after this epoch; if <0, use mask_thresh instead",
            )
            # parser.add_argument(
            #     "--NF_FG_alpha_loss_rolloff_epoch",
            #     type=int,
            #     default=-1,
            #     help="decrease masking loss after this epoch; if <0, use mask_thresh instead",
            # )
            parser.add_argument(
                "--cam_adj_epoch",
                type=int,
                default=0,
                help="when to start optimizing camera adjustment params",
            )
            parser.add_argument(
                "--jitter_rgb",
                type=float,
                default=0,
                help="amount of jitter to add to RGB",
            )
            parser.add_argument(
                "--jitter_epochs",
                type=int,
                default=0,
                help="number of epochs to jitter RGB",
            )
            parser.add_argument(
                "--flow_epochs",
                type=int,
                default=np.inf,
                help="number of epochs after which to bring back loss_recon_flow",
            )
            parser.add_argument(
                "--flow_reg_epochs",
                type=int,
                default=np.inf,
                help="number of epochs after which to slowly increase flow regularization of the residual layer",
            )
            parser.add_argument(
                "--update_D_epochs",
                default=1,
                type=int,
                help="Frequency to update discriminator. Only used in GAN training.",
            )
            parser.add_argument(
                "--update_G_epochs",
                default=1,
                type=int,
                help="Frequency to update discriminator. Only used in GAN training.",
            )
            parser.add_argument(
                "--lambda_rgb_warp",
                type=float,
                default=0.0,
                help="warped RGB loss weight",
            )
            parser.add_argument(
                "--lambda_recon",
                type=float,
                default=1.0,
                help="Coefficient for RGBA recon loss.",
            )
            parser.add_argument(
                "--lambda_recon_1",
                type=float,
                default=0.0,
                help="Coefficient for RGBA recon loss.",
            )
            parser.add_argument(
                "--lambda_recon_3",
                type=float,
                default=0.0,
                help="Coefficient for RGBA recon loss.",
            )

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options
        """
        BaseModel.__init__(self, opt)
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = [
            "target_image",
            "reconstruction",
            "rgba_vis",
            "alpha_vis",
            "rgb_vis",
            "input_vis",
        ]
        self.fg_ind = opt.fg_layer_ind
        self.res_ind = 3 - opt.fg_layer_ind
        self.model_names = ["factormatteGAN"]
        self.netfactormatteGAN = networks.define_factormatte(
            opt, gpu_ids=self.gpu_ids
        )
            
        self.do_cam_adj = True
        if self.isTrain or opt.gradient_debug:
            self.dis_n_layers = [int(i) for i in opt.n_layers.split(",")]
            self.dis_strides = [int(i) for i in opt.strides.split(",")]
            self.dis_num_Ds = [int(i) for i in opt.num_Ds.split(",")]
            # even if a certain layer doesn't need, you still should have "placeholders" at that index.
            # Start from l0, which is the BG.
            self.discriminators = []
            for i in range(len(self.dis_num_Ds)):
                need = self.dis_num_Ds[i] != 0
                print("need ith discriminator", i, need)
                if need:
#                     self.discriminators.append(NLayerDiscriminator(opt).cuda())
#                     print('using single scale 70x70 dis.')
                    self.discriminators.append(
                        MultiscaleDiscriminator(
                            opt,
                            int(self.dis_strides[i]),
                            int(self.dis_num_Ds[i]),
                            int(self.dis_n_layers[i]),
                        ).cuda()
                     )
                else:
                    self.discriminators.append(None)
            assert (
                len(self.dis_num_Ds)
                == len(self.dis_n_layers)
                == len(self.dis_strides)
                == len(self.discriminators)
            )
            self.plausibleLoss = nn.BCEWithLogitsLoss(reduction="none")
            self.criterionLoss = torch.nn.L1Loss()
            self.lambda_plausibles = [0.0, 1.0, 1.0]
        if self.isTrain:
            self.lambda_plausibles = [
                float(i) for i in opt.lambda_plausible_layers.split(",")
            ]
            self.lambda_Ds = [float(i) for i in opt.lambda_D_layers.split(",")]
            assert (
                len(self.lambda_plausibles)
                == len(self.lambda_Ds)
                == len(self.dis_num_Ds)
            )
            self.setup_train(opt)
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def update_learning_rate(self, update_which):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for i in update_which:
            scheduler = self.schedulers[i]
            old_lr = self.optimizers[i].param_groups[0]["lr"]
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()
            lr = self.optimizers[i].param_groups[0]["lr"]
            if old_lr != lr:
                print("%d th optimizer, learning rate %.7f -> %.7f" % (i, old_lr, lr))

    def setup_train(self, opt):
        """Setup the model for training mode."""
        print("setting up model")
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = [
            "total",
            "recon",
            "alpha_reg",
            "mask",
            "recon_flow",
            "recon_warp",
            "alpha_warp",
            "adj_reg",
            "plausible",
            "D",
            "rgb_warp",
        ]
        self.visual_names = [
            "target_image",
            "reconstruction",
            "rgba_vis",
            "alpha_vis",
            "rgb_vis",
            "flow_vis",
            "mask_vis",
            "mask_loss_vis",
            "input_vis",
        ]
        # model_v 42 is the front_alpha, back_alpha with the same RGB extension from one alpha for one RGB. Under development.
        # if opt.model_v == 42:
        #     self.visual_names.append("twoa_vis")
        self.lambda_names = [
            "mask",
            "adj_reg",
            "recon_flow",
            "recon_warp",
            "alpha_warp",
            "alpha_l0",
            "alpha_l1",
            "rgb_warp",
            "recon",
            "recon_1",
            "recon_3",
        ]
        self.do_cam_adj = False
        # Bind with test setups so comment out the overlap here.
        # self.criterionLoss = torch.nn.L1Loss()
        # self.plausibleLoss = nn.BCEWithLogitsLoss()
        self.criterionLoss_noreduce = torch.nn.L1Loss(reduction="none")
        self.criterionLossMask = MaskLoss().to(self.device)
        self.criterionLossMask_noreduce = MaskLoss_Noreduce().to(self.device)
        for name in self.lambda_names:
            if isinstance(name, str):
                setattr(self, "lambda_" + name, getattr(opt, "lambda_" + name))
        self.mask_loss_rolloff_epoch = opt.mask_loss_rolloff_epoch
        self.jitter_rgb = opt.jitter_rgb
        self.optimizer = torch.optim.Adam(self.netfactormatteGAN.parameters(), lr=opt.lr)
        self.optimizers = [self.optimizer]
        discriminator_params = []
        for d in self.discriminators:
            if d:
                discriminator_params.extend(list(d.parameters()))
        self.optimizer_D = None
        if discriminator_params:
            self.optimizer_D = torch.optim.Adam(discriminator_params, lr=opt.lr)
            self.optimizers.append(self.optimizer_D)

        print(
            "Make sure you want to use",
            self.opt.discriminator_transform,
            "for PatchGAN.",
        )
        if self.opt.discriminator_transform == "randomresizedcrop":
            self.discriminator_preprocessing = transforms.RandomResizedCrop(
                min(opt.height, opt.width)
            )
        elif self.opt.discriminator_transform == "randomcrop":
            self.discriminator_preprocessing = transforms.RandomCrop(
                min(opt.height, opt.width)
            ) 
        elif self.opt.discriminator_transform == "none":
            self.discriminator_preprocessing = None
        self.writer = SummaryWriter(log_dir="517_runs/" + opt.name)
        self.maxpool = nn.MaxPool3d((2, 1, 1), stride=1)
        self.consistLoss = torch.nn.MSELoss()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.target_image = input["image"].to(self.device)
        if self.isTrain and self.jitter_rgb > 0:
            # add brightness jitter to rgb
            self.target_image += self.jitter_rgb * torch.randn(
                self.target_image.shape[0], 1, 1, 1
            ).to(self.device)
            self.target_image = torch.clamp(self.target_image, -1, 1)
        self.input = input["input"].to(self.device)
        self.input_bg_flow = input["bg_flow"].to(self.device)
        self.input_bg_warp = input["bg_warp"].to(self.device)
        self.mask = input["mask"].to(self.device)
        self.flow_gt = input["flow"].to(self.device)
        self.flow_confidence = input["confidence"].to(self.device)
        self.jitter_grid = input["jitter_grid"].to(self.device)
        self.image_paths = input["image_path"]
        self.index = input["index"]

        #############################
        self.dis_gt = input["dis_gt"].to(self.device)
        self.dis_gt_exist = input["dis_gt_exist"].to(self.device)
        if not self.opt.bg_noise:
            self.bg_gt = input["bg_gt"].to(self.device)
        #############################
        self.n_layers = self.mask.shape[1] // 2
        if self.isTrain or self.opt.gradient_debug:
            self.dis_reals = []
            self.dis_fakes = []
            for i in range(len(self.discriminators)):
                # If discriminator exisits, dis_real must exist
                if self.discriminators[i]:
                    self.dis_reals.append(input["dis_real_l" + str(i)].to(self.device))
                else:
                    self.dis_reals.append(None)
                # However that's not the case for dis_fake, it's not necessary so we directly check whether it exists.
                if "dis_fake_l" + str(i) in input:
                    self.dis_fakes.append(input["dis_fake_l" + str(i)].to(self.device))
                else:
                    self.dis_fakes.append(None)

            assert (
                len(self.dis_reals)
                == len(self.dis_fakes)
                == self.n_layers
                == len(self.discriminators)
            )

    def gen_crop_params(self, orig_h, orig_w, crop_size=256):
        """Generate random square cropping parameters."""
        starty = np.random.randint(orig_h - crop_size + 1)
        startx = np.random.randint(orig_w - crop_size + 1)
        endy = starty + crop_size
        endx = startx + crop_size
        return starty, endy, startx, endx

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        outputs = self.netfactormatteGAN(
            self.input,
            self.input_bg_flow,
            self.input_bg_warp,
            self.jitter_grid,
            self.index,
            self.do_cam_adj,
        )
        #############################
        self.dis_gt = self.rearrange2batchdim(self.dis_gt)
        self.dis_gt_exist = self.rearrange2batchdim(self.dis_gt_exist)
        if not self.opt.bg_noise:
            self.bg_gt = self.rearrange2batchdim(self.bg_gt)
        #############################
        # rearrange t, t+1 to batch dimension
        self.target_image = self.rearrange2batchdim(self.target_image)
        self.mask = self.rearrange2batchdim(self.mask)
        self.flow_confidence = self.rearrange2batchdim(self.flow_confidence)
        self.flow_gt = self.rearrange2batchdim(self.flow_gt)
        reconstruction_rgb = self.rearrange2batchdim(outputs["reconstruction_rgb"])
        self.reconstruction = reconstruction_rgb[:, :3]
        self.alpha_composite = reconstruction_rgb[:, 3:]
        self.reconstruction_warped = outputs["reconstruction_warped"]
        self.alpha_warped = outputs["alpha_warped"]
        self.rgb_warped = outputs["rgb_warped"]
        self.bg_offset = self.rearrange2batchdim(outputs["bg_offset"])
        self.brightness_scale = self.rearrange2batchdim(outputs["brightness_scale"])
        self.reconstruction_flow = self.rearrange2batchdim(
            outputs["reconstruction_flow"]
        )
        self.output_flow = self.rearrange2batchdim(outputs["layers_flow"])
        self.output_rgba = self.rearrange2batchdim(outputs["layers_rgba"])

        # if self.opt.model_v == 42:
        #     self.alpha_unnorm_backs = self.rearrange2batchdim(
        #         outputs["alpha_unnorm_backs"]
        #     )
        #     self.alpha_unnorm_fronts = self.rearrange2batchdim(
        #         outputs["alpha_unnorm_fronts"]
        #     )
        #     self.twoa_vis = torch.cat(
        #         [self.alpha_unnorm_backs, self.alpha_unnorm_fronts], -2
        #     )

        layers = self.output_rgba.data
        if not self.opt.no_bg:
            layers[:, -1, 0] = 1  # Background layer's alpha is always 1
        layers = torch.cat([layers[:, :, l] for l in range(self.n_layers)], -2)
        self.alpha_vis = layers[:, 3:]
        self.rgb_vis = layers[:, :3]
        self.rgba_vis = layers  # [:, :4]
        self.mask_vis = torch.cat(
            [self.input[:, l : l + 1, 0] for l in range(self.n_layers)], -2
        )
        self.mask_loss_vis = torch.cat(
            [self.mask[:, l : l + 1] for l in range(self.n_layers)], -2
        )
        # self.input size [bs, #layer, 2*#channel, h, w]
        self.input_vis = torch.cat(
            [self.input[:, l, -4:-1] for l in range(self.n_layers)], -2
        )
        self.flow_vis = torch.cat(
            [self.output_flow[:, :, l] for l in range(self.n_layers)], -2
        )
        self.flow_vis = utils.tensor_flow_to_image(self.flow_vis[0].detach()).unsqueeze(
            0
        )  # batchsize 1

        reconstruction_rgb_no_cube = self.rearrange2batchdim(
            outputs["reconstruction_rgb_no_cube"]
        )
        self.reconstruction_rgb_no_cube = reconstruction_rgb_no_cube[:, :3]
        self.alpha_composite_no_cube = reconstruction_rgb_no_cube[:, 3:]
#         self.transfer_detail()
        self.rgba = self.output_rgba

        if self.isTrain or self.opt.gradient_debug:
            self.input_for_discriminators = []
            for i in range(self.n_layers):
                if self.dis_fakes[i] is not None:
                    self.dis_fakes[i] = self.rearrange2batchdim(self.dis_fakes[i])
                    if self.isTrain and self.discriminator_preprocessing is not None:
                        self.dis_fakes[i] = self.discriminator_preprocessing(
                            self.dis_fakes[i]
                        )

                if self.dis_reals[i] is not None:
                    self.dis_reals[i] = self.rearrange2batchdim(self.dis_reals[i])
                    # No more normalization to [0,1]
                    normalized_rgba = self.rgba[:, :, i].squeeze()
                    if self.opt.rgba_GAN == "L":
                        # Only keep alpha layer
                        normalized_rgba = normalized_rgba[:, -1:]
                    elif self.opt.rgba_GAN == "RGB":
                        # Only keep RGB layer
                        normalized_rgba = normalized_rgba[:, :-1]
                    if self.isTrain and self.discriminator_preprocessing is not None:
                        normalized_rgba = self.discriminator_preprocessing(
                            normalized_rgba
                        )
                        # self.dis_reals[i] = self.discriminator_preprocessing(
                        #     self.dis_reals[i]
                        # )

                    mask = torch.ones(
                        (
                            normalized_rgba.size(0),
                            1,
                            normalized_rgba.size(2),
                            normalized_rgba.size(3),
                        ),
                        requires_grad=False,
                    ).cuda()
                    self.input_for_discriminators.append(
                        torch.cat((normalized_rgba, mask), 1)
                    )
                else:
                    self.input_for_discriminators.append(None)


    def backward(self, total_iters):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        b_sz = self.target_image.shape[0]
        mask_fg = self.dis_gt

        if self.lambda_recon != 0:
            loss_recon = self.criterionLoss(
                self.reconstruction[:, :3], self.target_image
            )
            self.loss_recon = self.lambda_recon * loss_recon
        self.loss_a_front_reg = 0
        # if self.lambda_recon_1 != 0 and self.opt.model_v == 42:
        #     self.loss_a_front_reg = self.lambda_recon_1 * torch.sum(
        #         torch.abs(1.0 + self.alpha_unnorm_fronts)
        #     )

        self.loss_recon_noninter_fg = 0
        if self.lambda_recon_3 != 0:
            assert (
                self.opt.init_flowmask
            ), "this must be used with init_flowmask, otherwise would be in conflict with criterionLossMask"
            # a_l2 = self.output_rgba[:, 3:, 2] * 0.5 + 0.5
            self.loss_recon_noninter_fg = (
                self.lambda_recon_3
                * (
                    self.dis_gt_exist.view(b_sz, 1, 1, 1)
                    * self.criterionLossMask_noreduce(
                        self.output_rgba[:, 3:, self.fg_ind],
                        self.mask[:, self.fg_ind : self.fg_ind + 1],
                        mask_fg
                    )
                ).mean()
            )

        self.loss_alpha_reg = cal_alpha_reg(
            self.alpha_composite * 0.5 + 0.5, self.lambda_alpha_l1, self.lambda_alpha_l0
        )
        # alpha_layers and self.mask size [bs*2, #layer, h, w]
        alpha_layers = self.output_rgba[:, 3]
        if self.opt.stage < 3:
            # Apply only on the FG mask
            criterionLossMask = self.criterionLossMask(
                alpha_layers[:, self.fg_ind : self.fg_ind + 1],
                self.mask[:, self.fg_ind : self.fg_ind + 1],
            )
        else:
            criterionLossMask = self.criterionLossMask(
                alpha_layers,
                self.mask,
            )
        self.loss_mask =  self.lambda_mask * criterionLossMask

        # Regularize on the alpha of the residual layer.
        # self.output_rgba size [bs, #channel, #layer, h, w]
        self.loss_grad_exclusive = 0.0
        self.loss_flow_div_reg = 0
        LossFlowDiv, cos_sim = 0, 0
        if self.output_rgba.size(2) == 3:
            # Looking at the entire pixel, not the pixel's x and y flow recon separately.
            criterionLossFlow_l1 = self.criterionLoss_noreduce(
                self.flow_confidence * self.output_flow[:, :, 1],
                self.flow_confidence * self.flow_gt,
            ).sum(1)
            criterionLossFlow_l2 = self.criterionLoss_noreduce(
                self.flow_confidence * self.output_flow[:, :, 2],
                self.flow_confidence * self.flow_gt,
            ).sum(1)
            div_mask_l1 = (criterionLossFlow_l1 > criterionLossFlow_l2).int()
            div_mask_l2 = 1 - div_mask_l1
            # Where we should use l1 for div loss - where l1 flow recon loss is smaller than l2
            criterionLossFlow_pool = (
                criterionLossFlow_l1 * div_mask_l2 + criterionLossFlow_l2 * div_mask_l1
            )
            criterionLossFlow = criterionLossFlow_pool.mean()
        # Only 2 layers: BG and FG, no more distinction between residual and FG.
        elif self.output_rgba.size(2) == 2:
            # Use the original factormatte's flow loss
            criterionLossFlow = self.criterionLoss(
                self.flow_confidence * self.reconstruction_flow,
                self.flow_confidence * self.flow_gt,
            )
        self.loss_recon_flow = self.lambda_recon_flow * criterionLossFlow
        #################################
        alpha_t = alpha_layers[: b_sz // 2]
        criterionLossAlpha = self.criterionLoss(self.alpha_warped[:, 0], alpha_t)
        self.loss_alpha_warp = self.lambda_alpha_warp * criterionLossAlpha
        target_rgb_t = self.target_image[: b_sz // 2]
        criterionLossWarped = self.criterionLoss(
            self.reconstruction_warped, target_rgb_t
        )
        self.loss_recon_warp = self.lambda_recon_warp * criterionLossWarped
        brightness_reg = self.criterionLoss(
            self.brightness_scale, torch.ones_like(self.brightness_scale)
        )
        offset_reg = self.bg_offset.abs().mean()
        self.loss_adj_reg = self.lambda_adj_reg * (brightness_reg + offset_reg)
        # self.reconstruction size [32, 3, 256, 448]

        rgb_layers = self.output_rgba[:, :3]
        rgb_t = rgb_layers[: b_sz // 2]
        # self.rgb_warped and rgb_t size [8, 3, 3, 256, 448]; self.alpha_warped [8, 1, 3, 256, 448]
        criterionLossRGBWarped = self.criterionLoss(self.rgb_warped, rgb_t)
        self.loss_rgb_warp = self.lambda_rgb_warp * criterionLossRGBWarped

        self.loss_plausible = 0
        for i in range(self.n_layers):
            if self.discriminators[i] is not None:
                # Let the discriminator look at the entire image when training the generator
                validity, _ = self.discriminators[i](self.input_for_discriminators[i])
                valid = torch.ones_like(validity, requires_grad=False)
                g_loss = self.plausibleLoss(validity, valid).mean()
                self.loss_plausible = (
                    self.loss_plausible + self.lambda_plausibles[i] * g_loss
                )

        self.loss_total = (
            self.loss_recon
            + self.loss_alpha_reg
            + self.loss_mask
            + self.loss_recon_flow
            + self.loss_alpha_warp
            + self.loss_recon_warp
            + self.loss_adj_reg
            + self.loss_plausible
            + self.loss_rgb_warp
            + self.loss_a_front_reg
            + self.loss_recon_noninter_fg
            + self.loss_flow_div_reg
        )

        self.loss_total.backward()
        torch.cuda.empty_cache()
        for name in self.loss_names:
            if isinstance(name, str):
                self.writer.add_scalar(
                    "[train] loss_" + name, getattr(self, "loss_" + name), total_iters
                )
        for name in self.lambda_names:
            if isinstance(name, str):
                self.writer.add_scalar(
                    "[train] lambda_" + name,
                    getattr(self, "lambda_" + name),
                    total_iters,
                )
        self.writer.add_scalar("[train] loss_recon_raw", self.loss_recon, total_iters)
        self.writer.add_scalar(
            "[train] loss_alpha_reg_raw", self.loss_alpha_reg, total_iters
        )
        self.writer.add_scalar("[train] loss_mask_raw", criterionLossMask, total_iters)
        self.writer.add_scalar(
            "[train] loss_recon_flow_raw", criterionLossFlow, total_iters
        )
        self.writer.add_scalar(
            "[train] loss_alpha_warp_raw", criterionLossAlpha, total_iters
        )
        self.writer.add_scalar(
            "[train] loss_recon_warp_raw", criterionLossWarped, total_iters
        )
        self.writer.add_scalar(
            "[train] loss_adj_reg_raw", brightness_reg + offset_reg, total_iters
        )
        #         self.writer.add_scalar("[train] loss_plausible_l1_raw", g_loss_l1, total_iters)
        #         self.writer.add_scalar("[train] loss_plausible_l2_raw", g_loss_l2, total_iters)
        self.writer.add_scalar(
            "[train] loss_residual_alpha_reg_raw",
            torch.sum(torch.abs(1.0 + self.output_rgba[:, -1, self.res_ind])),
            total_iters,
        )
        self.writer.add_scalar(
            "[train] loss_residual_flow_reg_raw",
            torch.sum(torch.abs(self.output_flow[:, :, self.res_ind])),
            total_iters,
        )
        self.writer.add_scalar("[train] loss_grad_exclusive_raw", cos_sim, total_iters)
        self.writer.add_scalar(
            "[train] loss_rgb_warp_raw", criterionLossRGBWarped, total_iters
        )
        self.writer.add_scalar(
            "[train] loss_flow_div_reg_raw", LossFlowDiv, total_iters
        )
        self.writer.flush()

    def update_dis(self, total_iters):
        """Train the discriminators.

        Args:
            total_iters (int): Number of iter so far; for logging.
        """
        # Loss for real images
        d_real_loss = 0
        for i in range(self.n_layers):
            if self.discriminators[i] is not None:
                validity_real, valid_real_patches = self.discriminators[i](
                    self.dis_reals[i]
                )
                valid_real_patches.requires_grad = False
                valid = torch.ones_like(validity_real, requires_grad=False)
                d_loss = (
                    self.plausibleLoss(validity_real, valid) * valid_real_patches
                ).mean()
                d_real_loss = d_real_loss + self.lambda_Ds[i] * d_loss
        # Loss for fake images
        d_fake_loss = 0
        for i in range(self.n_layers):
            if self.discriminators[i] is not None:
                if self.dis_fakes[i] is not None:
                    # input_for_discriminators = self.dis_fakes[i]
                    input_for_discriminators = torch.cat(
                        (self.input_for_discriminators[i].detach(), self.dis_fakes[i]),
                        0,
                    )
                    # fake_inputs_np = (
                    #     1 + self.dis_fakes[i].cpu().detach().permute(0, 2, 3, 1).numpy()
                    # ) / 2
                    # print(str(self.index[0, 0].cpu().detach().item()))
                    # np.save(
                    #     str(self.index[0, 0].cpu().detach().item()) + "fake_inputs.npy",
                    #     fake_inputs_np,
                    # )
                else:
                    input_for_discriminators = self.input_for_discriminators[i].detach()
                validity_fake, valid_fake_patches = self.discriminators[i](
                    input_for_discriminators
                )
                valid_fake_patches.requires_grad = False
                fake = torch.zeros_like(validity_fake, requires_grad=False)
                d_loss = (
                    self.plausibleLoss(validity_fake, fake) * valid_fake_patches
                ).mean()
                d_fake_loss = d_fake_loss + self.lambda_Ds[i] * d_loss

        self.loss_D = (d_real_loss + d_fake_loss) / 2
        self.loss_D.backward()
        self.writer.add_scalar("[train] loss_real_D_raw", d_real_loss, total_iters)
        self.writer.add_scalar("[train] loss_fake_D_raw", d_fake_loss, total_iters)
        self.writer.flush()

    def gradient_debug(self, total_iters):
        """For visualizing the gradients of the discriminators. Set --gradient_debug and will stop the actual training.
        """
        def set_grad(var):
            def hook(grad):
                var.grad = grad

            return hook

        # assert self.opt.batch_size == 1
        # self.input_for_discriminator.register_hook(
        #     set_grad(self.input_for_discriminator)
        # )
        self.loss_plausible = 0
        for i in range(self.n_layers):
            if self.discriminators[i] is not None:
                validity, valid_patches = self.discriminators[i](
                    self.input_for_discriminators[i]
                )
                valid = torch.ones_like(validity, requires_grad=False)
                g_loss = (self.plausibleLoss(validity, valid) * valid_patches).mean()
                self.loss_plausible = (
                    self.loss_plausible + self.lambda_plausibles[i] * g_loss
                )

        self.output_rgba.register_hook(set_grad(self.output_rgba))
        self.loss_plausible.backward()
        # print(self.output_rgba.grad.size())
        # input_grad_map = self.input_for_discriminator.grad.detach().cpu().numpy()
        # np.save(
        #     os.path.join(
        #         self.opt.checkpoints_dir,
        #         self.opt.name,
        #         str(total_iters) + "_inputs_grad_map.npy",
        #     ),
        #     input_grad_map,
        # )
        grad_map = self.output_rgba.grad.detach().cpu().permute(0, 2, 3, 4, 1).numpy()
        return grad_map

    def rearrange2batchdim(self, tensor):
        n_c = tensor.shape[1]
        return torch.cat((tensor[:, : n_c // 2], tensor[:, n_c // 2 :]))

    def optimize_parameters(self, total_iters, epoch):
        """Update network weights; it will be called in every training iteration."""
        if (epoch - 1) % self.opt.update_G_epochs == 0:
            self.forward()
        else:
            with torch.no_grad():
                self.forward()
        # Based on pix2pix implementation, first D then G.
        if (epoch - 1) % self.opt.update_D_epochs == 0:
            self.loss_D = 0
            if self.optimizer_D is not None:
                for i in range(self.n_layers):
                    if self.discriminators[i] is not None:
                        self.set_requires_grad(self.discriminators[i], True)

                self.optimizer_D.zero_grad()
                self.update_dis(total_iters)
                self.optimizer_D.step()
                for i in range(self.n_layers):
                    if self.discriminators[i] is not None:
                        self.set_requires_grad(self.discriminators[i], False)
        if (epoch - 1) % self.opt.update_G_epochs == 0:
            self.optimizer.zero_grad()
            self.backward(total_iters)
            self.optimizer.step()

    def update_lambdas(self, epoch):
        """
        Update loss weights based on current epochs and losses.
        """
        # Changed to >= from = to include continue_train by loading cases.
        if epoch >= self.opt.alpha_l1_rolloff_epoch:
            self.lambda_alpha_l1 = 0
        if self.mask_loss_rolloff_epoch >= 0:
            if epoch == 2 * self.mask_loss_rolloff_epoch:
                self.lambda_mask = 0
        elif epoch > self.opt.epoch_count:
            if self.loss_mask < self.opt.mask_thresh * self.opt.lambda_mask:
                self.mask_loss_rolloff_epoch = epoch
                self.lambda_mask *= 0.1
        # if epoch >= self.NF_FG_alpha_loss_rolloff_epoch:
        #     self.lambda_recon_3 /= 2
        if epoch >= self.opt.jitter_epochs:
            self.jitter_rgb = 0
        self.do_cam_adj = epoch >= self.opt.cam_adj_epoch

    def compute_visuals(self, total_iters):
        for i in range(self.n_layers):
            if self.discriminators[i] is not None:
                self.set_requires_grad(self.discriminators[i], False)

        if self.opt.gradient_debug:
            self.forward()
            grad_map = self.gradient_debug(total_iters)
            np.save(
                os.path.join(
                    self.opt.checkpoints_dir,
                    self.opt.name,
                    str(total_iters) + "_grad_map.npy",
                ),
                grad_map,
            )
            self.optimizer.zero_grad()
            # Not necessary just to make sure
            self.optimizer_D.zero_grad()
        else:
            with torch.no_grad():
                self.forward()

    def transfer_detail(self):
        """Transfer detail to layers."""
        residual = self.target_image - self.reconstruction
        transmission_comp = torch.zeros_like(self.target_image[:, 0:1])
        rgba_detail = self.output_rgba.clone()  # used to have no clone()
        n_layers = self.output_rgba.shape[2]
        for i in range(
            n_layers - 1, 0, -1
        ):  # Don't do detail transfer for background layer, due to ghosting effects.
            transmission_i = 1.0 - transmission_comp
            alpha_i = self.output_rgba[:, 3:4, i] * 0.5 + 0.5
            #             rgba_detail[:, :3, i] += transmission_i * residual
            # x = (transmission_i * residual).detach().cpu().permute(0, 2, 3, 1).numpy()
            # np.save(str(self.index[0, 0].item()) + "_" + str(i) + "puddle.npy", x)
            rgba_detail[:, :3, i] = rgba_detail[:, :3, i] + transmission_i * residual * alpha_i
            transmission_comp = alpha_i + (1.0 - alpha_i) * transmission_comp
            if i == 1:
                self.reconstruction_rgb_no_cube[:, :3] = self.reconstruction_rgb_no_cube[:, :3] + transmission_i * residual
        self.rgba = torch.clamp(rgba_detail, -1, 1)
        self.reconstruction_rgb_no_cube = torch.clamp(self.reconstruction_rgb_no_cube, -1, 1)
        return residual 
        
    def get_results(self):
        """Return results. This is different from get_current_visuals, which gets visuals for monitoring training.

        Returns a dictionary:
            original - - original frame
            recon - - reconstruction
            rgba_l* - - RGBA for each layer
            mask_l* - - mask for each layer
        """
        residual = self.transfer_detail()
        # self.rgba = self.output_rgba.clone()
        results = {
            "reconstruction_no_cube": self.reconstruction_rgb_no_cube,
            "residual": residual,
            "reconstruction": self.reconstruction,
            "original": self.target_image,
            "reconstruction_flow": utils.tensor_flow_to_image(
                self.reconstruction_flow[0]
            ).unsqueeze(0),
            "flow_gt": utils.tensor_flow_to_image(self.flow_gt[0], global_max=5).unsqueeze(0),
            "bg_offset": utils.tensor_flow_to_image(self.bg_offset[0]).unsqueeze(0),
            "brightness_scale": self.brightness_scale - 1.
        }
        # if self.opt.model_v == 42:
        #     results["rgba_l1_alpha_unnorm_backs"] = self.alpha_unnorm_backs
        #     results["rgba_l1_alpha_unnorm_fronts"] = self.alpha_unnorm_fronts
        n_layers = self.rgba.shape[2]
        if not self.opt.no_bg:
            self.rgba[:, -1:, 0] = 1  # background layer's alpha is 1
        # rgba size [2, #channel, #layer, 256, 448]
        flow_layers = self.output_flow  # (self.rgba[:, -1:] * .5 + .5) *
        # Split layers
        for i in range(n_layers):
            results[f"rgb_l{i}"] = self.rgba[:, :3, i]
            results[f"mask_l{i}"] = self.mask[:, i : i + 1]
            results[f"a_l{i}"] = self.rgba[:, 3:4, i]
            results[f"rgb_warped_l{i}"] = self.rgb_warped[:, :, i]
            results[f"a_warped_l{i}"] = self.alpha_warped[:, :, i]
            results[f"rgba_l{i}"] = self.rgba[:, :, i]
            results[f"flow_l{i}"] = utils.tensor_flow_to_image(
                flow_layers[0, :, i], global_max=5
            ).unsqueeze(0)
        return results
