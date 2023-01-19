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
import torch.nn.functional as F
from third_party.models.networks import init_net
from third_party.models.networks_lnr import ConvBlock
import os
from PIL import Image


###############################################################################
# Helper Functions
###############################################################################
def define_factormatte(opt, gpu_ids=[]):
    """Create a layered neural renderer.

    Parameters:
        nf (int) -- the number of channels in the first/last conv layers
        in_c (int) -- the number of input channels
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a layered neural rendering model.
    """
    net = Factormatte(opt, opt.num_filters, opt.in_c, opt.max_frames, opt.coarseness)
    return init_net(net, gpu_ids)


##############################################################################
# Classes
##############################################################################
class Factormatte(nn.Module):
    """factormatte model for video decomposition.

    Consists of UNet.
    """

    def __init__(self, opt, nf, in_c, max_frames, coarseness):
        super(Factormatte, self).__init__(),
        """Initialize factormatte model.

        Parameters:
            nf (int) -- the number of channels in the first/last conv layers
            in_c (int) -- the number of channels in the input
            max_frames (int) -- max number of frames in video
            coarseness (int) -- controls temporal dimension of camera adjustment params 
        """
        # Define UNet
        self.encoder = nn.ModuleList(
            [
                ConvBlock(nn.Conv2d, in_c, nf, ksize=4, stride=2),
                ConvBlock(
                    nn.Conv2d,
                    nf,
                    nf * 2,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                    activation="leaky",
                ),
                ConvBlock(
                    nn.Conv2d,
                    nf * 2,
                    nf * 4,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                    activation="leaky",
                ),
                ConvBlock(
                    nn.Conv2d,
                    nf * 4,
                    nf * 4,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                    activation="leaky",
                ),
                ConvBlock(
                    nn.Conv2d,
                    nf * 4,
                    nf * 4,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                    activation="leaky",
                ),
                ConvBlock(
                    nn.Conv2d,
                    nf * 4,
                    nf * 4,
                    ksize=4,
                    stride=1,
                    dil=2,
                    norm=nn.BatchNorm2d,
                    activation="leaky",
                ),
                ConvBlock(
                    nn.Conv2d,
                    nf * 4,
                    nf * 4,
                    ksize=4,
                    stride=1,
                    dil=2,
                    norm=nn.BatchNorm2d,
                    activation="leaky",
                ),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                ConvBlock(
                    nn.ConvTranspose2d,
                    nf * 4 * 2,
                    nf * 4,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                ),
                ConvBlock(
                    nn.ConvTranspose2d,
                    nf * 4 * 2,
                    nf * 4,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                ),
                ConvBlock(
                    nn.ConvTranspose2d,
                    nf * 4 * 2,
                    nf * 2,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                ),
                ConvBlock(
                    nn.ConvTranspose2d,
                    nf * 2 * 2,
                    nf,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                ),
                ConvBlock(
                    nn.ConvTranspose2d,
                    nf * 2,
                    nf,
                    ksize=4,
                    stride=2,
                    norm=nn.BatchNorm2d,
                ),
            ]
        )
        self.final_rgb = nn.Sequential(
            ConvBlock(nn.Conv2d, nf, nf, ksize=4, stride=1, norm=nn.BatchNorm2d),
            ConvBlock(nn.Conv2d, nf, 3, ksize=4, stride=1, activation="tanh"),
        )
        self.final_aflow = nn.Sequential(
            ConvBlock(nn.Conv2d, nf, nf // 2, ksize=4, stride=1, norm=nn.BatchNorm2d)
        )
        self.final_a = nn.Sequential(
            ConvBlock(nn.Conv2d, nf // 2, 1, ksize=4, stride=1, activation="tanh")
        )
        self.final_flow = nn.Sequential(
            ConvBlock(nn.Conv2d, nf // 2, 2, ksize=4, stride=1, activation="none")
        )

        self.max_frames = max_frames
        self.bg_offset = nn.Parameter(torch.zeros(1, 2, max_frames // coarseness, 4, 7))
        self.brightness_scale = nn.Parameter(
            torch.ones(1, 1, max_frames // coarseness, 4, 7)
        )
        self.opt = opt

    def render(self, x):
        """Pass inputs for a single layer through UNet.

        Parameters:
            x (tensor) - - sampled texture concatenated with person IDs

        Returns RGBA for the input layer and the final feature maps.
        """
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)
        rgb = self.final_rgb(x)
        aflow = self.final_aflow(x)
        a = self.final_a(aflow)
        flow = self.final_flow(aflow)
        rgba = torch.cat((rgb, a[:, 0:1]), dim=1)
        return rgba, flow, x

    def forward(self, input, bg_flow, bg_warp, jitter, index, do_adj):
        """Forward pass through layered neural renderer.

        1. Split input to t and t+1 since they are concatenated channelwise
        2. Pass to UNet
        3. Composite RGBA outputs and flow outputs
        4. Warp alphas t+1 -> t using predicted flow layers
        5. Concat results t and t+1 channelwise (except warped alphas)

        Parameters:
            input (tensor) - - inputs for all layers, with shape [B, L, C*2, H, W]
            bg_flow (tensor) - - flow for background layer, with shape [B, 2*2, H, W]
            bg_warp (tensor) - - warping grid used to sample background layer from unwrapped background, with shape [B, 2*2, H, W]
            jitter (tensor) - - warping grid used to apply data transformation, with shape [B, 2*2, H, W]
            index (tensor) - - frame indices [B, 2]
            do_adj (bool) - - whether to apply camera adjustment parameters
        """

        b_sz, n_layers, channels, height, width = input.shape
        input_t = input[:, :, : channels // 2]
        input_t1 = input[:, :, channels // 2 :]
        bg_warp = torch.cat((bg_warp[:, :2], bg_warp[:, 2:]))
        bg_flow = torch.cat((bg_flow[:, :2], bg_flow[:, 2:]))
        jitter = torch.cat((jitter[:, :2], jitter[:, 2:]))
        index = index.transpose(0, 1).reshape(-1)
        composite_rgb = None
        compisite_rgb_no_cube = None
        composite_flow = bg_flow
        layers_rgba = []
        layers_flow = []
        alphas_warped = []
        rgbs_warped = []
        composite_warped = None

        # get camera adjustment params
        bg_offset = F.interpolate(
            self.bg_offset,
            (self.max_frames, 4, 7),
            mode="trilinear",
            align_corners=True,
        )
        bg_offset = bg_offset[0, :, index].transpose(0, 1)
        bg_offset = F.grid_sample(
            bg_offset, jitter.permute(0, 2, 3, 1), align_corners=True
        )
        br_scale = F.interpolate(
            self.brightness_scale,
            (self.max_frames, 4, 7),
            mode="trilinear",
            align_corners=True,
        )
        br_scale = br_scale[0, 0, index].unsqueeze(1)
        br_scale = F.grid_sample(
            br_scale, jitter.permute(0, 2, 3, 1), align_corners=True
        )

        for i in range(n_layers):
            # Get RGBA and flow for this layer.
            input_i = torch.cat((input_t[:, i], input_t1[:, i]))
            rgba, flow, last_feat = self.render(input_i)
            alpha = rgba[:, 3:4] * 0.5 + 0.5

            # Update the composite with this layer's RGBA output
            if i == 0:
                if self.opt.isTrain:
                    # web_dir = os.path.join(
                    #     self.opt.results_dir,
                    #     self.opt.name,
                    #     "{}_{}_{}".format(
                    #         self.opt.phase, self.opt.epoch, self.opt.test_suffix
                    #     ),
                    # )
                    web_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
                    if "panorama.png" not in os.listdir(web_dir):
                        from third_party import util

                        panorama = util.util.tensor2im(rgba)
                        panorama_img = Image.fromarray(panorama[:, :, :3])
                        panorama_img.save(os.path.join(web_dir, "panorama.png"))
                        print("panorama BG saved")

                if not self.opt.bg_noise:
                    rgba = input_i[:, -4:]

                # sample from unwrapped background
                rgba = F.grid_sample(
                    rgba, bg_warp.permute(0, 2, 3, 1), align_corners=True
                )
#                 # apply learned background offset
                if do_adj:
                    rgba = warp_flow(rgba, bg_offset.permute(0, 2, 3, 1))
                composite_rgb = rgba
                composite_rgb_no_cube = rgba
                flow = bg_flow
                # for background layer, use prediction for t
                rgba_warped = rgba[:b_sz]
                composite_warped = rgba_warped[:, :3]
            else:
                if i == (3 - self.opt.fg_layer_ind):
                    composite_rgb_no_cube = rgba * alpha + composite_rgb_no_cube * (
                        1.0 - alpha
                    )
                
                composite_rgb = rgba * alpha + composite_rgb * (1.0 - alpha)
                composite_flow = flow * alpha + composite_flow * (1.0 - alpha)

                # warp rgba t+1 -> t and composite
                rgba_t1 = rgba[b_sz:]
                rgba_warped = warp_flow(rgba_t1, flow[:b_sz].permute(0, 2, 3, 1))
                alpha_warped = rgba_warped[:, 3:4] * 0.5 + 0.5
                composite_warped = rgba_warped[
                    :, :3
                ] * alpha_warped + composite_warped * (1.0 - alpha_warped)

            layers_rgba.append(rgba)
            layers_flow.append(flow)
            alphas_warped.append(rgba_warped[:, 3:4])
            rgbs_warped.append(rgba_warped[:, :3])

        if do_adj:
            # apply learned brightness scaling
            # its called composite_rgb but actually has 4 channel rgba
#             print(br_scale.max(), br_scale.min())
            composite_rgb = br_scale * (composite_rgb * 0.5 + 0.5) 
            composite_rgb = torch.clamp(composite_rgb, 0, 1)
            composite_rgb = composite_rgb * 2 - 1  # map back to [-1, 1] range
            composite_rgb_no_cube = br_scale * (composite_rgb_no_cube * 0.5 + 0.5)
            composite_rgb_no_cube = torch.clamp(composite_rgb_no_cube, 0, 1)
            composite_rgb_no_cube = composite_rgb_no_cube * 2 - 1

        # stack t, t+1 channelwise
        composite_rgb = torch.cat((composite_rgb[:b_sz], composite_rgb[b_sz:]), 1)
        composite_rgb_no_cube = torch.cat(
            (composite_rgb_no_cube[:b_sz], composite_rgb_no_cube[b_sz:]), 1
        )
        composite_flow = torch.cat((composite_flow[:b_sz], composite_flow[b_sz:]), 1)
        layers_rgba = torch.stack(layers_rgba, 2) 
        #[bs*2, channel, layer, h, w]
        if do_adj and not self.opt.isTrain:
            for j in range(layers_rgba.size(2)-1, -1, -1):
                rgba = layers_rgba[:,:,j]
                rgba[:, :3] = br_scale * (rgba[:, :3] * 0.5 + 0.5)             
                rgba[:, :3] = torch.clamp(rgba[:, :3], 0, 1)
                layers_rgba[:,:3, j] = rgba[:, :3] * 2 - 1
                # print(j)
                    
        layers_rgba = torch.cat((layers_rgba[:b_sz], layers_rgba[b_sz:]), 1)
        #[bs, channel*2, layer, h, w]
        layers_flow = torch.stack(layers_flow, 2)
        layers_flow = torch.cat((layers_flow[:b_sz], layers_flow[b_sz:]), 1)
        br_scale = torch.cat((br_scale[:b_sz], br_scale[b_sz:]), 1)
        bg_offset = torch.cat((bg_offset[:b_sz], bg_offset[b_sz:]), 1)

        outputs = {
            "reconstruction_rgb_no_cube": composite_rgb_no_cube,
            "reconstruction_rgb": composite_rgb,
            "reconstruction_flow": composite_flow,
            "layers_rgba": layers_rgba,
            "layers_flow": layers_flow,
            "alpha_warped": torch.stack(alphas_warped, 2),
            "rgb_warped": torch.stack(rgbs_warped, 2),
            "reconstruction_warped": composite_warped,
            "bg_offset": bg_offset,
            "brightness_scale": br_scale,
        }
        return outputs


def warp_flow(tensor, flow):
    """Warp the tensor using the flow. Flow should be in pixel space."""
    h, w = tensor.shape[-2:]
    ramp_u = torch.arange(w).unsqueeze(0).repeat(h, 1)
    ramp_v = torch.arange(h).unsqueeze(-1).repeat(1, w)
    grid = torch.stack([ramp_u, ramp_v], -1).unsqueeze(0)
    grid = grid.float().to(flow.device) + flow
    grid[..., 0] /= w - 1
    grid[..., 1] /= h - 1
    grid = grid * 2 - 1
    warped = F.grid_sample(tensor, grid, align_corners=True)
    return warped
