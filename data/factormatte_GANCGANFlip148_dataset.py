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


import cv2
from third_party.data.base_dataset import BaseDataset
from third_party.data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import glob
import torch
import numpy as np
import json
import utils


def load_and_process_image(im_path, mode="RGB", size=None):
    """Read image file and return as tensor in range [-1, 1]."""
    image = Image.open(im_path).convert(mode)
    if size is not None:
        image = image.resize(size)
    image = transforms.ToTensor()(image)
    image = 2 * image - 1
    return image


def load_and_resize_flow(flow_path, width=None, height=None):
    flow = torch.from_numpy(utils.readFlow(flow_path)).permute(2, 0, 1)
    flow = utils.resize_flow(flow, width, height)
    return flow


def apply_transform(data, params, interp_mode="bilinear"):
    """Apply the transform to the data tensor."""
    if data is None:
        return None
    tensor_size = params["jitter size"].tolist()
    crop_pos = params["crop pos"]
    crop_size = params["crop size"]
    orig_shape = data.shape
    if len(orig_shape) < 4:
        data = F.interpolate(
            data.unsqueeze(0), size=tensor_size, mode=interp_mode
        ).squeeze(0)
    else:
        data = F.interpolate(data, size=tensor_size, mode=interp_mode)
    data = data[
        ...,
        crop_pos[0] : crop_pos[0] + crop_size[0],
        crop_pos[1] : crop_pos[1] + crop_size[1],
    ]
    return data


def transform2h(x, y, m):
    """Applies 2d homogeneous transformation."""
    A = torch.matmul(m, torch.stack([x, y, torch.ones(len(x))]))
    xt = A[0, :] / A[2, :]
    yt = A[1, :] / A[2, :]
    return xt, yt


class FactormatteGANCGANFlip148Dataset(BaseDataset):
    """A dataset class for video layers.

    It assumes that the directory specified by 'dataroot' contains metadata.json, and the directories iuv, rgb_256, and rgb_512.
    The 'iuv' directory should contain directories named 01, 02, etc. for each layer, each containing per-frame UV images.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--height", type=int, default=256, help="image height")
        parser.add_argument("--width", type=int, default=448, help="image width")
        parser.add_argument("--in_c", type=int, default=16, help="# input channels")
        parser.add_argument("--jitter_rate", type=float, default=0.75, help="")
        parser.add_argument(
            "--init_flowmask",
            action="store_true",
            help="if true, initialize non-interactive mask with\
        flow != 0 parts.",
        )
        parser.add_argument(
            "--pos_ex_dirs",
            type=str,
            default=",",
            help="positive examples for discriminators, ordered by layer index",
        )
        parser.add_argument(
            "--neg_ex_dirs",
            type=str,
            default=",",
            help="negative examples for discriminators, ordered by layer index",
        )
        parser.add_argument(
            "--start_ind", type=int, default=0, help="start index of dataset"
        )
        parser.add_argument(
            "--fake_corr_ind",
            action="store_true",
            help="whether use the same index for training and fake ex",
        )
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dis_pos_paths = {}
        # self.dis_pos_mask_paths = {}
        self.dis_neg_paths = {}
        # self.dis_neg_mask_paths = {}
        pos_ex_dirs = opt.pos_ex_dirs.split(",")
        neg_ex_dirs = opt.neg_ex_dirs.split(",")
        for f in os.listdir(opt.dataroot):
            if "dis_real_l" in f:
                ind = int(f[-1])
                if pos_ex_dirs[ind - 1] != "":
                    print(
                        "For discriminators pos examples found",
                        f,
                        ind,
                        pos_ex_dirs[ind - 1],
                    )
                    dis_pos_dir = os.path.join(opt.dataroot, f, pos_ex_dirs[ind - 1])
                    # The paths root is ./datasets, continues until the file name
                    self.dis_pos_paths[f] = sorted(make_dataset(dis_pos_dir, np.inf))
                    # if "mask" in os.listdir(dis_pos_dir):
                    #     self.dis_pos_mask_paths[f] = sorted(
                    #         make_dataset(os.path.join(dis_pos_dir, "mask"), np.inf)
                    #     )
                    print(f, len(self.dis_pos_paths[f]))
            if "dis_fake_l" in f:
                ind = int(f[-1])
                if neg_ex_dirs[ind - 1] != "":
                    print(
                        "For discriminators neg examples found",
                        f,
                        ind,
                        neg_ex_dirs[ind - 1],
                    )
                    dis_neg_dir = os.path.join(opt.dataroot, f, neg_ex_dirs[ind - 1])
                    self.dis_neg_paths[f] = sorted(make_dataset(dis_neg_dir, np.inf))
                    # if "mask" in os.listdir(dis_neg_dir):
                    #     self.dis_neg_mask_paths[f] = sorted(
                    #         make_dataset(os.path.join(dis_neg_dir, "mask"), np.inf)
                    #     )
                    print(f, len(self.dis_neg_paths[f]))

        print("pos ex used", self.dis_pos_paths.keys())
        print("neg ex used", self.dis_neg_paths.keys())
        self.dis_gt_indices = {}
        if "dis_gt_alpha_stage" + str(self.opt.stage) in os.listdir(opt.dataroot):
            dis_gt_dir = os.path.join(
                opt.dataroot, "dis_gt_alpha_stage" + str(self.opt.stage)#+'_shouldbe'
            )
            self.dis_gt_paths = sorted(make_dataset(dis_gt_dir, opt.max_dataset_size))
            gt_count = 0
            for f in sorted(os.listdir(dis_gt_dir)):
                if "png" in f:
                    try:
                        ind = int(f.split(".")[0].split("_")[0])
                    except ValueError:
                        ind = int(f.split(".")[0].split("_")[-1])
                    print("available gt alpha", ind)
                    self.dis_gt_indices[ind] = gt_count
                    gt_count += 1

        rgbdir = os.path.join(opt.dataroot, "rgb")
        # if self.opt.phase == "test" and os.path.isdir(
        #     os.path.join(opt.dataroot, "rgb_invis_gt")
        # ):
        #     print("loading rgb_invis_gt for evaluation")
        #     rgbdir = os.path.join(opt.dataroot, "rgb_invis_gt")

        if self.opt.prob_masks:
            print("using mask_nocushionmask")
            maskdir = os.path.join(opt.dataroot, "mask_nocushionmask")
        else:
            print("using original mask")
            maskdir = os.path.join(opt.dataroot, "mask")
        self.image_paths = sorted(make_dataset(rgbdir, opt.max_dataset_size))
        self.image_paths = self.image_paths[:-1]
        n_images = len(self.image_paths)
        layers = sorted(os.listdir(maskdir))
        layers = [l for l in layers if l.isdigit()]
        self.mask_paths = []
        for l in layers:
            layer_mask_paths = sorted(make_dataset(os.path.join(maskdir, l), n_images))
            if len(layer_mask_paths) != n_images:
                print(
                    f"UNEQUAL NUMBER OF IMAGES AND MASKS: {len(layer_mask_paths)} and {n_images}"
                )
            self.mask_paths.append(layer_mask_paths)

        assert len(self.opt.steps) >= 1
        self.step_choices = [int(i) for i in self.opt.steps]
        self.flow_paths, self.confidence_paths = {}, {}
        for i in self.step_choices:
            self.flow_paths[i] = sorted(
                glob.glob(
                    os.path.join(opt.dataroot, "forward_flow_step" + str(i), "*.flo")
                )
            )
            self.confidence_paths[i] = sorted(
                make_dataset(os.path.join(opt.dataroot, "confidence_step" + str(i)))
            )

        # Random noise is always used for foreground initialization.
        zbar_path = os.path.join(opt.dataroot, "zbar.pth")
        if not os.path.exists(zbar_path):
            zbar = torch.randn(1, opt.in_c - 3, opt.height // 16, opt.width // 16)
            torch.save(zbar, zbar_path)
        else:
            zbar = torch.load(zbar_path)

        self.Zbar = zbar
        if not opt.residual_noise:
            print("Initializing residual layer with GT image!")
            self.Zbar = self.Zbar[:, :4]
            if "bg_gt.png" in os.listdir(opt.dataroot):
                self.Zbar_residual = load_and_process_image(
                    os.path.join(opt.dataroot, "bg_gt.png"),
                    mode="RGBA",
                    size=(self.opt.width, self.opt.height),
                ).unsqueeze(0)
            else:
                raise "Required residual to be init with gt but bg_gt.png doesn't exist!"

        Zbar_up = F.interpolate(zbar, (opt.height, opt.width), mode="bilinear")
        if not opt.bg_noise:
            print("Initializing background layer with GT image!")
            if "bg_gt.png" in os.listdir(opt.dataroot):
                self.Zbar_up = load_and_process_image(
                    os.path.join(opt.dataroot, "bg_gt.png"),
                    mode="RGBA",
                    size=(self.opt.width, self.opt.height),
                ).unsqueeze(0)
            else:
                raise "Required bg to be init with gt but bg_gt.png doesn't exist!"
        else:
            self.Zbar_up = Zbar_up

        if self.Zbar_up.size(1) < self.Zbar.size(1):
            diff = self.Zbar.size(1) - self.Zbar_up.size(1)
            self.Zbar_up = torch.cat((Zbar_up[:, :diff], self.Zbar_up), dim=1)

        self.composite_order = [tuple(range(1, 1 + len(layers)))] * n_images
        self.init_homographies(
            os.path.join(opt.dataroot, "homographies.txt"), n_images + 1
        )
        self.bg_gt = None
        if "bg_gt.png" in os.listdir(opt.dataroot):
            self.bg_gt = load_and_process_image(
                os.path.join(opt.dataroot, "bg_gt.png"),
                mode="RGB",
                size=(self.opt.width, self.opt.height),
            )  # .unsqueeze(0)

        self.noninter_ind = []
        if "noninteraction_ind.txt" in os.listdir(opt.dataroot):
            f = open(os.path.join(opt.dataroot, "noninteraction_ind.txt"))
            for l in f:
                startend = l.split(", ")
                start, end = int(startend[0]), int(startend[1])
                self.noninter_ind += list(
                    range(start - self.opt.start_ind, end - self.opt.start_ind)
                )

        self.valid_ind = {i: i for i in range(len(self.image_paths))}
        if opt.noninter_only:
            assert (
                len(self.noninter_ind) != 0
            ), "Please provide noninteraction_ind.txt to use noninter_only training!"
            self.valid_ind = {}
            count = 0
            for ind in self.noninter_ind:
                self.valid_ind[count] = ind
                count += 1
            print("Using non-interaction frames only!", self.valid_ind)

        self.valid_steps = {i: [1] for i in list(self.valid_ind.values())}
        for k, v in self.valid_steps.items():
            for i in self.step_choices:
                # step+index should be within range, so the ones near the end can only take step=1
                if (
                    (i != 1)
                    and ((k + i) < len(self.image_paths))
                    and (bool(k in self.noninter_ind) != bool((k + i) in self.noninter_ind))
                ):
                    v.append(i)
        print("self.valid_steps", self.valid_steps)

    def __getitem__(self, ind):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains the outputs of get_transformed_item for two consecutive frames, concatenated
            channelwise.
        """
        if self.opt.phase == "train":
            transform_params = self.get_params(
                do_jitter=self.opt.jitter, jitter_rate=self.opt.jitter_rate
            )
        else:
            transform_params = self.get_params(
                do_jitter=False, jitter_rate=self.opt.jitter_rate
            )
        index = self.valid_ind[ind]
        step = np.random.choice(self.valid_steps[index])

        # Now the flow depends on the next frame
        data_t1 = self.get_transformed_item(index, transform_params, step)
        data_t2 = self.get_transformed_item(
            index + step, transform_params, self.step_choices[0]
        )
        data = {
            # k: torch.cat((data_t1[k], data_t2[k]), -3)
            # for k in data_t1
            # if k not in ["image_path", "index", "dis_gt_exist"]
        }
        for k in data_t1:
            if k not in ["image_path", "index", "dis_gt_exist"]:
                # print(k)
                data[k] = torch.cat((data_t1[k], data_t2[k]), -3)

        data["image_path"] = data_t1["image_path"]
        data["index"] = torch.cat((data_t1["index"], data_t2["index"]))
        data["dis_gt_exist"] = torch.cat(
            (data_t1["dis_gt_exist"], data_t2["dis_gt_exist"])
        )
        return data

    def get_transformed_item(self, index, transform_params, step):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
            step - - the flow and confidence depend on both frames, so the step matters

        Returns a dictionary that contains:
            input - - input to the Factormatte model
            mask - - object trimaps
            flow - - flow inside object region
            bg_flow - - flow for background layer
            bg_warp - - warping grid used to sample from unwrapped background
            confidence - - flow confidence map
            jitter_grid - - sampling grid that was used for cropping / resizing
            image_path - - path to frame
            index - - frame index
        """
        # Read the target image.
#         print('self.image_paths[index]', index, self.image_paths[index])
        image_path = self.image_paths[index]
        target_image = load_and_process_image(
            image_path, mode="RGB", size=(self.opt.width, self.opt.height)
        )
        nl = len(self.composite_order[index]) + 1

        # Create layer inputs by concatenating mask, flow, and background UVs.
        # Read the layer masks.
        masks = [
            load_and_process_image(
                self.mask_paths[l - 1][index],
                mode="L",
                size=(self.opt.width, self.opt.height),
            )
            for l in self.composite_order[index]
        ]
#         binary_masks = (torch.stack(masks) > 0).float()

        if self.opt.init_flowmask:
            if self.opt.prob_masks:
                if index + self.opt.start_ind in self.dis_gt_indices:
                    masks[self.opt.fg_layer_ind - 1] = load_and_process_image(
                        self.dis_gt_paths[
                            self.dis_gt_indices[index + self.opt.start_ind]
                        ],
                        mode="L",
                        size=(self.opt.width, self.opt.height),
                    )
            else:
                print("Hasn't thought about init_flowmask when prob_masks = False.")
        mask_h, mask_w = masks[0].shape[-2:]
        masks = torch.stack(masks)  # L-1, 1, H, W
        binary_masks = (masks > 0).float()

        # Read flow
        if index >= len(self.flow_paths[step]):
            # for last frame just use zero flow
            flow = torch.zeros(2, self.opt.height, self.opt.width)
        else:
            flow = load_and_resize_flow(self.flow_paths[step][index], mask_w, mask_h)

        # Create bg warp field from homographies.
        bg_warp = self.get_background_uv(index, mask_w, mask_h) * 2 - 1

        # Create the background Z_t from homographies.
        background_Zt = F.grid_sample(
            self.Zbar, bg_warp.permute(1, 2, 0).unsqueeze(0)
        )  # C, H, W
        if self.opt.residual_noise:
            background_Zt = background_Zt.repeat(nl - 1, 1, 1, 1)
        else:
            background_Zt = torch.cat([self.Zbar_residual, background_Zt], 0)

        # Build inputs from masks, flow, background UVs, and unwrapped bg
        input_flow = flow.unsqueeze(0).repeat(nl - 1, 1, 1, 1)
        input_flow *= binary_masks

        if self.opt.prob_masks:
            prob_masks = torch.zeros_like(binary_masks)
            # Using prob_masks assume there are exactly 2 content layers: residual and FG.
            prob_masks[0] = 0.5 * (binary_masks[0] - binary_masks[1]) + 0.5
            prob_masks[1] = 0.5 * (binary_masks[1] - binary_masks[0]) + 0.5
            # prob_masks = binary_masks
            pids = prob_masks  # L-1, 1, H, W
            if self.opt.orderscale:
                pids = (
                    torch.Tensor(self.composite_order[index]).view(-1, 1, 1, 1)
                    * prob_masks
                )
        else:
            pids = binary_masks
            if self.opt.orderscale:
                pids = (
                    torch.Tensor(self.composite_order[index]).view(-1, 1, 1, 1)
                    * binary_masks
                )  # L-1, 1, H, W

        inputs = torch.cat((pids, input_flow, background_Zt), 1)  # L-1, 16, H, W
        background_input = torch.cat(
            (torch.zeros(1, 3, mask_h, mask_w), self.Zbar_up), 1
        )
        inputs = torch.cat((background_input, inputs))  # L, 16, H, W

        # Create bg flow and read confidence
        if index == len(self):
            # for last frame just set to zero (not used)
            bg_flow = torch.zeros(2, mask_h, mask_w)
            confidence = torch.zeros(1, mask_h, mask_w)
        else:
            bg_flow = self.get_background_flow(index, mask_w, mask_h)  # 2, H, W
            confidence = (
                load_and_process_image(
                    self.confidence_paths[step][index],
                    mode="L",
                    size=(self.opt.width, self.opt.height),
                )
                * 0.5
                + 0.5
            )  # [0, 1] range
        #             confidence *= (binary_masks.sum(0) > 0).float()

        masks = masks[:, 0]
#         masks = torch.stack([masks[i] for i in range(nl - 1)])
        masks = torch.stack([self.mask2trimap(masks[i]) for i in range(nl - 1)])
        masks = torch.cat((torch.zeros_like(masks[0:1]), masks))  # add bg mask

        jitter_grid = self.create_grid(mask_w, mask_h)
        jitter_grid = apply_transform(jitter_grid, transform_params, "bilinear")
        masks = apply_transform(masks, transform_params, "bilinear")
        inputs = apply_transform(inputs, transform_params, "bilinear")
        bg_warp = apply_transform(bg_warp, transform_params, "bilinear")
        confidence = apply_transform(confidence, transform_params, "bilinear")
        # when applying transform to flow, also need to rescale
        scale_w = transform_params["jitter size"][1] / mask_w
        scale_h = transform_params["jitter size"][0] / mask_h
        inputs[:, 1] *= scale_w
        inputs[:, 2] *= scale_h
        flow = apply_transform(flow, transform_params, "bilinear")
        flow[0] *= scale_w
        flow[1] *= scale_h
        bg_flow = apply_transform(bg_flow, transform_params, "bilinear")
        bg_flow[0] *= scale_w
        bg_flow[1] *= scale_h

        image_transform_params = transform_params
        target_image = apply_transform(target_image, image_transform_params, "bilinear")

        if index + self.opt.start_ind in self.dis_gt_indices:
            # IF USE ALPHA GT, LET THE INPUT TO load_and_process_image BE SINGLE CHANNEL.
            # RGB WILL REPEAT THAT CHANNEL 3 TIMES, RGBA WILL ADD A ALL-ONES 4TH CHANNEL.
            dis_gt = load_and_process_image(
                self.dis_gt_paths[self.dis_gt_indices[index + self.opt.start_ind]],
                mode=self.opt.rgba,
                size=(self.opt.width, self.opt.height),
            )
            dis_gt_exist = 1
#             if index + self.opt.start_ind in self.noninter_ind:
# #                 print('here', index + self.opt.start_ind)
#                 dis_gt_exist = 1
#             else:
#                 dis_gt_exist = 0
        else:
            # RGBA 4, RGB 3, A 1
            dis_gt = torch.zeros(
                (len(self.opt.rgba), target_image.size(0), target_image.size(1))
            )
            dis_gt_exist = 0
        dis_gt = apply_transform(dis_gt, transform_params, "bilinear")
        bg_gt = apply_transform(self.bg_gt, image_transform_params, "bilinear")
        data = {
            "image": target_image,
            "input": inputs,
            "mask": masks,
            "flow": flow,
            "bg_flow": bg_flow,
            "bg_warp": bg_warp,
            "confidence": confidence,
            "jitter_grid": jitter_grid,
            "image_path": image_path,
            "index": torch.Tensor([index]).long(),
            "dis_gt_exist": torch.Tensor([dis_gt_exist]).long(),
            "dis_gt": dis_gt,
        }
        if bg_gt is not None:
            data["bg_gt"] = bg_gt
        for j in range(2):
            dis_paths = [self.dis_pos_paths, self.dis_neg_paths][j]
            for (layer_name, path) in dis_paths.items():
                random_ind = np.random.randint(len(path))
                if j == 1 and self.opt.fake_corr_ind:
                    # print('using same index', random_ind)
                    random_ind = index
                dis_ex = load_and_process_image(
                    path[random_ind],
                    mode=self.opt.rgba_GAN,
                )
                if layer_name == "dis_real_l1" and (
                    "trampoline_crop" in self.opt.dataroot
                ):
                    dis_pos_transform_params = transform_params.copy()
                    dis_pos_transform_params["jitter size"] = np.array(
                        [160, 256]
                    )  # [166, 198] for greenpuddle [140, 448] for cannonball
                    dis_ex = apply_transform(
                        dis_ex, dis_pos_transform_params, "bilinear"
                    )
                dirname, filename = os.path.split(path[random_ind])
                root_dir, img_dir = os.path.split(dirname)
                if ("mask_" + img_dir in os.listdir(root_dir)) and (
                    filename in os.listdir(os.path.join(root_dir, "mask_" + img_dir))
                ):
                    # print("found mask", root_dir, img_dir, filename, layer_name)
                    # Should have either 1 or 0 as pixel value
                    dis_mask = (
                        load_and_process_image(
                            os.path.join(root_dir, "mask_" + img_dir, filename),
                            mode="L",
                        )
                        * 0.5
                        + 0.5
                    )
                    # No need to apply_transform for these examples
                    # dis_pos_mask = apply_transform(
                    #     dis_pos_mask, transform_params, "bilinear"
                    # )
                else:
                    dis_mask = torch.ones((1, dis_ex.size(1), dis_ex.size(2))).to(
                        dis_ex
                    )

                data[layer_name] = torch.cat((dis_ex, dis_mask))
        return data

    def __len__(self):
        """Return the total number of images."""
        return len(self.valid_ind) - min(self.step_choices)

    def get_params(self, do_jitter=False, jitter_rate=0.75):
        """Get transformation parameters."""
        if do_jitter:
            if np.random.uniform() > jitter_rate:
                scale = 1.0
            else:
                scale = np.random.uniform(1, 1.25)
            jitter_size = (scale * np.array([self.opt.height, self.opt.width])).astype(
                np.int
            )
            start1 = np.random.randint(jitter_size[0] - self.opt.height + 1)
            start2 = np.random.randint(jitter_size[1] - self.opt.width + 1)
        else:
            jitter_size = np.array([self.opt.height, self.opt.width])
            start1 = 0
            start2 = 0
        crop_pos = np.array([start1, start2])
        crop_size = np.array([self.opt.height, self.opt.width])
        return {
            "jitter size": jitter_size,
            "crop pos": crop_pos,
            "crop size": crop_size,
        }

    def init_homographies(self, homography_path, n_images):
        """Read homography file and set up homography data."""
        with open(homography_path) as f:
            h_data = f.readlines()
        h_scale = h_data[0].rstrip().split(" ")
        self.h_scale_x = int(h_scale[1])
        self.h_scale_y = int(h_scale[2])
        h_bounds = h_data[1].rstrip().split(" ")
        self.h_bounds_x = [float(h_bounds[1]), float(h_bounds[2])]
        self.h_bounds_y = [float(h_bounds[3]), float(h_bounds[4])]
        homographies = h_data[2 : 2 + n_images]
        homographies = [
            torch.from_numpy(
                np.array(line.rstrip().split(" ")).astype(np.float32).reshape(3, 3)
            )
            for line in homographies
        ]
        self.homographies = homographies

    def create_grid(self, w, h):
        ramp_u = torch.linspace(-1, 1, steps=w).unsqueeze(0).repeat(h, 1)
        ramp_v = torch.linspace(-1, 1, steps=h).unsqueeze(-1).repeat(1, w)
        grid = torch.stack([ramp_u, ramp_v], 0)
        return grid

    def get_background_flow(self, index, w, h):
        """Return background layer UVs at 'index' (output range [0, 1])."""
        if hasattr(self, "homographies"):
            ramp_u = (
                torch.linspace(0, self.h_scale_x, steps=w).unsqueeze(0).repeat(h, 1)
            )
            ramp_v = (
                torch.linspace(0, self.h_scale_y, steps=h).unsqueeze(-1).repeat(1, w)
            )
            ramp_ = torch.stack([ramp_u, ramp_v], 0)
            ramp = ramp_.reshape(2, -1)
            H_0 = self.homographies[index]
            H_1 = self.homographies[index + 1]
            # apply homography
            [xt, yt] = transform2h(ramp[0], ramp[1], torch.inverse(H_0))
            [xt, yt] = transform2h(xt, yt, H_1)
            # restore shape
            flow = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
            flow -= ramp_
            # # scale from world to [-1, 1]
            # flow[0] /= .5 * self.h_scale_x
            # flow[1] /= .5 * self.h_scale_y
            # scale from world to image space
            flow[0] *= self.opt.width / self.h_scale_x
            flow[1] *= self.opt.height / self.h_scale_y
        else:
            flow = torch.zeros(2, h, w)
        return flow

    def get_background_uv(self, index, w, h):
        """Return background layer UVs at 'index' (output range [0, 1])."""
        ramp_u = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
        ramp_v = torch.linspace(0, 1, steps=h).unsqueeze(-1).repeat(1, w)
        ramp = torch.stack([ramp_u, ramp_v], 0)
        if hasattr(self, "homographies"):
            # scale to [0, orig width/height]
            ramp[0] *= self.h_scale_x
            ramp[1] *= self.h_scale_y
            # apply homography
            ramp = ramp.reshape(2, -1)  # [2, H, W]
            H = self.homographies[index]
            [xt, yt] = transform2h(ramp[0], ramp[1], torch.inverse(H))
            # scale from world to [0,1]
            xt -= self.h_bounds_x[0]
            xt /= self.h_bounds_x[1] - self.h_bounds_x[0]
            yt -= self.h_bounds_y[0]
            yt /= self.h_bounds_y[1] - self.h_bounds_y[0]
            # restore shape
            ramp = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
        return ramp

    def mask2trimap(self, mask):
        """Convert binary mask to trimap with values in [-1, 0, 1]."""
        assert mask.min() == -1
        fg_mask = (mask > 0).float()
        bg_mask = (mask <= 0).float()
        trimap_width = getattr(self.opt, "trimap_width", 20)
        trimap_width *= bg_mask.shape[-1] / self.opt.width
        trimap_width = int(trimap_width)
        bg_mask = cv2.erode(
            bg_mask.numpy(), kernel=np.ones((trimap_width, trimap_width)), iterations=1
        )
        bg_mask = torch.from_numpy(bg_mask)
        mask = fg_mask - bg_mask
        return mask
