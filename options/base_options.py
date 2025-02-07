import argparse
import os
from third_party.util import util
from third_party import models
from third_party import data
import torch
import json


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument(
            "--dataroot",
            required=True,
            help="path to images (should have subfolders rgb_256, etc)",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./1110_checkpoints",
            help="models are saved here",
        )
        parser.add_argument("--seed", type=int, default=35, help="initial random seed")
        # model parameters
        parser.add_argument(
            "--model",
            type=str,
            default="factormatte_GANFlip",
            help="chooses which model to use. [lnr | kp2uv]",
        )
        parser.add_argument(
            "--num_filters",
            type=int,
            default=64,
            help="# filters in the first and last conv layers",
        )
        # dataset parameters
        parser.add_argument(
            "--coarseness",
            type=int,
            default=10,
            help="Coarness of background offset interpolation",
        )
        parser.add_argument(
            "--max_frames",
            type=int,
            default=200,
            help="Similar meaning as max_dataset_size but cannot be infinite for background interpolation.",
        )
        parser.add_argument(
            "--dataset_mode",
            type=str,
            default="factormatte_GANCGANFlip148",
            help="chooses how datasets are loaded.",
        )
        parser.add_argument(
            "--serial_batches",
            action="store_true",
            help="if true, takes images in order to make batches, otherwise takes them randomly",
        )
        parser.add_argument(
            "--num_threads", default=4, type=int, help="# threads for loading data"
        )
        parser.add_argument(
            "--batch_size", type=int, default=8, help="input batch size"
        )
        parser.add_argument(
            "--max_dataset_size",
            type=int,
            default=float("inf"),
            help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
        )
        parser.add_argument(
            "--display_winsize",
            type=int,
            default=256,
            help="display window size for both visdom and HTML",
        )
        # additional parameters
        parser.add_argument(
            "--epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="if specified, print more debugging information",
        )
        parser.add_argument(
            "--suffix",
            default="",
            type=str,
            help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}",
        )
        parser.add_argument(
            "--prob_masks",
            action="store_true",
            help="if true, use 1 over #layer probability mask initialization, otherwise binary",
        )
        parser.add_argument(
            "--rgba",
            default="L",
            type=str,
            help="If true the input FG is RGBA, RGB, or L only.",
        )
        parser.add_argument(
            "--rgba_GAN",
            default="RGB",
            type=str,
            help="If true the input to the GAN discriminator is RGBA, RGB, or L only. Only used when there exists GAN, not CGAN.",
        )
        parser.add_argument(
            "--residual_noise",
            action="store_true",
            help="if true, use random noise for Z initialization.",
        )
        parser.add_argument(
            "--bg_noise",
            action="store_true",
            help="if true, use random noise for background Z initialization.",
        )
        parser.add_argument(
            "--no_bg",
            action="store_true",
            help="If true exclude the bg layer as defined in the original Omnimatte.",
        )
        parser.add_argument(
            "--orderscale",
            action="store_true",
            help="if true, keep the original Omnimatte's version of mask scaling.",
        )
        parser.add_argument(
            "--steps",
            type=str,
            default="1",
            help="X steps apart to consider. Specify without space.",
        )
        parser.add_argument(
            "--noninter_only",
            action="store_true",
            help="if true, only use nonteractive frames of the video.",
        )
        parser.add_argument(
            "--gradient_debug",
            action="store_true",
            help="whether to do the real gradient descent or just to record the gradients.",
        )
        parser.add_argument(
            "--num_Ds",
            default="0,3,3",
            type=str,
            help="Number of multiscale discriminators.",
        )
        parser.add_argument(
            "--strides",
            default="0,2,2",
            type=str,
            help="Number of stride in the convs of multiscale discriminators.",
        )
        parser.add_argument(
            "--n_layers",
            default="0,1,3",
            type=str,
            help="Number of stride in the convs of multiscale discriminators.",
        )
        parser.add_argument(
            "--fg_layer_ind",
            type=int,
            default=2,
            help="Which layer is the foreground, starting from 0.",
        )
        parser.add_argument(
            "--stage",
            type=int,
            help="Tells the dataset which dis_gt_alpha to use; index starting from 1. Stage 1: get bg, shouldn't have any dis_gt_alpha; stage 2: alpha from stage 1, for regularizing color, should run only on NFs; stage 3: alpha from stage 2 to constrain the alpha in IFs.",
        )
        parser.add_argument(
            "--get_bg",
            action="store_true",
            help="if specified, generate the bg panorama and quit",
        )
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "{}_opt.txt".format(opt.phase))
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
