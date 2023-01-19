from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument(
            "--display_ind",
            type=int,
            default=25,
            help="The index frame to visualize during training.",
        )
        parser.add_argument(
            "--display_freq",
            type=int,
            default=10,
            help="frequency of showing training results on screen (in epochs)",
        )
        parser.add_argument(
            "--display_ncols",
            type=int,
            default=0,
            help="if positive, display all images in a single visdom web panel with certain number of images per row.",
        )
        parser.add_argument(
            "--display_id", type=int, default=1, help="window id of the web display"
        )
        parser.add_argument(
            "--display_server",
            type=str,
            default="http://localhost",
            help="visdom server of the web display",
        )
        parser.add_argument(
            "--display_env",
            type=str,
            default="main",
            help='visdom display environment name (default is "main")',
        )
        parser.add_argument(
            "--display_port",
            type=int,
            default=8097,
            help="visdom port of the web display",
        )
        parser.add_argument(
            "--update_html_freq",
            type=int,
            default=10,
            help="frequency of saving training results to html",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=10,
            help="frequency of showing training results on console (in steps per epoch)",
        )
        parser.add_argument(
            "--no_html",
            action="store_true",
            help="do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/",
        )
        # network saving and loading parameters
        parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=20,
            help="frequency of saving the latest results (in epochs)",
        )
        parser.add_argument(
            "--save_by_epoch",
            action="store_true",
            help='whether saves model as "epoch" or "latest" (overwrites previous)',
        )
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training: load the latest model",
        )
        parser.add_argument(
            "--overwrite_lambdas",
            action="store_true",
            help="continue training and overwrite lambdas and epochs hyperparams by history",
        )
        parser.add_argument(
            "--overwrite_lrs",
            action="store_true",
            help="continue training and overwrite lr by history",
        )
        parser.add_argument(
            "--epoch_count",
            type=int,
            default=1,
            help="the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...",
        )
        parser.add_argument(
            "--phase", type=str, default="train", help="train, val, test, etc"
        )
        # training parameters
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=None,
            help="number of training epochs with the initial learning rate.\
        You only need to specify one of this or n_steps",
        )
        parser.add_argument(
            "--n_steps",
            type=int,
            default=24000,
            help="number of training steps with the initial learning rate",
        )
        parser.add_argument(
            "--n_steps_decay",
            type=int,
            default=0,
            help="number of steps to linearly decay learning rate to zero",
        )
        parser.add_argument(
            "--lr", type=float, default=0.001, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--lr_policy",
            type=str,
            default="linear",
            help="learning rate policy. [linear | step | plateau | cosine]",
        )
        parser.add_argument(
            "--pretrained",
            action="store_true",
            help="Whether use part of a pretrained resnet18 for the discriminator.",
        )
        parser.add_argument(
            "--discriminator_transform",
            type=str,
            default="none",
            help="What transform to apply to omnimatte\
        reconstruction before feeding into the discriminator.",
        )
        parser.add_argument(
            "--jitter",
            action="store_true",
            help="Whether use the original jitter for training.",
        )

        self.isTrain = True
        return parser
