"""Script for training an Omnimatte model on a video.

You need to specify the dataset ('--dataroot') and experiment name ('--name').

Example:
    python train.py --dataroot ./datasets/tennis --name tennis --gpu_ids 0,1

The script first creates a model, dataset, and visualizer given the options.
It then does standard network training. During training, it also visualizes/saves the images, prints/saves the loss
plot, and saves the model.
Use '--continue_train' to resume your previous training.

See options/base_options.py and options/train_options.py for more training options.
"""
import time
from options.train_options import TrainOptions
from third_party.data import create_dataset
from third_party.models import create_model
from third_party.util.visualizer import Visualizer
import torch
import numpy as np
import random
import os


def main():
    trainopt = TrainOptions()
    opt = trainopt.parse()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print("The number of training images = %d" % dataset_size)
    if opt.n_epochs is None:
        assert opt.n_steps, "You must specify one of n_epochs or n_steps."
        opt.n_epochs = int(
            opt.n_steps / np.ceil(dataset_size)
        )  # / opt.batch_size divide by bs seems wierd
    opt.n_epochs_decay = int(opt.n_steps_decay / np.ceil(dataset_size / opt.batch_size))
    total_iters = 0
    model = create_model(opt)
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if opt.continue_train:
        opt.epoch_count = int(opt.epoch) + 1
        if opt.overwrite_lambdas:
            # Setting parameters here will overwrite the previous code
            history = torch.load(
                os.path.join(model.save_dir, opt.epoch + "_others.pth"), map_location='cuda:0' 
            )
            for name in model.lambda_names:
                if isinstance(name, str):
                    setattr(model, "lambda_" + name, history["lambda_" + name])
                    print(
                        "lambdas overwritten args",
                        "lambda_" + name,
                        getattr(model, "lambda_" + name, None),
                    )
            total_iters = history["total_iters"]
            model.jitter_rgb = history["jitter_rgb"]
            model.do_cam_adj = history["do_cam_adj"]
            # Assume when continue by loading, there're already plenty of epochs passed
            # such that mask loss is no longer needed (set to 0)
            # model.mask_loss_rolloff_epoch = 0
            model.mask_loss_rolloff_epoch = history["mask_loss_rolloff_epoch"]
            print(
                "other params overwritten args",
                model.jitter_rgb,
                model.do_cam_adj,
                total_iters,
                opt.epoch_count,
                model.mask_loss_rolloff_epoch,
            )

            for i in range(len(model.discriminators)):
                if (model.discriminators[i] is not None) and (
                    "discriminator_l" + str(i) in history
                ):
                    model.discriminators[i].load_state_dict(
                        history["discriminator_l" + str(i)], strict=False
                    )
                    print(i, "th discriminator weights loaded unstrictly")
                    print(
                        "the dict in the history is",
                        history["discriminator_l" + str(i)].keys(),
                    )
                    print(
                        "the dict in current model is",
                        model.discriminators[i].state_dict().keys(),
                    )
                    model.discriminators[i].train()

        if opt.overwrite_lrs:
            print("lr overwritten args", history["lrs"])
            for i in range(len(model.optimizers)):
                optimizer = model.optimizers[i]
                for g in optimizer.param_groups:
                    g["lr"] = history["lrs"][i]

    visualizer = Visualizer(opt)
    train(model, dataset, visualizer, opt, total_iters)


def train(model, dataset, visualizer, opt, total_iters):
    dataset_size = len(dataset)
    for epoch in range(
        opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_lambdas(epoch)
        if epoch == opt.epoch_count:
            save_result = True
            dp = dataset.dataset[opt.display_ind]
            for k, v in dp.items():
                if torch.is_tensor(v):
                    dp[k] = v.unsqueeze(0)
                else:
                    dp[k] = [v]
            model.set_input(dp)
            model.compute_visuals(total_iters)
            visualizer.display_current_results(
                model.get_current_visuals(), 0, save_result
            )

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if i % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # #iters are not exact because the last batch might not suffice.
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(total_iters, epoch)

            if (
                i % opt.print_freq == 0
            ):  # print training losses and save logging information to the disk
                print(opt.name)
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data
                )
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )
            iter_data_time = time.time()

        if (
            epoch % opt.display_freq == 1
        ):  # display images on visdom and save images to a HTML file
            save_result = epoch % opt.update_html_freq == 1
            dp = dataset.dataset[opt.display_ind]
            for k, v in dp.items():
                if torch.is_tensor(v):
                    dp[k] = v.unsqueeze(0)
                else:
                    dp[k] = [v]
            model.set_input(dp)
            model.compute_visuals(total_iters)
            visualizer.display_current_results(
                model.get_current_visuals(), epoch, save_result
            )

        if (
            epoch % opt.save_latest_freq == 0 or epoch == opt.epoch_count
        ):  # opt.n_epochs + opt.n_epochs_decay:   # cache our latest model every <save_latest_freq> epochs
            print(
                "saving the latest model (epoch %d, total_iters %d)"
                % (epoch, total_iters)
            )
            save_suffix = "epoch_%d" % epoch if opt.save_by_epoch else "latest"
            model.save_networks(save_suffix)
            others = {
                "lrs": [i.param_groups[0]["lr"] for i in model.optimizers],
                "jitter_rgb": model.jitter_rgb,
                "do_cam_adj": model.do_cam_adj,
                "total_iters": total_iters,
            }
            for i in range(len(model.discriminators)):
                if model.discriminators[i] is not None:
                    others["discriminator_l" + str(i)] = model.discriminators[
                        i
                    ].state_dict()
            for name in model.lambda_names:
                if isinstance(name, str):
                    others["lambda_" + name] = float(getattr(model, "lambda_" + name))
            others["lambda_Ds"] = torch.tensor(model.lambda_Ds)
            others["lambda_plausibles"] = torch.tensor(model.lambda_plausibles)
            others["mask_loss_rolloff_epoch"] = model.mask_loss_rolloff_epoch
            torch.save(
                others,
                os.path.join(opt.checkpoints_dir, opt.name, str(epoch) + "_others.pth"),
            )

        if ((epoch == 1) or (epoch % opt.update_D_epochs == 0)) and (
            model.optimizer_D is not None
        ):
            model.update_learning_rate([1])
        model.update_learning_rate(
            [0]
        )  # update learning rates at the end of every epoch.
        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )


def see_grad(model, dataset, visualizer, opt):
    total_iters = 0  # the total number of training iterations
    for f in os.listdir(opt.ckpt_dir):
        if "net_Omnimatte.pth" in f:
            weight = torch.load(os.path.join(opt.ckpt_dir, f))
            model.netOmnimatte.load_state_dict(weight)
    for epoch in range(
        1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_lambdas(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if i == 0:
                iter_start_time = time.time()  # timer for computation per iteration
                if i % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters(total_iters)
            else:
                break


if __name__ == "__main__":
    main()
