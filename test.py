"""Script to save the full outputs of an Omnimatte model.

Once you have trained the Omnimatte model with train.py, you can use this script to save the model's final omnimattes.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates a model and dataset given the options. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (after training a model):
    python test.py --dataroot ./datasets/tennis --name tennis

    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

See options/base_options.py and options/test_options.py for more test options.
"""
import os
from options.test_options import TestOptions
from third_party.data import create_dataset
from third_party.models import create_model
from third_party.util.visualizer import save_images, save_videos
from third_party.util import html
import torch


if __name__ == "__main__":
    testopt = TestOptions()
    opt = testopt.parse()
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if opt.gradient_debug:
        weight = torch.load(
            os.path.join(opt.checkpoints_dir, opt.name, str(opt.epoch) + "_others.pth")
        )
        for i in range(len(model.discriminators)):
            if model.discriminators[i] is not None:
                model.discriminators[i].load_state_dict(
                    weight["discriminator_l" + str(i)], strict=False
                )
                print(i, "th discriminator weights loaded unstrictly")
                print(
                    "the dict in the history is",
                    weight["discriminator_l" + str(i)].keys(),
                )
                print(
                    "the dict in current model is",
                    model.discriminators[i].state_dict().keys(),
                )

    # create a website
    web_dir = os.path.join(
        opt.results_dir,
        opt.name,
        "{}_{}_{}".format(opt.phase, opt.epoch, opt.test_suffix),
    )  # define the website directory
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )
    video_visuals = None
    loss_recon = 0
    model.do_cam_adj = True #False
    for i, data in enumerate(dataset):
#         print(i)
#         if i < 130:
#             continue
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test(i)  # run inference
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print("processing (%04d)-th image... %s" % (i, img_path))
        with torch.no_grad():
            visuals = model.get_results()  # rgba, reconstruction, original, mask
            if video_visuals is None:
                video_visuals = visuals
            else:
                for k in video_visuals:
                    video_visuals[k] = torch.cat((video_visuals[k], visuals[k]))
            for k in video_visuals:
                rgba = {k: visuals[k]}  # for k in visuals if "rgba" in k
                # save RGBA layers
                save_images(
                    webpage,
                    rgba,
                    img_path,
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.display_winsize,
                )
            # if os.path.isdir(os.path.join(opt.dataroot, "rgb_invis_gt")):
            #     print(
            #         model.criterionLoss(
            #             model.reconstruction_rgb_no_cube, model.target_image
            #         ),
            #     )
            #     loss_recon += model.criterionLoss(
            #         model.reconstruction_rgb_no_cube, model.target_image
            #     )

    save_videos(webpage, video_visuals, width=opt.display_winsize)
    webpage.save()  # save the HTML of videos
    with open(os.path.join(web_dir, "invis_gt_eval.txt"), "w") as f:
        print("avg recon no cube L1Loss " + str(loss_recon / len(dataset)), file=f)
