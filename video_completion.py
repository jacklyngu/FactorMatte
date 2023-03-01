import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
import os
import numpy as np
import torch
from PIL import Image
import glob
import torchvision.transforms.functional as F

from RAFT import utils
from RAFT import RAFT


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def calculate_flow(args, model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    # if os.path.isdir(os.path.join(args.outroot, 'flow', mode + '_flo')):
    #     for flow_name in sorted(glob.glob(os.path.join(args.outroot, 'flow', mode + '_flo', '*.flo'))):
    #         print("Loading {0}".format(flow_name), '\r', end='')
    #         flow = utils.frame_utils.readFlow(flow_name)
    #         Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
    #     return Flow
    flow_folder = 'flow' + args.path.replace("/","")
    create_dir(os.path.join(args.outroot, flow_folder, mode + '_flo'))
    create_dir(os.path.join(args.outroot, flow_folder, mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            # Flow visualization.
            flow_img = utils.flow_viz.flow_to_image(flow)
            flow_img = Image.fromarray(flow_img)

            # Saves the flow and flow_img.
            flow_img.save(os.path.join(args.outroot, flow_folder, mode + '_png', '%05d.png'%i))
#             np.save(os.path.join(args.outroot, 'flow', mode + '_flo', '%05d.npy'%i), flow)
            utils.frame_utils.writeFlow(os.path.join(args.outroot, flow_folder, mode + '_flo', '%05d.flo'%i), flow)

    return Flow


def calculate_flow_global(args, model, video, mode, step=1):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    # if os.path.isdir(os.path.join(args.outroot, 'flow', mode + '_flo')):
    #     for flow_name in sorted(glob.glob(os.path.join(args.outroot, 'flow', mode + '_flo', '*.flo'))):
    #         print("Loading {0}".format(flow_name), '\r', end='')
    #         flow = utils.frame_utils.readFlow(flow_name)
    #         Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
    #     return Flow
    flow_folder = args.path.replace("/","")
    create_dir(os.path.join(args.outroot, flow_folder, mode + '_flow_step' + str(step)))
    create_dir(os.path.join(args.outroot, flow_folder, mode + '_png_step' + str(step)))
    global_max = -10000000
    with torch.no_grad():
#         for i in range(10):
        for i in range(video.shape[0] - step):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + step), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + step, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + step, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters = 20, test_mode = True)
            flow_max = torch.sqrt(flow[0,0,:,:] ** 2 + flow[0, 1, :, :] ** 2).max()
            global_max = max(global_max, flow_max.cpu().numpy())
            print(global_max)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            Flow = np.concatenate((Flow, flow[..., None]), axis = -1)
        
        for j in range(Flow.shape[-1]):
            flow=Flow[:,:,:,j]
            print(j)
            # Flow visualization.
            flow_img = utils.flow_viz.flow_to_image(flow, rad_max=global_max)
            flow_img = Image.fromarray(flow_img)

            # Saves the flow and flow_img.
            flow_img.save(os.path.join(args.outroot, flow_folder, mode + '_png_step' + str(step), '%05d.png'%j))
            utils.frame_utils.writeFlow(os.path.join(args.outroot, flow_folder, mode + '_flow_step' + str(step), '%05d.flo'%j), flow)

    return Flow

def video_completion(args):

    # Flow model.
    RAFT_model = initialize_RAFT(args)

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png'))
    #        glob.glob(os.path.join(args.path, '*.jpg'))

    # Obtains imgH, imgW and nFrame.
    imgH, imgW = np.array(Image.open(filename_list[0]).convert('RGB')).shape[:2]
    nFrame = len(filename_list)

    # Loads video.
    video = []
    for filename in sorted(filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename).convert('RGB')).astype(np.uint8)).permute(2, 0, 1).float())

    video = torch.stack(video, dim=0)
    video = video.to('cuda')

    # Calcutes the corrupted flow.
    print('STEP', str(args.step))
    corrFlowF = calculate_flow_global(args, RAFT_model, video, 'forward', step=args.step) #_interval
    corrFlowB = calculate_flow_global(args, RAFT_model, video, 'backward', step=args.step) #_interval
    print('\nFinish flow prediction.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # video completion
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='../data/tennis', help="dataset for evaluation")
    parser.add_argument('--outroot', default='RAFT_result/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)
    parser.add_argument('--step', default=1, type=int)

    # RAFT
    parser.add_argument('--model', default='../weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    video_completion(args)
