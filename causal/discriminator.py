import torch
import torchvision.models as models
import torch.nn as nn
import functools
import torch.nn.functional as F
import numpy as np


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_4MultiscaleDiscriminator(nn.Module):
    def __init__(self, args, input_nc, ndf, n_layers, s, norm_layer, use_sigmoid):
        super(NLayerDiscriminator_4MultiscaleDiscriminator, self).__init__()
        self.conv = nn.Conv2d
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[self.conv(input_nc, ndf, kernel_size=kw, stride=s, padding=padw), \
                     nn.LeakyReLU(0.2, True)]]
        nf = ndf
        # start from 1 because already 1 layer, minus 1 because another layer in the end
        for n in range(1, n_layers-1):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                self.conv(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf), 
                nn.LeakyReLU(0.2, True)
            ]]

        sequence += [[self.conv(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, args, stride, num_D, n_layers, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, ndf=64):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.args = args
        if args.rgba_GAN == 'RGBA':
            input_nc = 4 
        if args.rgba_GAN == 'RGB':
            input_nc = 3 
        elif args.rgba_GAN == 'A':
            input_nc = 1 #a+mask
        
     
        for i in range(num_D):
            print('Initializing', i, 'th-scale discriminator. n_layers', n_layers, 'ndf', ndf, 'stride', stride,\
                 norm_layer, use_sigmoid)
            netD = NLayerDiscriminator_4MultiscaleDiscriminator(args, input_nc, ndf, n_layers, stride, \
                                                                norm_layer, use_sigmoid)
            setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, x):
        return model(x).flatten(1)
        
    def forward(self, x):        
        num_D = self.num_D
        result = []
        result_valid = []
        input_downsampled = x
        for i in range(num_D):
            model = getattr(self, 'layer'+str(num_D-1-i))
            patches = self.singleD_forward(model, input_downsampled)
            result.append(patches)
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return torch.cat(result, 1)
    
    