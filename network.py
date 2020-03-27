import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        # wf: the fourier transformation of correlation kernel w. You will need to calculate the best wf in update method.
        self.wf = 0 #need to set equal to 0 by default to avoid Nonetype errors
        # xf: the fourier transformation of target patch x.
        self.xf = 0
        self.config = config

    def forward(self, z):
        """
        :param z: the multiscale searching patch. Shape (num_scale, 3, height, width)
        :return response: the response of cross correlation. Shape (num_scale, 1, height, width)

        You are required to calculate response using self.wf to do cross correlation on the searching patch z
        """
        # obtain feature of z and add hanning window
        z = self.feature(z) * self.config.cos_window
        # TODO: You are required to calculate response using self.wf to do cross correlation on the searching patch z
        # put your code here
        response = torch.zeros([z.size()[0], 1, z.size()[2], z.size()[3]])
        for scale in range(z.size()[0]):
            g_tmp = 0
            for channel in range(z.size()[1]):
                #conjugate
                w_conj = self.wf.clone()
                w_conj[0, channel, :, :, 1] = w_conj[0, channel, :, :, 1] * (-1)
                
                g_arg = w_conj[0, channel].to(device)*torch.rfft(z[scale, channel], 2).to(device)
                g_tmp += g_arg
            response[scale, 0] = (torch.irfft(g_tmp, 2))
        return response

    def update(self, x, lr=1.0):
        """
        this is the to get the fourier transformation of  optimal correlation kernel w
        :param x: the input target patch (1, 3, h ,w)
        :param lr: the learning rate to update self.xf and self.wf

        The other arguments concealed in self.config that will be used here:
        -- self.config.cos_window: the hanning window applied to the x feature. Shape (crop_sz, crop_sz),
                                   where crop_sz is 125 in default.
        -- self.config.yf: the fourier transform of idea gaussian response. Shape (1, 1, crop_sz, crop_sz//2+1, 2)
        -- self.config.lambda0: the coefficient of the normalize term.

        things you need to calculate:
        -- self.xf: the fourier transformation of x. Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        -- self.wf: the fourier transformation of optimal correlation filter w, calculated by the formula,
                    Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        """
        # x: feature of patch x with hanning window. Shape (1, 32, crop_sz, crop_sz)
        x = self.feature(x) * self.config.cos_window
        # TODO: calculate self.xf and self.wf
        # put your code here
        self.xf = ((1 - lr)*self.xf) + (lr*(torch.rfft(x, 3).data))
        self.xf.detach()
        #conjugate
        y_conj = self.config.yf.clone()
        y_conj[:, :, :, :, 1] = y_conj[:, :, :, :, 1] * (-1)

        denom = 0
        
        W = torch.zeros(self.xf.size()) #empty array to hold w_l values
        #this portion to calculate the denominator
        for channel in range(x.size()[1]):
            x_forier = torch.rfft(x[0, channel], 2)

            #find conjugate by 
            x_conj = x_forier.clone()  
            x_conj[:,:,1] = x_conj[:,:,1] * (-1)

            denom += torch.rfft(x[0, channel], 2) * x_conj + self.config.lambda0
        
        for channel in range(x.size()[1]):
            numerator = torch.rfft(x[0, channel], 2)*self.config.yf
            w_l = numerator/denom

            W[0, channel] = w_l #to make output 5 dimensional

        #learning rate for all w_l
        self.wf = ((1-lr)*self.wf) + (lr*W.data)
        self.wf.detach()

        '''
        taking the CC here works as follows:
            because the last dimension of each matrix we are tasked with finding the CC of has a last dimension in its shape corresponding to the real-valued layer (0th layer) and 
            complex-valued layer (1st layer), we can leave the first layer A[0, channel_num, : (height), : (width), real] the same and multiply the 
            second layer A[0, channel_num, :, :, complex] by (-1) in order to effectively accomplish a complex conjugation  
        '''

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)

