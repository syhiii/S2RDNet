import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import torch
import scipy.linalg
import thops as thops
from scipy.ndimage import gaussian_filter
from utils import SSFFcn, ConvFcn, positionencoding1D, positionencoding2D, BasicStage
import torch.fft
# 本文中，我们建议采用多层感知（MLP）网络来表示SSF和卷积核，而不是在损失函数中添加手工正则化器。
# 隐式神经表示的主要优点是隐式正则化仅与退化模型相关，不受损失函数中不同图像重建误差的影响。
# 此外，神经表示的优化自由度高于手工表示。
# 定义尺度嵌入层


def GettPSF(llrms, psf1, lrms, hrms):
    # 对 HRMS 和 LRMS 进行傅里叶变换
    HRMS_freq = torch.fft.fft2(llrms)
    LRMS_freq = torch.fft.fft2(psf1)
    HRMS_freq = torch.fft.fft2(llrms)
    LRMS_freq = torch.fft.fft2(psf1)


device = 'cuda:2'
class Net(nn.Module):
    def __init__(self, num_channels=4,channels=None,base_filter=16, args=None):
        super(Net, self).__init__()
        channels = base_filter
        #-----------------
        petype = 'sin_cos'
        self.input_1D = torch.from_numpy(positionencoding1D(num_channels, 32)).float().to(device)
        self.ssffcn = SSFFcn(32, 1)
        self.input_1D2 = torch.from_numpy(positionencoding1D(num_channels, 32)).float().to(device)
        self.ssffcn2 = SSFFcn(32, 7)

        self.input_1D3 = torch.from_numpy(positionencoding1D(num_channels, 32)).float().to(device)
        self.ssffcn3 = SSFFcn(32, 1)

        self.input_2D = torch.from_numpy(positionencoding2D(4, 4, 8, petype)).float().to(device).permute(2, 0, 1).unsqueeze(0)
        self.convfcn = ConvFcn(8, 1)

        self.input_2D2 = torch.from_numpy(positionencoding2D(4, 4, 4, petype)).float().to(device).permute(2, 0, 1).unsqueeze(0)
        self.convfcn2 = ConvFcn(4, 1)

        onelayer = []
        for i in range(3):
            onelayer.append(BasicStage(i))
        self.fcs = nn.ModuleList(onelayer)
        # self.Convweight = nn.Parameter(torch.ones(4, 4))


    def normalize(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def denormalize(self, img, original):
        return img * (original.max() - original.min()) + original.min()

    def forward(self, ms, pan, pos, phi,phi2,Sc, kernel, imsz=64, test=False):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        # pos  [N,C,H,W]
        ksz = 4
        imsz = imsz
        # Phi :SSF 光谱响应函数    PAN = HRMS * SSF
        # Phi2
        # Conv： 点扩散*Down      LRMS = HRMS * Conv    as C
        if phi != None:
            if test == False:
                Phi = phi
                Phi2 = phi2
                Sc = Sc
            else:
                Phi = phi
                Phi2 = phi2
                Sc = Sc
        else:
            Phi = self.ssffcn(self.input_1D)  # self.Phi*self.Phi
            Phi = Phi ** 2  # [C,1]
            # 这里
            Phi2 = self.ssffcn2(self.input_1D2)
            Phi2 = Phi2 ** 2

            Sc = (self.ssffcn3(self.input_1D3)) ** 2

        if kernel != None:
            if test == False:
                Conv = kernel # * self.Convweight
            else:
                Conv = kernel
        else:
            Conv1 = self.convfcn(self.input_2D) ** 2
            # Conv1 = torch.nn.Softmax(dim=0)(Conv1.reshape(ksz ** 2)).reshape(ksz, ksz)  # [ksz,  ksz]
            Conv2 = self.convfcn2(self.input_2D2) ** 2
            Conv= torch.nn.Softmax(dim=0)((Conv2*Conv1).reshape(ksz ** 2)).reshape(ksz, ksz)  # [ksz,  ksz]
            # Conv = Conv1*Conv2
        # Phi  A
        # Phi2 B
        # Conv C
        # Phi3 = S
        #
        Scy = torch.matmul(Sc, pan.view(1, imsz * imsz)).view(-1, 8, imsz, imsz)
        ScTSc = torch.matmul(Sc, Sc.permute(1, 0))
        #
        pos = pos[:,[0,63,128,193,-2,-1],:,:]
        y = pan
        z = ms
        Phiy = torch.matmul(Phi, y.view(1, imsz * imsz)).view(-1, 8, imsz, imsz)
        PhiTPhi = torch.matmul(Phi, Phi.permute(1, 0))

        Phi2TPhi2 = torch.matmul(Phi2, Phi2.permute(1, 0))
        Phiy2 = torch.matmul(Phi2, Phiy[:,0:7,:,:].view(7, imsz * imsz)).view(-1, 8, imsz, imsz)

        ConvTConv = torch.matmul(Conv.reshape(ksz ** 2, 1), Conv.reshape(1, ksz ** 2))
        convTz = z.reshape(8, imsz // ksz, imsz // ksz, 1, 1).repeat(1, 1, 1, ksz, ksz) \
                 * Conv.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(8, imsz // ksz, imsz // ksz, 1, 1)

        convTz = convTz.permute(0, 1, 3, 2, 4).reshape(-1, 8, imsz, imsz)

        y_ = F.interpolate(z, size=(y.shape[2], y.shape[3]), mode='bilinear', align_corners=False)[:,0:7,:,:]
        v = y_
        x =  torch.matmul(Phi, y.view(1, imsz * imsz)).view(-1, 8, imsz, imsz) + torch.matmul(Phi2, v.view(7, imsz * imsz)).view(-1, 8, imsz, imsz)
        # pos, ms, pan, Y, A, B, Xi, X, v
        for i in range(3):
            # x, v,  PANF, LMSF  = self.fcs[i](x, v, Phiy, PhiTPhi, convTz, ConvTConv, pos, ms, pan, imsz)
            y_, v, PANF, LMSF, x = self.fcs[i](pos, ms, pan, y, Phi, Phi2,Scy,ScTSc, Conv, x, y_, v, imsz)
        # Phi :SSF 光谱响应函数    PAN = HRMS * SSF
        # Conv： 点扩散*Down      LRMS = HRMS * Conv
        AY  = torch.matmul(Phi, y.view(1, imsz * imsz)).view(-1, 8, imsz, imsz)
        BY_ = torch.matmul(Phi2, v.view(7, imsz * imsz)).view(-1, 8, imsz, imsz)
        spec_cnn = 1.0 * AY + 1.0 * BY_
        temp_spec_cnn = spec_cnn.view((8, imsz // ksz, ksz, imsz // ksz, ksz)).permute(0, 1, 3, 2, 4)
        pre_spec_cnn = torch.permute(
            torch.matmul(temp_spec_cnn.reshape(8, imsz // ksz, imsz // ksz, ksz**2), Conv.reshape(ksz**2, 1)),
            [3,0,1,2]
        )
        pre_ryb_cnn = torch.matmul(Sc.transpose(1, 0), spec_cnn.view(-1, 8, imsz**2)).view(-1, 1, imsz, imsz)
        '''
        spec_cnn = v
        pre_ryb_cnn = torch.matmul(Phi.transpose(1, 0), spec_cnn.view(-1, 8, imsz**2)).view(-1, 1, imsz, imsz)

        temp_spec_cnn = spec_cnn.view((8, imsz // ksz, ksz, imsz // ksz, ksz)).permute(0, 1, 3, 2, 4)
        pre_spec_cnn = torch.permute(
            torch.matmul(temp_spec_cnn.reshape(8, imsz // ksz, imsz // ksz, ksz**2), Conv.reshape(ksz**2, 1)),
            [3,0,1,2]
        )
        '''
        return spec_cnn, pre_spec_cnn, pre_ryb_cnn, Phi ,Phi2, Sc, Conv, PANF, LMSF, v, AY, BY_
