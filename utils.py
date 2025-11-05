import numpy as np
import torch
from scipy import special
from scipy import signal
import math
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

# --------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import torch
import scipy.linalg
import thops as thops

withposi = False
separate = False
xydim = 6    # pos dim

class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class Refine(nn.Module):

    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            # CALayer(n_feat,4),
            # CALayer(n_feat,4),
            CALayer(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float() \
                    .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size + in_size, out_size, 3, 1, 1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x + resi


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, msf, panf):
        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf) + 1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf) + 1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)
        amp_fuse = self.amp_fuse(torch.cat([msF_amp, panF_amp], 1))         # 拼接振幅
        pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], 1))         # 拼接相位

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)


def downsample(x, h, w):
    pass


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


'''
class FilterNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=8, base_filter=8, num_embedding=4):
        super(FilterNet, self,).__init__()
        self.conv1 = nn.Conv2d(1, base_filter, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_filter, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        # self.token_fc = nn.Linear(4096, base_filter)  # Token to modify parameters
        # Token 嵌入层，将整数 token 转换为 embedding，范围从 0 到 3
        self.token_embedding = nn.Embedding(num_embedding, base_filter)  # 4 个整数 token (0, 1, 2, 3)
        self.fc_token = nn.Linear(base_filter, 1)  # 将嵌入的 token 变换为和图像通道匹配的维度

        # self.weights = nn.Parameter(torch.arange())
        # Parameters A and B initialized
        self.A = nn.Parameter(torch.full((1, 1, 64, 64), 0.5))  # Shape: B1hw
        self.B = nn.Parameter(torch.full((1, 1, 64, 64), 0.5))

    def forward(self, pan, ms):
        A_out = self.A * pan  # Shape: B1hw
        B_out = self.B * ms  # Shape: BChw
        out = A_out + B_out
        noise = torch.randn_like(out) * 0.01  # Example noise
        output = out + noise
'''
'''
        x = self.relu(self.conv1(pan))
        x = self.relu(self.conv2(x))
        x = (x + pan)
        w = ms * x
        w_min = w.min()
        w_max = w.max()
        w_normalized = (w - w_min) / (w_max - w_min)
        x = x * w_normalized

        return output
'''


def compute_entropy(tensor):
    # 假设输入 tensor 的形状是 [B, C, H, W]
    B, C, H, W = tensor.shape
    entropies = []
    # 对每个 batch 和每个 channel 进行单独计算
    for b in range(B):
        batch_entropy = []
        for c in range(C):
            # 取出单个通道 [H, W]，并展平
            flattened = tensor[b, c].view(-1)  # 形状 [H * W]
            # 计算每个通道的概率分布
            min_val, max_val = flattened.min(), flattened.max()
            num_bins = 256  # 你可以根据需要调整 bins 的数量
            bins = torch.histc(flattened, bins=num_bins, min=min_val.item(), max=max_val.item())
            # 归一化到概率分布
            probs = bins / bins.sum()
            # 计算熵
            entropy = -probs * torch.log2(probs + 1e-12)  # 避免 log(0) 的问题
            entropy = entropy.sum()  # 计算熵的总和
            batch_entropy.append(entropy)
        # 将每个通道的熵放入 batch_entropy 列表中
        entropies.append(torch.tensor(batch_entropy))
    # 拼接结果为 [B, C]，每个 batch 的所有通道的熵值
    entropies = torch.stack(entropies)  # 形状 [B, C]
    return entropies


class Net(nn.Module):
    def __init__(self, num_channels=4, channels=None, base_filter=16, args=None):
        super(Net, self).__init__()
        channels = base_filter
        self.fuse1 = nn.Sequential(InvBlock(HinResBlock, 2 * channels, channels),
                                   nn.Conv2d(2 * channels, channels, 1, 1, 0))
        self.fuse2 = nn.Sequential(InvBlock(HinResBlock, 2 * channels, channels),
                                   nn.Conv2d(2 * channels, channels, 1, 1, 0))
        self.fuse3 = nn.Sequential(InvBlock(HinResBlock, 2 * channels, channels),
                                   nn.Conv2d(2 * channels, channels, 1, 1, 0))
        self.fuse4 = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(channels, channels, 1, 1, 0))
        self.fuse5 = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(channels, channels, 1, 1, 0))
        self.fuse6 = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(channels, channels, 1, 1, 0))
        self.msconv = nn.Conv2d(num_channels, channels, 3, 1, 1)  # conv for ms
        self.panconv = nn.Conv2d(1, channels, 3, 1, 1)
        self.conv0 = HinResBlock(channels, channels)
        self.conv1 = HinResBlock(channels, channels)
        self.conv2 = HinResBlock(channels, channels)

        self.conv_ada = nn.Conv2d(2 * channels, channels, 3, 1, 1)
        self.fft1 = Freprocess(channels)
        self.fft2 = Freprocess(channels)
        self.fft3 = Freprocess(channels)
        self.conv_out = Refine(channels, num_channels-1)
        self.down_pan1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_spa1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_spa2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_pan2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_pan3 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_ms1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_ms2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_ms3 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.f1 = nn.Conv2d(3 * channels, channels, 3, 1, 1)
        self.f2 = nn.Conv2d(3 * channels, channels, 3, 1, 1)
        self.f3 = nn.Conv2d(2 * channels, channels, 3, 1, 1)

        # nex
        self.nexconv = nn.Conv2d(num_channels -1 + xydim if withposi else num_channels - 1, channels, 3, 1, 1)
        self.down_nex1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_nex2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_nex3 = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self,nex, ms, pan):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan is None:
            raise Exception('User does not provide pan image!')
        outputs = []
        _, C, m, n = ms.shape
        b, _, M, N = pan.shape
        mHR = F.interpolate(ms, size=(M, N), mode='bilinear',
                            align_corners=False)  # Upsample to match PAN size
        PAN_FILLTTER = []
        seq = []
        msf = self.msconv(mHR)  # (1->channels)
        panf = self.panconv(pan)  # (1->channels)
        nexf = self.nexconv(nex)
        # PANF.append(panf)
        # panf = panfiltter
        seq.append(panf)
        panf_2 = self.conv1(panf)  # (panf channels->channels)
        seq.append(panf_2)
        panf = self.f3(torch.cat([panf, nexf], 1))
        spa_fuse = self.fuse1(torch.cat([msf, panf], 1))  # concat msf & panf to inn block (2*channels -> channels)

        '''downsample spa_fuse and panf2'''
        M = M // 2
        N = N // 2
        d_spa = self.down_spa1(spa_fuse)  # downsample(spa_fuse, M, N)
        d_panf = self.down_pan1(panf_2)  # downsample(panf_2, M, N)
        d_nexf = self.down_nex1(nexf)
        d_ms = self.down_ms1(msf)
        d_panf = self.f1(torch.cat([d_panf, d_nexf, d_ms], 1))
        spa_fuse = self.fuse2(
            torch.cat([d_spa, d_panf], 1))  # downsampled features into inn block (2*channels-> channels)
        '''downsample spa_fuse and d_panf'''
        M = M // 2
        N = N // 2
        d_spa = self.down_spa2(spa_fuse)
        panf_3 = self.conv0(d_panf)
        seq.append(panf_3)
        d_panf = self.down_pan2(panf_3)
        d_nexf = self.down_nex2(d_nexf)
        d_ms = self.down_ms2(d_ms)
        d_panf = self.f2(torch.cat([d_panf, d_nexf, d_ms], 1))
        spa_fuse = self.fuse3(torch.cat([d_spa, d_panf], 1))

        '''upsample and do fft'''
        M *= 2
        N *= 2
        spa_fuse = F.interpolate(spa_fuse, size=(M, N), mode='bilinear', align_corners=False)
        fft_out = self.fft1(spa_fuse, seq[-1])
        t = self.fuse4(torch.cat([fft_out, spa_fuse], 1))
        t += spa_fuse

        spa_fuse = t
        M *= 2
        N *= 2
        spa_fuse = F.interpolate(spa_fuse, size=(M, N), mode='bilinear', align_corners=False)
        fft_out = self.fft2(spa_fuse, seq[-2])
        t = self.fuse5(torch.cat([fft_out, spa_fuse], 1))
        t += spa_fuse

        t = self.conv_out(t)  # channels to 1
        HR_single = t # + mHR[:,0:7,:,:]  # ms_single
        # HR_single = self.denormalize(HR_single, ms_single_original)  # Denormalize the resul
        outputs.append(HR_single)
        # PAN_FILLTTER.append(panfiltter)
        # HR = torch.cat(outputs, dim=1)  # Concatenate along the channel dimension
        HR = HR_single + nex
        PANF = []
        LMSF = []
        return HR, PANF, LMSF


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d=1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d=1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


# from vgg import VGG161,VGG162,VGG163
'''
class Net222(nn.Module):
    def __init__(self, num_channels=8,channels=None,base_filter=16, args=None):
        super(Net222, self).__init__()
        self.net1 = Net(num_channels=1)
        # self.net1 = VGG161(1, base_filter, num_channels)
        self.net2 = VGG162(1, base_filter, num_channels)
        self.net3 = VGG163(num_channels, base_filter, num_channels)


    def forward(self, ms, pan):
        M_, PAN_FILLTER, PANF, LMSF = self.net1(ms,pan)
        m_ = self.net2(pan,M_)
        p_ = self.net3(ms, M_)
        M__, _, _ , _ = self.net1(m_,p_)

        return M_, m_, p_, M__, PAN_FILLTER, PANF, LMSF
'''
# ------------
def positionencoding1D(W, L):

    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1

    x_el = []

    x_el_hf = []

    pe_1d = np.zeros((W, 2*L+1))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L):
        val = 2 ** el

        x = np.sin(val * np.pi * x_linspace)
        x_el.append(x)

        x = np.cos(val * np.pi * x_linspace)
        x_el_hf.append(x)


    for x_i in range(0, W):

        p_enc = []

        for li in range(0, L):
            p_enc.append(x_el[li][x_i])
            p_enc.append(x_el_hf[li][x_i])

        p_enc.append(x_linspace[x_i])

        pe_1d[x_i] = np.array(p_enc)

    return pe_1d.astype('float32')



def positionencoding2D(W, H, L, basis_function):

    x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1
    y_linspace = (np.linspace(0, H - 1, H) / H) * 2 - 1

    x_el = []
    y_el = []

    x_el_hf = []
    y_el_hf = []

    pe_2d = np.zeros((W, H, 4*L+2))
    # cache the values so you don't have to do function calls at every pixel
    for el in range(0, L):
        val = 2 ** el

        if basis_function == 'rbf':

            # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
            # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

            # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

            M_1 = np.random.rand(2, 2)

            phase_shift = np.random.rand(1) * np.pi

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))))
            x_el.append(x_1_y_1[0, :])
            y_el.append(x_1_y_1[1, :])

            x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
            x_el_hf.append(x_1_y_1[0, :])
            y_el_hf.append(x_1_y_1[1, :])

        elif basis_function == 'diric':

            x = special.diric(np.pi * x_linspace, val)
            x_el.append(x)

            x = special.diric(np.pi * x_linspace + np.pi / 2.0, val)
            x_el_hf.append(x)

            y = special.diric(np.pi * y_linspace, val)
            y_el.append(y)

            y = special.diric(np.pi * y_linspace + np.pi / 2.0, val)
            y_el_hf.append(y)

        elif basis_function == 'sawtooth':
            x = signal.sawtooth(val * np.pi * x_linspace)
            x_el.append(x)

            x = signal.sawtooth(val * np.pi * x_linspace + np.pi / 2.0)
            x_el_hf.append(x)

            y = signal.sawtooth(val * np.pi * y_linspace)
            y_el.append(y)

            y = signal.sawtooth(val * np.pi * y_linspace + np.pi / 2.0)
            y_el_hf.append(y)

        elif basis_function == 'sin_cos':

            x = np.sin(val * np.pi * x_linspace)
            x_el.append(x)

            x = np.cos(val * np.pi * x_linspace)
            x_el_hf.append(x)

            y = np.sin(val * np.pi * y_linspace)
            y_el.append(y)

            y = np.cos(val * np.pi * y_linspace)
            y_el_hf.append(y)

    for y_i in range(0, H):
        for x_i in range(0, W):

            p_enc = []

            for li in range(0, L):
                p_enc.append(x_el[li][x_i])
                p_enc.append(x_el_hf[li][x_i])

                p_enc.append(y_el[li][y_i])
                p_enc.append(y_el_hf[li][y_i])

            p_enc.append(x_linspace[x_i])
            p_enc.append(y_linspace[y_i])


            pe_2d[x_i, y_i] = np.array(p_enc)

    return pe_2d.astype('float32')

# 获取位置编码 映射到高维
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model + 2, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))

    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[1:d_model:2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[d_model:d_model*2:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[d_model + 1:d_model*2:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)

    # add xy
    pe[d_model*2,...] = pos_w * 2/ width - 1.0
    pe[d_model*2+1,...] = pos_h.T * 2 / height - 1.0


    return pe



class SSFFcn(torch.nn.Module):
    def __init__(self, L, out_dim):
        super(SSFFcn, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * L + 1, out_features=4 * L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=4*L, out_features=4*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=4*L, out_features=4*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=4*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=out_dim)#,
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y = self.layers(x)

        return y


class ConvFcn(torch.nn.Module):
    def __init__(self, L, out_dim):

        super(ConvFcn, self).__init__()

        self.layers = nn.Sequential(
            torch.nn.Conv2d(4 * L + 2, 6 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(6 * L, 6 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(6 * L, 3 * L, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(3 * L, 2, (1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(2, out_dim, (1, 1), padding=0)#,
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y = self.layers(x)

        return y

class ResBlock(torch.nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        num_filter = 64
        self.numfilter = num_filter
        self.conv1 = spectral_norm(torch.nn.Conv2d(8 - 1  + xydim if withposi else  8 - 1, num_filter, (3, 3), padding=1))
        self.conv2 = spectral_norm(torch.nn.Conv2d(num_filter, num_filter, (3, 3), padding=1))
        self.conv3 = spectral_norm(torch.nn.Conv2d(num_filter, num_filter, (3, 3), padding=1))
        self.conv4 = spectral_norm(torch.nn.Conv2d(num_filter, num_filter, (3, 3), padding=1))
        self.conv5 = spectral_norm(torch.nn.Conv2d(num_filter, 7, (3, 3), padding=1))


    def output(self, input, output):
        return input + output

    def forward(self, x_input):

        y = x_input
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = self.conv5(y)
        result = self.output(x_input[:,0:7,:,:], y)
        return result



class BasicStage(torch.nn.Module):
    def __init__(self, index):
        super(BasicStage, self).__init__()

        A = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(A, 1e-1)
        self.lamda = nn.Parameter(A)

        B = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(B, 1e-1)
        self.alpha = nn.Parameter(B)

        C = torch.empty(1, 1)#.type(torch.cuda.FloatTensor)
        torch.nn.init.constant_(C, 1e-1)
        self.beta = nn.Parameter(C)

        self.denoisenet1 = Net(num_channels=8)
        self.denoisenet2 = ResBlock()

    def forward(self, pos, ms, pan, Y, A, B, Scy,ScTSc, Conv, x, y_, v, imsz=64):
        ksz = 4
        imsz = imsz
        eta = 1.0
        batch, H, W = 1, imsz, imsz
        #===============================gd===========================

        #  固定v 求解x
        # g = PhiTPhix - Phiy + eta*(convxTx - convTz) + F.relu(self.lamda)*(x-v)  SE
        # g = ConvT * Conv * y * A * BT + ConvT * Conv  y_ * B * BT -ConvT * x * BT  ABSE
        ConvTConv = torch.matmul(Conv.view(ksz ** 2, 1), Conv.view(1, ksz ** 2))  # Shape [ksz^2, ksz^2]

        # ConvT * Conv * Y * A
        temp_Y = torch.einsum('bohw,co->bchw', Y, A)  # Y * A, Shape: [B, 8, H, W]
        # print(temp_Y.shape)#  1 8 16 16
        temptemp_Y = temp_Y.view(-1, 1, imsz//ksz, ksz, imsz//ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1,8, ksz**2)
        ConvTConv_YA = torch.matmul(temptemp_Y, ConvTConv.reshape(ksz**2,ksz**2)).view(-1, 8, imsz//ksz, imsz//ksz, ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1, 8, imsz, imsz)

        # ConvT * Conv * Y_ * B
        temp_Y_tilde =  torch.matmul(B, y_.view(7, imsz * imsz)).view(-1, 8, imsz, imsz)
        temptemp_Y = temp_Y.view(-1, 1, imsz // ksz, ksz, imsz // ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1,8, ksz ** 2)
        ConvTConv_Y_tilde_B = torch.matmul(temptemp_Y, ConvTConv.reshape(ksz ** 2, ksz ** 2)).view(-1, 8, imsz//ksz, imsz//ksz, ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1, 8, imsz, imsz)

        # ConvT * X
        temp_x = x.view(-1, 1, imsz // ksz, ksz, imsz // ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1,8, ksz ** 2)
        ConvT_X = torch.matmul(temp_x, ConvTConv.reshape(ksz ** 2, ksz ** 2)).view(-1, 8, imsz//ksz, imsz//ksz, ksz, ksz).permute(0, 1, 2, 4, 3, 5).reshape(-1, 8, imsz, imsz)
        #
        ScTScx = torch.matmul(ScTSc, x.reshape(-1, 8, imsz * imsz))
        ScTScx = ScTScx.reshape(-1, 8, imsz, imsz)

        g = torch.einsum('ij, bjhw -> bihw', B.T, ConvTConv_YA) + torch.einsum('ij, bjhw -> bihw', B.T, ConvTConv_Y_tilde_B) - torch.einsum('ij, bjhw -> bihw', B.T, ConvT_X) + F.relu(self.lamda)*(y_ - v)
        # 更新y_
        y__next = y_ - self.alpha * g

        x = torch.matmul(A, pan.view(1, imsz * imsz)).view(-1, 8, imsz, imsz) + torch.matmul(B, y__next.view(7, imsz * imsz)).view(-1, 8, imsz, imsz)
        g2 = ScTScx + Scy
        # 更新x
        x_next = x - self.beta * g2
        if withposi:
            # v_next = self.denoisenet2(torch.cat((y__next, pos), dim=1))
            # v_next, PANF, LMSF = self.denoisenet1(torch.cat((v_next, pos), dim=1), ms, pan)
            v_next = self.denoisenet2(y__next)
            v_next, PANF, LMSF = self.denoisenet1(v_next, ms, pan)
        else:
            v_next = self.denoisenet2(y__next)
            v_next, PANF, LMSF = self.denoisenet1(v_next, ms, pan)

        return y__next, v_next,  PANF, LMSF, x_next



