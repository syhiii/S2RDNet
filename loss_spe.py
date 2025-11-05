import torch
import torch.nn as nn
# import lpips
# import piq
# from skimage.metrics import structural_similarity as ssim
from skimage.filters import gaussian
import numpy as np
import torch.nn.functional as F
import pywt
# from sklearn.metrics import mutual_info_score

class MultiMetricLoss(nn.Module):
    def __init__(self, lpips_net='alex'):
        super(MultiMetricLoss, self).__init__()

    
    def forward(self, ms, MS):
        loss_v = 0.0
        loss_s = 0.0

        for i in range(ms.shape[1]):
            ms_channel = ms[:, i, :, :]
            MS_channel = MS[:, i, :, :]
            ms_mean = ms_channel.mean()
            ms_var = ms_channel.var()
            MS_mean = MS_channel.mean()
            MS_var = MS_channel.var()

            mean_diff = (ms_mean - MS_mean).abs()
            var_diff = (ms_var - MS_var).abs()

            loss_v = loss_v + mean_diff
            loss_s = loss_s + var_diff

        combined_loss = 0.5 * loss_v + 0.5 * loss_v
        
        return combined_loss


# 小波变换函数，执行两次 2D 小波变换
def two_level_dwt(tensor, wavelet='haar'):
    B, C, H, W = tensor.shape
    final_LL = []
    for b in range(B):
        LL_channels = []
        for c in range(C):
            channel_data = tensor[b, c].detach().cpu().numpy() # 转换为 numpy 格式
            LL1, (LH1, HL1, HH1) = pywt.dwt2(channel_data, wavelet)
            LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, wavelet)
            LL_channels.append(LL2)
        final_LL.append(np.stack(LL_channels, axis=0))  # C x H//4 x W//4
    final_LL = np.stack(final_LL, axis=0)  # B x C x H//4 x W//4
    final_LL_tensor = torch.tensor(final_LL, device=tensor.device)

    return final_LL_tensor

def Spectral_Wavelet_Loss(img1, img2):
    t = 1e-2
    return torch.abs(two_level_dwt(img1)-img2).sum(dim=1).mean() - t

