import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from skimage.feature import canny
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
import numpy as np
from metric import _ssim
# SSIM 计算函数

# 高频滤波器
def high_pass_filter(img):
    # 使用 Sobel 过滤器提取高频信息
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(img.device)
    high_pass = F.conv2d(img, kernel, padding=1)
    return high_pass

# 傅里叶变换相似性计算
def fourier_similarity(img1, img2):
    fft_1 = fft2(img1)
    fft_2 = fft2(img2)
    # 计算频域相似性（例如，使用均方误差）
    return F.mse_loss(torch.abs(fft_1), torch.abs(fft_2))

# 损失函数
def custom_loss(pan, ms):
    B, C, H, W = ms.size()
    ssim_loss = 0
    gradient_loss = 0
    edge_loss = 0
    l1_loss = 0
    pan = pan.repeat(1, C, 1, 1)  # 将单通道的 PAN 扩展为多通道，与 MS 一致

    for i in range(C):
        ms_channel = ms[:, i:i + 1, :, :]
        pan_channel = pan[:, i:i + 1, :, :]

        # 将 Tensor 转换为 numpy 数组进行计算
        ms_np = ms_channel.detach().cpu().numpy()
        pan_np = pan_channel.detach().cpu().numpy()
        # 计算 L1 范式
        l1_loss_np = np.abs(ms_np - pan_np).sum(axis=(1, 2, 3))  # 在 C, H, W 维度上求和，保留 batch 维度

        # 如果你想要计算平均的 L1 范式
        l1_mean_np = np.mean(l1_loss_np)  # 计算 batch 中所有样本的平均 L1 范式
        l1_loss = l1_loss + l1_mean_np
        # SSIM Loss
        ssim_channel_loss = 0
        gradient_channel_loss = 0
        edge_channel_loss = 0
        fourier_channel_loss = 0

        for j in range(B):
            # 计算 SSIM
            ssim_value = ssim(ms_np[j, 0], pan_np[j, 0], data_range=ms_np[j, 0].max() - ms_np[j, 0].min())
            ssim_channel_loss += 1.0 - ssim_value  # SSIM 越大越好，loss 越小

            # 计算梯度相似度（MSE）
            gradient_ms = sobel(ms_np[j, 0])
            gradient_pan = sobel(pan_np[j, 0])
            gradient_channel_loss += np.mean((gradient_ms - gradient_pan))

            # 计算边缘相似度（1 - 平均边缘相似性）
            edges_ms = canny(ms_np[j, 0])
            edges_pan = canny(pan_np[j, 0])
            edge_channel_loss += 1 - np.mean(edges_ms == edges_pan)

            # 计算傅里叶相似度（MSE of high-frequency components）
            # fft_ms = fftshift(fft2(ms_np[j, 0]))
            # fft_pan = fftshift(fft2(pan_np[j, 0]))
            # high_freq_ms = fft_ms - gaussian_filter(np.abs(fft_ms), sigma=10)
            # high_freq_pan = fft_pan - gaussian_filter(np.abs(fft_pan), sigma=10)
            # fourier_channel_loss += np.mean((np.abs(high_freq_ms) - np.abs(high_freq_pan)) ** 2)

        # 将通道的平均损失累加到总损失中
        ssim_loss += ssim_channel_loss / B
        # gradient_loss += gradient_channel_loss / B
        edge_loss += edge_channel_loss / B
        # fourier_loss += fourier_channel_loss / B

        '''
        print('ssim_loss',ssim_loss)
        print('gradient_loss', gradient_loss)
        print('edge_loss', edge_loss)
        print('fourier_loss', fourier_loss)
        '''
        # 合成最终的损失值
    total_loss = ssim_loss + 0.0 + edge_loss # + l1_mean_np # + 0.01 * fourier_loss
    return total_loss

def calculate_different_size(ms, MS):
    # (1) 计算 ms 和 MS 的分辨率因子 factor
    B, C, H_ms, W_ms = ms.shape  # ms 的形状
    _, _, H_MS, W_MS = MS.shape  # MS 的形状

    # 分辨率因子 (假设 MS 的分辨率是 ms 的整数倍)
    factor_H = H_MS // H_ms
    factor_W = W_MS // W_ms
    assert factor_H == factor_W, "The factor for height and width should be the same."
    factor = factor_H
    pool = torch.nn.AvgPool2d(kernel_size=factor, stride=factor, padding=0)
    MS = pool(MS)
    # 最终确保 MS 和 ms 形状一致
    assert MS.shape[2:] == ms.shape[2:], "The final pooled MS shape should match ms shape."
    # (4) 计算与 ms 的一阶范式
    difference = MS - ms
    l1_norm = difference.mean()
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

    combined_loss = 0.5 * loss_s + 0.5 * loss_v

    return l1_norm # + combined_loss


def features_grad(features):
    # 定义卷积核
    kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8],
                           [1 / 8, -1, 1 / 8],
                           [1 / 8, 1 / 8, 1 / 8]], dtype=torch.float32, device='cuda:1')
    # 将kernel扩展为符合conv2d的卷积格式
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # kernel shape: [1, 1, 3, 3]
    # 获取输入的通道数
    B, C, H, W = features.shape
    # 对每个通道应用卷积操作
    for i in range(C):
        fg = F.conv2d(features[:, i:i + 1, :, :], kernel, padding=1)  # 单通道卷积
        if i == 0:
            fgs = fg
        else:
            fgs = torch.cat([fgs, fg], dim=1)  # 在通道维度拼接
    return fgs

def SSIM_LOSS(img1, img2, same=True):
    # 扩展 img1 的形状到 [B, C, H, W]
    if same==False:
        img1_expanded = img1.repeat(1, img2.size(1), 1, 1)  # 形状: [B, C, H, W]
        img1_np = img1_expanded.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]
        img2_np = img2.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]
    else:
        img1_np = img1.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]
        img2_np = img2.permute(0, 2, 3, 1).detach().cpu().numpy()  # [B, H, W, C]

    # 计算每个通道的 SSIM
    ssim_values = []
    for b in range(img2_np.shape[0]):  # 遍历每张图像
        ssim_per_image = []
        for c in range(img2_np.shape[3]):  # 遍历每个通道
            ssim_val = _ssim(img1_np[b, :, :, c], img2_np[b, :, :, c])
            ssim_per_image.append(ssim_val)
            # 使用 torch.tensor 创建张量并计算均值
        ssim_values.append(torch.tensor(ssim_per_image).mean())

        # 使用 torch.tensor 创建张量并计算均值
    return torch.tensor(ssim_values).mean()


def Fro_LOSS(batchimg):
    """计算 Frobenius 范数损失"""
    fro_norm = torch.square(torch.norm(batchimg, p='fro', dim=[1, 2])) / (batchimg.size(1) * batchimg.size(2))
    E = torch.mean(fro_norm)
    return E
def compute_fro_loss(img1, img2):
    num_channels = img2.size(1)  # img2 的通道数
    fro_loss_sum = 0

    for i in range(num_channels):
        img2_channel = img2[:, i:i+1, :, :]  # 选择 img2 的第 i 个通道
        loss = Fro_LOSS(img2_channel - img1)  # 计算损失
        fro_loss_sum += loss

    mean_fro_loss = fro_loss_sum / num_channels  # 返回均值
    return mean_fro_loss


def calculate_mean_squared_grad(S1_FEAS):
    grad_features = features_grad(S1_FEAS)  # 计算梯度特征
    squared_grad = torch.square(grad_features)  # 计算平方
    mean_squared_grad = torch.mean(squared_grad, dim=(1, 2, 3))  # 对[H, W, C]维度取平均值
    return mean_squared_grad

def retainLoss(FeatureList, FeatureList2, img1, img2, img3): # pan ms hx
    # _, _  pan,ms, hrms
    img2 = F.interpolate(img2, size=(img3.shape[2], img3.shape[3]), mode='bilinear', align_corners=False)  # Upsample to match PAN size
    c = 1e-2    # 3.5e3 1e2
    SSIM1 = 1 - SSIM_LOSS(img1, img3, same=False)
    mse1 = compute_fro_loss(img1, img3)
    SSIM2 = 1 - SSIM_LOSS(img2, img3,same=True)
    mse2 = compute_fro_loss(img2, img3)
    for i in range(len(FeatureList)):
        m1 = calculate_mean_squared_grad(FeatureList[i])
        m2 = calculate_mean_squared_grad(FeatureList2[i])
        if i == 0:
            ws1 = torch.unsqueeze(m1, dim=-1)  # 扩展维度
            ws2 = torch.unsqueeze(m2, dim=-1)  # 扩展维度
        else:
            ws1 = torch.cat([ws1, torch.unsqueeze(m1, dim=-1)], dim=-1)  # 拼接
            ws2 = torch.cat([ws2, torch.unsqueeze(m2, dim=-1)], dim=-1)  # 拼接
    s1 = torch.mean(ws1, dim=-1) / c  # g_{i}
    s2 = torch.mean(ws2, dim=-1) / 2.0 * c  # g_{i}
    # s2 = torch.zeros_like(s1)
    s = F.softmax(torch.cat([torch.unsqueeze(s1, dim=-1), torch.unsqueeze(s2, dim=-1)], dim=-1), dim=-1)
    ssim_loss = torch.mean(s[:, 0] * SSIM1 + s[:,1] * SSIM2)
    mse_loss = torch.mean(s[:, 0] * mse1 + s[:,1] * mse2)
    '''
    print('ssim1', SSIM1)  # -1.3
    print('mse1', mse1)  # 0.01
    print('ssim2', SSIM2)  # -1.3
    print('mse2', mse2)  # 0.01
    print('ssim loss',ssim_loss) # -1.3
    print('mse loss',mse_loss)   # 0.01
    print('s1',s[:, 0])  # [0.50 0.50]
    print('s2',s[:, 1])  # [0.50 0.50]
    print('m1',m1) # [0.04 0.04]
    print('m2', m2)  # [0.04 0.04]
    print('ws1',ws1) # [0.04 0.04]
    print('ws2', ws2)  # [0.04 0.04]
    '''
    content_loss = ssim_loss + 20 * mse_loss
    return ssim_loss




