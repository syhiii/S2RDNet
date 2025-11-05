from __future__ import print_function
# 20241209
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datetime import datetime
from metric import ref_evaluate, no_ref_evaluate
from dataset import prepare_training_data, Dataset_Pro, Dataset_Pro_full
import pywt
from tqdm import tqdm
from loss_spa import custom_loss,calculate_different_size, retainLoss
from loss_spe import MultiMetricLoss, Spectral_Wavelet_Loss
import torch.nn.functional as F
## 2024 10 09

# from data import get_data
# 定义 PSNR 计算函数
def calculate_psnr(img1, img2, dynamic_range=1):
    """PSNR metric, img uint8 if 225; uint16 if 2047"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    dynamic_range = img1_.max()
    mse = np.mean((img1_ - img2_) ** 2)
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / (np.sqrt(mse) + np.finfo(np.float64).eps))


def save_npy(image_tensor, file_path):
    image = image_tensor.detach().cpu().numpy()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, image)


# Training settings 24.3.13
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=8, help='output channel number')  # gf2:4 wv3:8   qb:4
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002,help='learningRate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='/data1/syh/output/uns_output/TrainedNet_wv3_big/',
                    help='Directory to keep training outputs.')  # /data1/syh/output/model/TrainedNet panmamba gf2
parser.add_argument('--outputpath', type=str, default='/data1/syh/output/uns_output/resultswv3_big/',
                    help='Path to output img')
parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
parser.add_argument('--local_rank', default=1, type=int, help='None')
parser.add_argument('--imsz', type=int, default=64, help='None')
parser.add_argument('--dataset', type=str, default='wv3', help='dataset')

# parser.add_argument('--algorithm', type=str, default='dispnet', help='dataset')
opt = parser.parse_args()
device = 'cuda:2'
print(opt)
from model import Net

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(opt.seed)
model = Net(num_channels=opt.ChDim).to(device)

# model = Net222().to(device)
print('===> Loading datasets')

training_data_loader, testing_data_loader = prepare_training_data(dataname=opt.dataset,
                                                                  batch_size=opt.batchSize)  # gf2 wv3

print('===> Building model')
print("===> distribute model")

print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
# print('# network parameters: {}'.format(PS_trainer.count_model_parameters()))
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[100, 150, 175, 190, 195], gamma=0.5)

# criterion = nn.L1Loss(reduction='none')
criterion = torch.nn.MSELoss()

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
# tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0

from scipy.ndimage import gaussian_filter
def decompose_image(image, sigma=5):
    # 将 GPU 上的 Tensor 转为 CPU 上的 NumPy 数组
    image_np = image.cpu().detach().numpy()
    B, C, H, W = image_np.shape
    # 初始化低频和高频分量
    low_freq = np.zeros_like(image_np)
    high_freq = np.zeros_like(image_np)
    # 对每个样本的每个通道进行高斯滤波
    for b in range(B):
        for c in range(C):
            low_freq[b, c] = gaussian_filter(image_np[b, c], sigma=sigma)
            high_freq[b, c] = image_np[b, c] - low_freq[b, c]
    # 将结果转换回 PyTorch tensor，并移回 GPU
    low_freq_tensor = torch.from_numpy(low_freq).to(image.device)
    high_freq_tensor = torch.from_numpy(high_freq).to(image.device)
    return low_freq_tensor
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


mkdir(opt.save_folder)
mkdir(opt.outputpath)

if opt.nEpochs != 0:
    load_dict = torch.load(opt.save_folder + "_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])


def downsample(image, r=4):
    if not image.is_cuda:
        image = image.cuda()
    B, C, H, W = image.shape
    new_height = H // r
    new_width = W // r
    downsampled_image = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return downsampled_image
def get_factor(x):
    return np.float32(np.sqrt(x))
def LossConv(Z):
    ksz=4
    # 正值约束：计算负数的平方和作为惩罚项
    positive_loss = torch.sum(torch.relu(-Z))  # 如果有负数，则返回它们的总和

    # 中心对称性约束：计算不满足对称性的平方和作为惩罚项
    symmetry_loss = torch.sum((Z - Z.flip([0, 1]))**2) / (ksz * ksz)

    return positive_loss + symmetry_loss

def compute_frobenius_norm(tensor1, tensor2):
    assert tensor1.shape[2:] == tensor2.shape[2:], "Spatial dimensions must match"
    B, C1, H, W = tensor1.shape
    _, C2, _, _ = tensor2.shape
    tensor1_flat = tensor1.view(B, C1, -1)  # Shape: (B, C1, H*W)
    tensor2_flat = tensor2.view(B, C2, -1)  # Shape: (B, C2, H*W)
    R = torch.bmm(tensor1_flat, tensor2_flat.transpose(1, 2))  # Shape: (B, C1, C2)
    frobenius_norm = torch.norm(R, p='fro')  # Frobenius norm
    return frobenius_norm.item()

# 计算SVD的相似性
def svd_similarity(A, B):
    batch, C1, H, W = A.shape
    _, C2, _, _ = B.shape
    A_flat = A.view(batch, C1, -1)  # Shape: [B, C1, H*W]
    B_flat = B.view(batch, C2, -1)  # Shape: [B, C2, H*W]
    # Singular Value Decomposition (SVD)
    svd_A = torch.linalg.svd(A_flat, full_matrices=False).S
    svd_B = torch.linalg.svd(B_flat, full_matrices=False).S
    # Normalize singular values
    svd_A_norm = F.normalize(svd_A, p=2, dim=-1)
    svd_B_norm = F.normalize(svd_B, p=2, dim=-1)
    # Compute cosine similarity
    similarity = torch.sum(svd_A_norm * svd_B_norm, dim=-1)
    return similarity
# 计算SSIM的相似性
def train(epoch, optimizer, scheduler):
    avg_loss = 0
    avg_psnr_cnn = 0
    avg_ssim = 0
    max_psnr = 0
    global current_step
    model.train()
    ref_results_all = []
    # temp = 1 if epoch % 2 == 0 else 0  # 偶数 epoch 执行 runk=1，奇数 epoch 执行 runk=0
    for iteration, batch in tqdm(enumerate(training_data_loader, 1)):
        X, Z, Y, pos = batch[0].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device)  # gt pan ms pos
        optimizer.zero_grad()
        resLspec_cnn, _, _, phi,phi2,Sc, kernel, PANF, LMSF, _, _, _ = model(downsample(Y), downsample(Z), downsample(pos), phi=None,phi2=None,Sc=None, kernel=None, imsz=16)  # ms pan
        K = kernel
        P = phi
        resspec_cnn, spec_pred_cnn, pre_ryb_cnn, phi,phi2,Sc, kernel, PANF, LMSF, v, AY, BY_ = model(Y, Z, pos, phi=phi, phi2=phi2, Sc = Sc, kernel=kernel)  # ms pan
        K2 = kernel
        P2 = phi
        # v shape B, C-1 , H, W
        l1loss = criterion(resLspec_cnn,Y)
        ryb_loss_cnn = criterion(pre_ryb_cnn, Z) * 1.0
        spec_loss_cnn = criterion(spec_pred_cnn, Y)
        # frobenius = compute_frobenius_norm(Z, v)
        tv_wei = 1.5
        lowrank_wei = 0.7
        tv_loss = 1 * 5e-6 * tv_wei * torch.sum(torch.abs(phi[1:, :] - phi[:-1, :])) * 1.0
        U, S, Vh = torch.linalg.svd(kernel, full_matrices=True)
        lr_loss = 1 * 1e-4 * lowrank_wei * torch.sum(S) * 1.0
        tv_img_loss_cnn = 5e-7 * torch.sum(torch.abs(resspec_cnn[:, 1:, :, :] - resspec_cnn[:, :-1, :, :]))
        loss_all = spec_loss_cnn + ryb_loss_cnn + 0.1 * l1loss # + 1e-4 * frobenius
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        '''
        if epoch % 2 == 0 and iteration == 1000:
            save_npy(resspec_cnn, f'/data1/syh/output/uns_output/results2/resspec_cnn_epoch_{epoch}.npy')
            save_npy(X, f'/data1/syh/output/uns_output/results2/X_epoch_{epoch}.npy')
            save_npy(Y, f'/data1/syh/output/uns_output/results2/Y_epoch_{epoch}.npy')
            save_npy(Z, f'/data1/syh/output/uns_output/results2/Z_epoch_{epoch}.npy')
            save_npy(ryb_pred_cnn, f'/data1/syh/output/uns_output/results2/ryb_pred_cnn_epoch_{epoch}.npy')
            save_npy(spec_pred_cnn, f'/data1/syh/output/uns_output/results2/spec_pred_cnn_epoch_{epoch}.npy')
            save_npy(kernel, f'/data1/syh/output/uns_output/results2/kernel_epoch_{epoch}.npy')
            save_npy(phi, f'/data1/syh/output/uns_output/results2/phi_epoch_{epoch}.npy')
            print('Save True')
        '''
        #
        if iteration == 1000:
            save_npy(resspec_cnn, f'/data1/syh/output/uns_output/results2/resspec_cnn_epoch_{epoch}.npy')
            save_npy(X, f'/data1/syh/output/uns_output/results2/X_epoch_{epoch}.npy')
            save_npy(Y, f'/data1/syh/output/uns_output/results2/Y_epoch_{epoch}.npy')
            save_npy(Z, f'/data1/syh/output/uns_output/results2/Z_epoch_{epoch}.npy')
            save_npy(spec_pred_cnn, f'/data1/syh/output/uns_output/results2/spec_pred_cnn_epoch_{epoch}.npy')
            save_npy(kernel, f'/data1/syh/output/uns_output/results2/kernel_epoch_{epoch}.npy')
            save_npy(phi, f'/data1/syh/output/uns_output/results2/phi_epoch_{epoch}.npy')
            save_npy(phi2, f'/data1/syh/output/uns_output/results2/phi2_epoch_{epoch}.npy')
            save_npy(v, f'/data1/syh/output/uns_output/results2/v_epoch_{epoch}.npy')
            save_npy(torch.matmul(phi, Z.view(1, 64 * 64)).view(-1, 8, 64, 64), f'/data1/syh/output/uns_output/results2/Y1_epoch_{epoch}.npy')
            save_npy(torch.matmul(phi2, v.view(7, 64 * 64)).view(-1, 8, 64, 64), f'/data1/syh/output/uns_output/results2/Y2_epoch_{epoch}.npy')

            print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
            print('spec_loss_cnn', spec_loss_cnn)
            print('ryb_loss_cnn', ryb_loss_cnn)
            print('l1loss',l1loss)
            print('tv_loss', tv_loss)
            print('lr_loss', lr_loss)
            print('tv_img_loss_cnn', tv_img_loss_cnn)
            #print('frobenius', frobenius)
            print('Save True')
        for i5 in range(resspec_cnn.shape[0]):
            temp_ref_results = ref_evaluate(np.transpose(resspec_cnn[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                np.transpose(X[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                scale=4)
            ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))


            # print('psnr:', psnr_total / psnr_count)
    # print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
    print('spec_loss_cnn', spec_loss_cnn)
    print('tv_loss', tv_loss)
    print('lr_loss', lr_loss)
    print('tv_img_loss_cnn',tv_img_loss_cnn)
    # print('frobenius', frobenius)
    # print('lossconv',LossConv(kernel))
    # print('maploss',map_loss)
    # save_npy(HX, f'/data1/syh/output/uns_output/results/HX_epoch_{epoch}.npy')
    # save_npy(X, f'/data1/syh/output/uns_output/results/X_epoch_{epoch}.npy')
    return phi,phi2,Sc, kernel


def test(phi,phi2,Sc, kernel):
    print('in test')
    avg_psnr = 0
    avg_time = 0
    model.eval()
    with torch.no_grad():
        ref_results_all = []
        noref_results_all = []
        for iteration, batch in tqdm(enumerate(testing_data_loader, 1)):
            X, Z, Y, pos = batch[0].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device)  # gt pan ms pos
            HX, _, _, _, _, _, _, _, _, _, _, _ = model(Y, Z, pos, phi=phi,phi2=phi2, Sc = Sc, kernel=kernel,test=True)  # ms pan
            # shape (8,8,64,64)
            for i5 in range(HX.shape[0]):
                temp_ref_results = ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                np.transpose(X[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                scale=4)
                ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))

                temp_noref_results = no_ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Z[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Y[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     scale=4, block_size=16)
                noref_results_all.append(np.expand_dims(temp_noref_results, axis=0))

    print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
    print('value :', np.mean(noref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse
    return np.mean(ref_results_all, axis=0)[0] / len(testing_data_loader)


def checkpoint(epoch):
    model_out_path = opt.save_folder + "_epoch_{}.pth".format(epoch)
    if epoch % 1 == 0:
        save_dict = dict(
            lr=optimizer.state_dict()['param_groups'][0]['lr'],
            param=model.state_dict(),
            adam=optimizer.state_dict(),
            epoch=epoch
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))


if opt.mode == 1:
    for epoch in range(opt.nEpochs + 1, 201):
        phi,phi2,Sc, kernel = train(epoch, optimizer, scheduler)
        if epoch % 2 == 0:
            test(phi,phi2,Sc, kernel)
            checkpoint(epoch)
        torch.cuda.empty_cache()
        # if local_rank == 0:
        # tb_logger.add_scalar('psnr', avg_psnr, epoch)
        scheduler.step()


elif opt.mode == 3:  ## test gf2
    print('test\n')
    save_dir = '/data1/syh/output/dct/results_panmamba/'
    if opt.dataset == 'gf2':
        test_set = Dataset_Pro("/data1/syh/syh/GF2/test_gf2_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    elif opt.dataset == 'wv3':
        test_set = Dataset_Pro("/data1/syh/syh/WV3/test_wv3_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    elif opt.dataset == 'qb':
        test_set = Dataset_Pro("/data1/syh/syh/QB/test_qb_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    with torch.no_grad():
        ref_results_all = []
        for batch in testing_data_loader:
            X, Z, Y = batch[0].to(device), batch[3].to(device), batch[4].to(device)  # gt pan ms
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            X = Variable(X).float()
            temp = X.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_gt.npy'.format(opt.algorithm, opt.dataset), temp)

            temp = Y.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_ms.npy'.format(opt.algorithm, opt.dataset), temp)
            temp = Z.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_pan.npy'.format(opt.algorithm, opt.dataset), temp)

            start_time = time.time()
            HX = model(Y, Z)
            temp = HX.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_f.npy'.format(opt.algorithm, opt.dataset), temp)

            print('---')
            input()
            end_time = time.time()
            # shape (8,8,64,64) bchw
            for i5 in range(HX.shape[0]):
                temp_ref_results = ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                np.transpose(X[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                scale=4)
                ref_results_all.append(np.expand_dims(temp_ref_results, axis=0))
            print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_ergas, c_scc, c_q, c_rmse
    print('value :', np.mean(ref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse



elif opt.mode == 4:  ## 测试 full
    print('test\n')
    model.eval()
    if opt.dataset == 'gf2':
        test_set = Dataset_Pro_full("/data1/syh/syh/GF2/test_gf2_OrigScale_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    elif opt.dataset == 'wv3':
        test_set = Dataset_Pro_full("/data1/syh/syh/WV3/test_wv3_OrigScale_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    elif opt.dataset == 'qb':
        test_set = Dataset_Pro_full("/data1/syh/syh/QB/test_qb_OrigScale_multiExm1.h5")
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=2, shuffle=False, pin_memory=True,
                                         drop_last=True)
    with torch.no_grad():
        noref_results_all = []

        for batch in testing_data_loader:
            Z, Y = batch[1].to(device), batch[2].to(device)  # pan ms
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            start_time = time.time()
            HX = model(Y, Z)
            temp = HX.data.cpu().numpy()  # 数据类型转换
            np.save('/data1/syh/output/dcfn_result/{}_{}_full_f.npy'.format(opt.algorithm, opt.dataset), temp)
            print('---')
            input()
            end_time = time.time()
            # print('D_lambda :', D_lambda)
            for i5 in range(Z.shape[0]):
                temp_noref_results = no_ref_evaluate(np.transpose(HX[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Z[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     np.transpose(Y[i5, :, :, :].detach().cpu().numpy(), [1, 2, 0]),
                                                     scale=4)
                noref_results_all.append(np.expand_dims(temp_noref_results, axis=0))
            print('value :', np.mean(noref_results_all, axis=0))  # c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse


else:
    test()
