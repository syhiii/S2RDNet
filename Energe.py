import torch


def uncalculateE(HX, MS):
    C_HX, H_HX, W_HX = HX.shape  # 获取 HX 的维度
    C_MS, H_MS, W_MS = MS.shape  # 获取 MS 的维度
    device = HX.device
    # HX 和 MS 的通道数 C 应该相等
    assert C_HX == C_MS, "HX 和 MS 的通道数不一致"
    # 创建空的势能矩阵，用于存储每个像素点的势能
    potential_energy_HX = torch.zeros((C_HX, H_HX, W_HX), device=device)
    potential_energy_MS = torch.zeros((C_MS, H_MS, W_MS), device=device)
    # 创建 HX 和 MS 的坐标网格
    grid_i_HX, grid_j_HX = torch.meshgrid(torch.arange(H_HX, device=device), torch.arange(W_HX, device=device),
                                          indexing='ij')
    grid_i_HX = grid_i_HX.unsqueeze(0)  # Shape: [1, H_HX, W_HX]
    grid_j_HX = grid_j_HX.unsqueeze(0)  # Shape: [1, H_HX, W_HX]
    grid_i_MS, grid_j_MS = torch.meshgrid(torch.arange(H_MS, device=device), torch.arange(W_MS, device=device),
                                          indexing='ij')
    grid_i_MS = grid_i_MS.unsqueeze(0)  # Shape: [1, H_MS, W_MS]
    grid_j_MS = grid_j_MS.unsqueeze(0)  # Shape: [1, H_MS, W_MS]
    # 在同一个循环中计算 HX 和 MS 的势能
    for c in range(C_HX):
        # 对 HX 的当前通道计算势能
        image_HX = HX[c].unsqueeze(0)  # Shape: [1, H_HX, W_HX]
        distance_HX = torch.abs(grid_i_HX - grid_i_HX.transpose(1, 2)) + torch.abs(
            grid_j_HX - grid_j_HX.transpose(1, 2))  # Shape: [H_HX, W_HX, H_HX, W_HX]
        mask_HX = distance_HX != 0  # 忽略自身点的掩码
        potential_energy_HX[c] = (image_HX[mask_HX] / distance_HX[mask_HX]).sum()  # 使用掩码忽略自身点

        # 对 MS 的当前通道计算势能
        image_MS = MS[c].unsqueeze(0)  # Shape: [1, H_MS, W_MS]
        distance_MS = torch.abs(grid_i_MS - grid_i_MS.transpose(1, 2)) + torch.abs(
            grid_j_MS - grid_j_MS.transpose(1, 2))  # Shape: [H_MS, W_MS, H_MS, W_MS]
        mask_MS = distance_MS != 0  # 忽略自身点的掩码
        potential_energy_MS[c] = (image_MS[mask_MS] / distance_MS[mask_MS]).sum()  # 使用掩码忽略自身点

    return potential_energy_HX, potential_energy_MS

def uncalculateE2(HX):
    C_HX, H_HX, W_HX = HX.shape  # 获取 HX 的维度
    device = HX.device
    potential_energy_HX = torch.zeros((C_HX, H_HX, W_HX), device=device)
    # 创建 HX 和 MS 的坐标网格
    grid_i_HX, grid_j_HX = torch.meshgrid(torch.arange(H_HX, device=device), torch.arange(W_HX, device=device),
                                          indexing='ij')
    grid_i_HX = grid_i_HX.unsqueeze(0)  # Shape: [1, H_HX, W_HX]
    grid_j_HX = grid_j_HX.unsqueeze(0)  # Shape: [1, H_HX, W_HX]
    # 在同一个循环中计算 HX 和 MS 的势能
    for c in range(C_HX):
        # 对 HX 的当前通道计算势能
        image_HX = HX[c].unsqueeze(0)  # Shape: [1, H_HX, W_HX]
        distance_HX = torch.abs(grid_i_HX - grid_i_HX.transpose(1, 2)) + torch.abs(
            grid_j_HX - grid_j_HX.transpose(1, 2))  # Shape: [H_HX, W_HX, H_HX, W_HX]
        mask_HX = distance_HX != 0  # 忽略自身点的掩码
        potential_energy_HX[c] = (image_HX[mask_HX] / distance_HX[mask_HX]).sum()  # 使用掩码忽略自身
    return potential_energy_HX