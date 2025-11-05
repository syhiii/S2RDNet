import torch
import h5py
import cv2
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils import positionalencoding2d

def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)

        # tensor type:
        gt = data["gt"][...]
        gt = np.array(gt, dtype=np.float32) / float(gt.max())
        self.gt = torch.from_numpy(gt)

        lms = data["lms"][...]
        lms = np.array(lms, dtype=np.float32) / float(lms.max())
        self.lms = torch.from_numpy(lms)

        MS = data["ms"][...]
        ms = np.array(MS, dtype=np.float32) / float(MS.max())
        self.ms = torch.from_numpy(ms)

        MS = np.array(MS.transpose(0, 2, 3, 1), dtype=np.float32) / MS.max()
        ms_hp = get_edge(MS)
        self.ms_hp = torch.from_numpy(ms_hp).permute(0, 3, 1, 2)

        pan = data['pan'][...]
        pan = np.array(pan, dtype=np.float32) / float(pan.max())
        self.pan = torch.from_numpy(pan)

        self.pos = positionalencoding2d(64 * 4, 64, 64)



    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms_hp[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.pos[ :, :, :].float(),

    def __len__(self):
        return self.gt.shape[0]
class Dataset_Pro_full(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro_full, self).__init__()
        data = h5py.File(file_path)
        # tensor type:
        lms = data["lms"][...]
        lms = np.array(lms, dtype=np.float32) / 2308.
        self.lms = torch.from_numpy(lms)

        MS = data["ms"][...]


        pan = data['pan'][...]
        pan = np.array(pan, dtype=np.float32) / 2308.
        self.pan = torch.from_numpy(pan)

    def __getitem__(self, index):
        return self.lms[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(),

    def __len__(self):
        return self.lms.shape[0]
def prepare_training_data(dataname, batch_size=8):
    if dataname == 'wv3':
        train_set = Dataset_Pro("/data1/syh/syh/WV3/train_wv3.h5")
        validate_set = Dataset_Pro("/data1/syh/syh/WV3/valid_wv3.h5")
    elif dataname == 'gf2':
        train_set = Dataset_Pro("/data1/syh/syh/GF2/train_gf2.h5")
        validate_set = Dataset_Pro("/data1/syh/syh/GF2/valid_gf2.h5")
    elif dataname == 'qb':
        train_set = Dataset_Pro("/data1/syh/syh/QB/train_qb.h5")
        validate_set = Dataset_Pro("/data1/syh/syh/QB/valid_qb.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                      shuffle=True, pin_memory=True, drop_last=True)
    return training_data_loader, validate_data_loader