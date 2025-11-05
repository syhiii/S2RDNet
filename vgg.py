import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG161(nn.Module):
    def __init__(self,c1,dim,c2):
        super(VGG161, self).__init__()

        self.convx = nn.Conv2d(c1,dim,3,1,1)
        self.convy = nn.Conv2d(c2,dim,3,1,1)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2*dim, 64, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64,  3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 2
            nn.Conv2d(64, 128,  3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128,  3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 3
            nn.Conv2d(128, 256,  3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,  3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256,  3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 4
            nn.Conv2d(256, 512, 3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,  3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512,  3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 5
            nn.Conv2d(512, 512,  3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256,  3,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, c2,  3,1,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),
        )


    def forward(self, pan, ms):
        x = self.convx(pan)
        upms = F.interpolate(ms, size=(pan.shape[2], pan.shape[3]), mode='bilinear', align_corners=False)  # Upsample to match PAN size
        y = self.convy(upms)

        out = self.features(torch.cat([x,y],1))
        # 将卷积层输出 flatten
        return upms + out


class VGG162(nn.Module):
    def __init__(self,c1,dim,c2):
        super(VGG162, self).__init__()

        self.convx = nn.Conv2d(c1,dim,3,1,1)
        self.convy = nn.Conv2d(c2,dim,3,1,1)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2*dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,0),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2,0),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),
        )


    def forward(self, pan, MS):
        x = self.convx(pan)
        y = self.convy(MS)

        out = self.features(torch.cat([x,y],1))
        # 将卷积层输出 flatten
        return out


class VGG163(nn.Module):
    def __init__(self,c1,dim,c2):
        super(VGG163, self).__init__()

        self.convx = nn.Conv2d(c1,dim,3,1,1)
        self.convy = nn.Conv2d(c2,dim,3,1,1)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(2*dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1),
        )


    def forward(self, ms, MS):
        upms = F.interpolate(ms, size=(MS.shape[2], MS.shape[3]), mode='bilinear',
                             align_corners=False)  # Upsample to match PAN size
        x = self.convx(upms)
        y = self.convy(MS)

        out = self.features(torch.cat([x,y],1))
        # 将卷积层输出 flatten
        return out