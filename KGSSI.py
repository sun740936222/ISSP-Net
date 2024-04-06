import torch
import torch.nn as nn
import torch.nn.functional as F


class PanStem0(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(PanStem0, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        return x

class PanStem1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PanStem1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x
class EMA(nn.Module):
    def __init__(self, channels, out_channels,pooling=True):
        super(EMA, self).__init__()
        # self.groups = factor
        # assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.softmax1=nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(out_channels, out_channels)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [x.size(2), x.size(3)], dim=2)
        x2 = x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        x3 = self.conv3x3(x)
        x21 = self.softmax(self.agp(x2).reshape(b, hw.size(1), -1).permute(0, 2, 1))
        x22 = x3.reshape(b, -1, x_h.size(2) * x_w.size(2))
        x30 = x3.reshape(b, x_h.size(2) * x_w.size(2),-1)
        x31 = self.gn(torch.matmul(x22,x30))
        x31 = torch.sum(x31,dim=1)
        x31 = self.softmax1(x31)
        x32 = x2.reshape(b, -1, x_h.size(2) * x_w.size(2))
        weights = (torch.matmul(x21, x22) + torch.matmul(x31, x32)).reshape(b, 1, x_h.size(2), x_w.size(2))
        return (x * weights.sigmoid()).reshape(b, -1, x_h.size(2), x_w.size(2))

class MsStem0(nn.Module):
    def __init__(self, in_channels=4, out_channels=32):
        super(MsStem0, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x
class MsStem1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MsStem1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels*2)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x
class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self,
                 op_channel: int,
                 out_channel:int,
                 pooling=True,
                 alpha: float = 4/ 8,
                 squeeze_radio: int = 1,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, out_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, out_channel, kernel_size=1, bias=False)
        if pooling:
            self.pooling1=nn.AvgPool2d(kernel_size=2,stride=2)
            self.PWC2pooling = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pooling1=nn.AvgPool2d(kernel_size=1,stride=1)
            self.PWC2pooling = nn.AvgPool2d(kernel_size=1, stride=1)
            self.pooling = nn.MaxPool2d(kernel_size=1, stride=1)
        self.FFT1 =nn.Conv2d(up_channel // squeeze_radio, out_channel, kernel_size=group_kernel_size, stride=1, padding=1,
                      bias=False)
        self.afterFFT=nn.BatchNorm2d(out_channel)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, out_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)
        #spatial
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.MaxPool2d(1)
        self.conv3x3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y=self.FFT1(up)
        Y=torch.fft.rfft2(Y,dim=(2,3),norm='ortho')
        weight=torch.view_as_complex(nn.Parameter(torch.randn(Y.size(1),Y.size(2),Y.size(3),2,dtype=torch.float32)*0.02))
        Y=(Y*weight)
        Y=torch.fft.irfft2(Y,dim=(2,3),norm='ortho')
        Y=self.afterFFT(Y)
        Y1 = self.GWC(up) + self.PWC1(up)+Y
        Y1 = self.pooling1(Y1)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        Y2 = self.PWC2pooling(Y2)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        #Spatial
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv3x3(x)
        return out1 + out2 + (x*self.sigmoid(x))
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stem = PanStem0(1, 64)
        self.ema1 = EMA(64, 64,pooling=False)
        self.stem1 = PanStem1(64,128)
        self.ema2 = EMA(128,128,pooling=False)
        self.ema3 = EMA(128,128,pooling=False)
        self.stem2 = PanStem1(128,256)
        self.ema4 = EMA(256,256,pooling=False)
        self.ema5 = EMA(256,256,pooling=False)
        self.ema6 = EMA(256,256,pooling=False)
        self.stem3 = PanStem1(256,512)
        self.ema7 = EMA(512,512,pooling=False)

        self.msstem= MsStem0(4, 64)
        self.cur1=CRU(64,64,pooling=False)
        self.msstem1 = MsStem1(64, 128)
        self.cru2 = CRU(128, 128,pooling=False)
        self.cur3 = CRU(128,128,pooling=False)
        self.msstem2 = MsStem1(128, 256)
        self.cru4 = CRU(256, 256,pooling=False)
        self.cru5 = CRU(256,256,pooling=False)
        self.cru6=CRU(256,256,pooling=False)
        self.msstem3 = MsStem1(256, 512)
        self.cru7 = CRU(512, 512,pooling=False)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.conn=nn.Linear(1024,11)
    def forward(self, ms,pan):
        x = self.stem(pan)
        x = self.ema1(x)
        x = self.stem1(x)
        x = self.ema2(x)
        x = self.ema3(x)
        x = self.stem2(x)
        x = self.ema4(x)
        x = self.ema5(x)
        x = self.ema6(x)
        x = self.stem3(x)
        x = self.ema7(x)

        x1=self.msstem(ms)
        x1=self.cur1(x1)
        x1=self.msstem1(x1)
        x1=self.cru2(x1)
        x1=self.cur3(x1)
        x1=self.msstem2(x1)
        x1=self.cru4(x1)
        x1=self.cru5(x1)
        x1 = self.cru6(x1)
        x1=self.msstem3(x1)
        x1=self.cru7(x1)

        x2=torch.cat([x,x1],dim=1)
        x2=self.avg(x2)
        x2=x2.view(x2.size(0),-1)
        x3=self.conn(x2)

        return x3
if __name__ == '__main__':
    model = Model()
    x=torch.randn(1,1,64,64)
    y=torch.randn(1,4,16,16)
    out=model(y,x)
    #获取模型参数量
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(out.shape)



