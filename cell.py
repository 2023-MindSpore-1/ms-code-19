import mindspore
from mindspore import nn
import mindspore.nn.functional as F

from similarity_of_poi_flowdata import *

# ------------------- completion net -------------------
# adaptive convolution net
class gen_weight(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, feature):
        super(gen_weight, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.dimension = feature.shape[0] * feature.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(self.dimension, 4096, bias=True),
            nn.ReLU(),
            nn.Linear(4096, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, self.out_channel * self.in_channel * self.kernel_size * self.kernel_size, bias=True),
            nn.ReLU(),
        )
    def forward(self, e):
        e = mindspore.Tensor.from_numpy(e.reshape(-1))
        w = self.fc(e)
        w = w.view(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        return w

# generate the kernel weight
def gen_kernel(in_channel, out_channel, kernel_size, feature):
    if feature is None:
        weight = mindspore.ops.zeros(out_channel, in_channel, kernel_size, kernel_size)
    else:
        gen_w = gen_weight(in_channel, out_channel, kernel_size, feature)
        weight = gen_w(feature)
    return weight

class AdaCNN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, feature, device):
        super(AdaCNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.feature = feature
        self.padding = int((kernel_size - 1) / 2)
        self.wx = nn.Parameter(gen_kernel(self.in_channel, self.out_channel, self.kernel_size, self.feature), requires_grad=True).to(device)
        self.wh = nn.Parameter(gen_kernel(self.in_channel, self.out_channel, self.kernel_size, self.feature), requires_grad=True).to(device)

    def forward(self, h, x):
        output = mindspore.ops.Sigmoid(F.conv2d(x, self.wx, padding=1) + F.conv2d(h, self.wh, padding=1))
        return output

class AuxCMP(nn.Module):
    def __init__(self, poi, device):
        super(AuxCMP, self).__init__()
        self.device =device
        poi_index, _ = POI_similarity(poi)
        self.poi_index = poi_index
        self.weight = nn.Parameter(mindspore.Tensor(0.5) , requires_grad=True)

    # d-(64, 5, 2, 16, 16)
    def forward(self, d, m):
        # (64, 5, 2, 16, 16) --->(64, 2, 256)
        _, _, _, _, side = d.shape
        data = d[:, -1].reshape(d.shape[0], d.shape[2], -1).to(self.device)
        mask = m[:, -1].reshape(d.shape[0], d.shape[2], -1).to(self.device)
        poicmp = mindspore.ops.zeros((d.shape[0], d.shape[2], side * side)).to(self.device)
        for i in range(side * side):
            if mask[:, :, i].sum() == 0:
                poiindex = self.poi_index[i]
                poicmp[:, :, i] = data[:, :, poiindex]
        data_poi = (data + poicmp).reshape(d.shape[0], d.shape[2], side, side)
        output = data_poi
#       data_aux = auxnet(d)
#       output  = self.weight * data_poi + (1 - self.weight) * data_aux
        return output

# ------------------- super resolution net -------------------
# feature extraction
class FeaExtraction(nn.Module):
    def __init__(self, side):
        super(FeaExtraction, self).__init__()
        self.side = side
        self.ext = nn.Sequential(
            nn.Linear(4, self.side * self.side),
            nn.ReLU(inplace=True),
        )

    def forward(self, f):
        batch, _ = f.shape
        result = self.ext(f).view(batch, 1, self.side, self.side)
        return result

# upsample block
class UpscaleBlock(nn.Module):
    def __init__(self, channel, scaler_n):
        super(UpscaleBlock, self).__init__()
        num = int(1)
        upscale = []
        for _ in range(num):
            upscale += [nn.Conv2d(channel, 4 * channel, 3, 1, 1),
                        nn.BatchNorm2d(4 * channel),
                        nn.PixelShuffle(upscale_factor=2),
                        nn.ReLU(inplace=True)]
        self.scale = nn.Sequential(*upscale)

    def forward(self, x):
        output = self.scale(x)
        return output

# downsample block
class DownscaleBlcok(nn.Module):
    def __init__(self, channel, scaler_n):
        super(DownscaleBlcok, self).__init__()
        num = int(1)
        downscale = []
        for _ in range(num):
            downscale += [nn.Conv2d(channel, channel, 2, 2, 0),
                        nn.BatchNorm2d(channel),
                        nn.ReLU(inplace=True)]
        self.scale = nn.Sequential(*downscale)

    def forward(self, x):
        output = self.scale(x)
        return output

# raw resnet
class Resblock(nn.Module):
    def __init__(self, channel):
        super(Resblock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
        )
    def forward(self, x):
        output = mindspore.ops.add(x, self.resblock(x))
        return output

class Resnet(nn.Module):
    def __init__(self, channel, resnet_n):
        super(Resnet, self).__init__()
        resnet = []
        for _ in range(resnet_n):
            resnet.append(Resblock(channel))
        self.resnet = nn.Sequential(*resnet)

    def forward(self, x):
        output = mindspore.ops.add(x, self.resnet(x))
        return output

# resblock adding upscaling and downscaling
class ResExtraction(nn.Module):
    def __init__(self, channel, scaler_n):
        super(ResExtraction, self).__init__()
        self.upscale = UpscaleBlock(channel, scaler_n)
        self.resext = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
        )
        self.downscale = DownscaleBlcok(channel, scaler_n)

    def forward(self, x):
        x0 = x
        x1 = self.upscale(x0)
        x2 = mindspore.ops.add(x1, self.resext(x1))
        output = mindspore.ops.add(x0, self.downscale(x2))
        return output

# extract feature of SR module
class ExtractNet(nn.Module):
    def __init__(self, channel, scaler_n, resnet_n):
        super(ExtractNet, self).__init__()
        # concat feature with 1 channel
        self.net1 = nn.Sequential(
            nn.Conv2d(channel, channel, 9, 1, 4),
            nn.ReLU(inplace=True),
        )
        resnet = []
        for _ in range(resnet_n):
            resnet.append(ResExtraction(channel, scaler_n))
        self.resnet = nn.Sequential(*resnet)
        self.net2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
        )

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.resnet(x1)
        output = mindspore.ops.add(x1, self.net2(x2))
        return output

class Distribution_upsampling(nn.Module):
    def __init__(self, scaler_n):
        super(Distribution_upsampling, self).__init__()
        self.scaler_n = scaler_n
        self.avgpool = nn.AvgPool2d(scaler_n)
        self.upsample = nn.Upsample(scale_factor=scaler_n, mode='nearest')
        self.epsilon = 1e-5

    def forward(self, x):
        out = self.avgpool(x) * self.scaler_n ** 2
        out = self.upsample(out)
        return mindspore.ops.div(x, out + self.epsilon)

class combine_with_distribution(nn.Module):
    def __init__(self, scaler_n):
        super(combine_with_distribution, self).__init__()
        self.scaler_n = scaler_n
        self.upsample = nn.Upsample(scale_factor=scaler_n, mode='nearest')

    def forward(self, hr_data, lr_data):
        out = self.upsample(lr_data)
        return mindspore.ops.Mul(hr_data, out)





