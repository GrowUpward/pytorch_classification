import torch
from torch import nn
from torchvision.models import mobilenetv2
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    该函数的作用是确保所有的层的通道数都能被8整除。它会将输入的通道数调整为最接近的可以被8整除的数，并且保证调整后的通道数不会比原来的数减少超过10%
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLU(nn.Sequential):
    def __init__(self,in_channel,out_channel,kernel_size=1,stride=1,groups=1):
        padding=(kernel_size-1)//2 #为了确保输出前后特征图的尺寸大小不变
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding,groups=groups,bias=False), #groups参数为1表示普通卷积，否则表示分组卷积
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self,in_channel,out_channel,stride,expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel=in_channel*expand_ratio
        self.use_shortcut=stride ==1 and in_channel==out_channel

        layers=[]
        if expand_ratio!=1:
            # 1x1 conv
            layers.append(ConvBNReLU(in_channel,hidden_channel,kernel_size=1))
            layers.extend([
            # 3x3 DW
            ConvBNReLU(hidden_channel,hidden_channel,stride=stride,groups=hidden_channel),#DW卷积的卷积数为输入特征矩阵的深度
            # 1x1 PW conv(linear)
            nn.Conv2d(hidden_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel)
        ])
        self.conv=nn.Sequential(*layers)#将所有层打包到一起
    def forward(self,x):
        if self.use_shortcut:
            return x+self.conv(x)
        else:
            return self.conv(x)
class MobileNetV2(nn.Module):
    def __init__(self,num_classes=1000,alpha=1.0,round_nearest=8):
        super(MobileNetV2, self).__init__()
        block=InvertedResidual
        input_channel=_make_divisible(32*alpha,round_nearest)
        last_channel=_make_divisible(1280*alpha,round_nearest)
        inverted_residual_setting=[
            #t,c,n,s(倍率因子，channels,num_classes,stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features=[]
        # conv1 layer
        features.append(ConvBNReLU(3,input_channel,stride=2))
        # building inverted residual residual blockes
        for t,c,n,s in inverted_residual_setting:
            output_channel=_make_divisible(c*alpha,round_nearest)
            for i in range(n):
                stride=s if i==0 else 1
                features.append(block(input_channel,output_channel,stride,expand_ratio=t))
                input_channel=output_channel
        # last
        features.append(ConvBNReLU(input_channel,last_channel,1))
        self.features=nn.Sequential(*features)

        # building classifier
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel,num_classes)
        )
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

model=MobileNetV2(num_classes=5)
print(model)