import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),  # input[3, 224, 224]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),               # output[48, 27, 27]

            nn.Conv2d(48,128,kernel_size=5,stride=1,padding=2), # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),               # output[128, 13, 13]

            nn.Conv2d(128,192,kernel_size=3,stride=1,padding=1),# output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),# output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3,stride=1,padding=1),# output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)                # output[128, 6, 6]

        )
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weight:
            self._initialize_weights()
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)#[batch,channel,height,width],将channel高度宽度这三个维度展成一个一维向量
        x=self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():#返回一个迭代器遍历我们定义的所有层结构
            if isinstance(m,nn.Conv2d):#如果是卷积层
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')#使用此函数进行初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):#如果是全连接层，则进行如下初始化
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

