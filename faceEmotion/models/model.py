import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy
import torchvision
from .BasicNet import *

class ERNet1(BasicNet):
    """Some Information about MyModule"""
    def __init__(self, numClass):
        super(ERNet, self).__init__()
        self.numClass = numClass

        self.pre_layer =nn.Sequential(

            nn.Conv2d(1,32,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=2,stride=1),
            nn.PReLU()
        )
        self.fc_1 = nn.Linear(2*2*128,256)
        self.prelu_1 = nn.PReLU()
        self.fc = nn.Linear(256,self.numClass)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)
        x = self.fc_1(x)
        x = self.prelu_1(x)
        output = self.fc(x)
        return output

class ERNet(BasicNet):
    """Some Information about MyModule"""
    def __init__(self, numClass):
        super(ERNet, self).__init__()
        self.numClass = numClass

        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc.apply(self.weights_init_kaiming)  # 初始化权重

        # for c in range(numClass):
        #     self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.numBottleNeck, class_num=1, activ='sigmoid') )

        # 将特征映射到对应类别
        self.classifierLayer = nn.Sequential(
            nn.Linear(self.numBottleNeck, numClass)
        )
        self.classifierLayer.apply(self.weights_init_classifier)  # 初始化权重

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        # output = [self.__getattr__('class_%d' % c)(x) for c in range(self.numClass)]
        # output = torch.cat(output, dim=1)
        return  self.classifierLayer(x)

if __name__ == "__main__":
    model =  ERNet(7)
    testTensor = torch.Tensor(2,1,42,42);
    # print(testTensor)
    # print(model.parameters().class_0)
    out = model(testTensor)
    # label = torch.zeros(1, 7).scatter_(1, torch.tensor([[2,3]]), 1)
    # print(label)
    print(out.shape)