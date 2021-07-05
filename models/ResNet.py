from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d

__all__ = ['ResNet50', 'ResNet101']

class ResNet50(nn.Module):
    """
    1.**kwargs接入符 可有可无参数
    2.一个网络有两部分，一个inition，一个forward
    """
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        #使用super继承model库
        super(ResNet50, self).__init__()
        self.loss = loss
        #使用自带的resnet50，并且使用预训练模型,如果数据集足够大，在其他任务中也可加长训练时间实现同等效果
        resnet50 = torchvision.models.resnet50(pretrained=True)
        #将网络转换为list，提取其中的除了后两行(FC和globel pooling（经常变化，与图片有关，一般需要重写）)的其他层，再使用pytorch提供的Sequential转换回去
        #children()是generator迭代器,这是网络的大部分
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        #定义自己的分类器 输入2048维度，和类别数量
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    """"
    只需要自定义前向传播，反向传播自动
    """
    def forward(self, x):
        #此时x是feature map
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        #使用torch.functional的avg_pool2d，更新维度为32 * 2048 * 1 * 1
        x = F.avg_pool2d(x, x.size()[2:])
        #使用view函数，展平,多行的Tensor,拼接成一行,下一行接到屁股上去，
        # 保证batch size的32维不能改变 32,2048
        #f 用于检索的特征向量
        f = x.view(x.size(0), -1)
        #如果需要做归一化 torch.norm 在-1维（batchSize）点乘 （acc会略微下降）
        # 保持维度不变参数 keepdim=True
        #分母需要加入很小的值，防止梯度爆炸 + 1e-12
        #f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        #推理阶段（不是训练阶段）：直接ruturn 特征，进行检索 ，不需要向下运行
        #model.eval 可以转换为评估（测试）模式
        if not self.training:
            return f,lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return  f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=False)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f, lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
