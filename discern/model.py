from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import Dropout
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from settings import *


class CaptchaModelCNN(Module):
    """用于识别验证码的卷积神经网络"""

    def __init__(self):
        super(CaptchaModelCNN, self).__init__()

        # 设定参数
        self.pool = 2  # 最大池化
        self.padding = 1  # 矩形边的补充层数
        self.dropout = 0.5  # 随机概率
        self.kernel_size = 3  # 卷积核大小 3x3

        # 卷积池化
        self.layer1 = Sequential(
            # 时序容器Sequential,参数按顺序传入
            # 2维卷积层，卷积核大小为self.kernel_size，边的补充层数为self.padding
            Conv2d(1, 32, kernel_size=self.kernel_size, padding=self.padding),
            # 对小批量3d数据组成的4d输入进行批标准化(Batch Normalization)操作
            BatchNorm2d(32),
            # 随机将输入张量中部分元素设置为0，随机概率为self.dropout。
            Dropout(self.dropout),
            # 对输入数据运用修正线性单元函数
            ReLU(),
            # 最大池化
            MaxPool2d(2))

        # 卷积池化
        self.layer2 = Sequential(
            Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.padding),
            BatchNorm2d(64),
            Dropout(self.dropout),
            ReLU(),
            MaxPool2d(2))

        # 卷积池化
        self.layer3 = Sequential(
            Conv2d(64, 64, kernel_size=self.kernel_size, padding=self.padding),
            BatchNorm2d(64),
            Dropout(self.dropout),
            ReLU(),
            MaxPool2d(2))

        # 全连接
        self.fc = Sequential(
            Linear((IMAGE_WIDTH // 8) * (IMAGE_HEIGHT // 8) * 64, 1024),
            Dropout(self.dropout),
            ReLU())
        self.rfc = Sequential(Linear(1024, CAPTCHA_NUMBER * len(CHARACTER)))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out
