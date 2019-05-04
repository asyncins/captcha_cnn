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
    """定义卷积网络"""

    def __init__(self):
        super(CaptchaModelCNN, self).__init__()

        # 设定参数
        self.pool = 2  # 2维最大池化
        self.padding = 1  # 边的补充层数
        self.dropout = 0.5
        self.kernel_size = 3  # 卷积核大小 3x3

        # 第1层，卷积池化
        self.layer1 = Sequential(
            # 时序容器Sequential,参数按顺序传入
            Conv2d(1, 32, kernel_size=self.kernel_size, padding=self.padding),
            BatchNorm2d(32),
            Dropout(self.dropout),
            ReLU(),
            MaxPool2d(2))

        # 第2层，卷积池化
        self.layer2 = Sequential(
            Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.padding),
            BatchNorm2d(64),
            Dropout(self.dropout),
            ReLU(),
            MaxPool2d(2))

        # 第3层，卷积池化
        self.layer3 = Sequential(
            Conv2d(64, 64, kernel_size=self.kernel_size, padding=self.padding),
            BatchNorm2d(64),
            Dropout(self.dropout),
            ReLU(),
            MaxPool2d(2))

        # 全连接
        self.fc = Sequential(
            # Linear(每个输入样本的大小, 每个输出样本的大小, 学习偏置)
            # 对输入数据做线性变换：y=Ax+b
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
