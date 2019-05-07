import torch
import logging
from torch.nn import MultiLabelSoftMarginLoss
from torch.autograd import Variable
from torch.optim import Adam
from settings import *
from model import CaptchaModelCNN
from loader import loaders
logging.basicConfig(level=logging.INFO)


def start_train():
    # 使用自定义的卷积神经网络训练
    model = CaptchaModelCNN().cuda()
    model.train()  # 训练模式
    logging.info('Train start')
    # 损失函数
    criterion = MultiLabelSoftMarginLoss()
    # Adam算法
    optimizer = Adam(model.parameters(), lr=RATE)
    ids = loaders(PATH_TRAIN, BATCH_SIZE)
    logging.info('Iteration is %s' % len(ids))
    for epoch in range(EPOCHS):
        for i, (image, label, order) in enumerate(ids):
            # 包装Tensor对象并记录其operations
            images = Variable(image).cuda()
            labels = Variable(label.float()).cuda()
            predict_labels = model(images)
            loss = criterion(predict_labels, labels)
            # 保持当前参数状态并基于计算得到的梯度进行参数更新。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            if i % 100 == 0:
                logging.info("epoch:%s, step:%s, loss:%s" % (epoch, i, loss.item()))
                # 保存训练结果
                torch.save(model.state_dict(), MODEL_NAME)
    # 保存训练结果
    torch.save(model.state_dict(), MODEL_NAME)
    logging.info('Train done')


if __name__ == '__main__':
    start_train()
