import numpy
import torch
import logging
from torch.autograd import Variable

from settings import *
from convert import one_hot_decode
from loader import loaders
from model import CaptchaModelCNN

logging.basicConfig(level=logging.INFO)


def start_verifies(folder):
    model = CaptchaModelCNN().cuda()
    model.eval()  # 预测模式
    # 载入模型
    model.load_state_dict(torch.load(MODEL_NAME))
    logging.info('load cnn model')
    verifies = loaders(folder, 1)
    correct, total, current, cha_len,  = 0, 0, 0, len(CHARACTER)
    for i, (image, label, order) in enumerate(verifies):
        captcha = one_hot_decode(order)  # 正确的验证码
        images = Variable(image).cuda()
        predict_label = model(images)
        predicts = []
        for k in range(CAPTCHA_NUMBER):
            # 根据预测结果取值
            code = one_hot_decode([(numpy.argmax(predict_label[0, k * cha_len: (k + 1) * cha_len].data.cpu().numpy()))])
            predicts.append(code)
        predict = ''.join(predicts)  # 预测结果
        current += 1
        total += 1
        if predict == captcha:
            logging.info('Success, captcha:%s->%s' % (captcha, predict))
            correct += 1
        else:
            logging.info('Fail, captcha:%s->%s' % (captcha, predict))
        if total % 300 == 0:
            logging.info('当前预测图片数为%s张，准确率为%s%%' % (current, int(100 * correct / current)))
    logging.info('完成。数据集%s当前预测图片数为%s张，准确率为%s%%' % (folder, total, int(100 * correct / total)))


def get_image_name(folder):
    # 加载指定路径下的图片，并返回图片名称列表
    images = list(pathlib.Path(folder).glob('*.{}'.format(IMAGE_TYPE)))
    image_name = [i.stem for i in images]
    return image_name


if __name__ == '__main__':
    folders = PATH_TEST  # 指定预测集路径
    trains = get_image_name(PATH_TRAIN)  # 获取训练样本所有图片的名称
    pres = get_image_name(folders)  # 获取预测集所有图片的名称
    repeat = len([p for p in pres if p in trains])  # 获取重复数量
    start_verifies(folders)  # 开启预测
    logging.info('预测前确认待预测图片与训练样本的重复情况，'
                 '待预测图片%s张，训练样本%s张，重复数量为%s张' % (len(pres), len(trains), repeat))
