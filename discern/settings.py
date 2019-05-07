import pathlib
from os import path


# 路径
PARENT_LAYER = pathlib.Path.cwd().parent

# 数字与大写字母混合
NUMBER = [str(_) for _ in range(0, 10)]
LETTER = [chr(_).upper() for _ in range(97, 123)]
CHARACTER = {v: k for k, v in enumerate(NUMBER + LETTER)}

# 图片路径
PATH_IMAGE = path.join(PARENT_LAYER, 'images')
PATH_TRAIN = path.join(PATH_IMAGE, 'train')
PATH_TEST = path.join(PATH_IMAGE, 'test')
PATH_PREDICT = path.join(PATH_IMAGE, 'predict')

# 图片规格
CAPTCHA_NUMBER = 6
IMAGE_HEIGHT = 40
IMAGE_WIDTH = 200
IMAGE_TYPE = 'png'

# 训练参数
EPOCHS = 15
BATCH_SIZE = 32
RATE = 0.001
MODEL_NAME = 'result.pkl'

