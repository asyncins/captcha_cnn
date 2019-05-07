import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from convert import one_hot_encode
from settings import *


class ImageDataSet(Dataset):
    """ 图片加载和处理 """

    def __init__(self, folder):
        self.transform = transforms.Compose([
            # 图片灰度处理
            transforms.Grayscale(),
            # 把一个取值范围是[0,255]的PIL.Image对象转换成取值范围是[0,1.0]的Tensor对象
            transforms.ToTensor()
        ])
        self.folder = folder
        # 从传入的文件夹路径中载入指定后缀为IMAGE_TYPE值的文件
        self.images = list(pathlib.Path(folder).glob('*.{}'.format(IMAGE_TYPE)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = self.transform(Image.open(image_path))
        # 获取独热码和字符位置列表
        vector, order = one_hot_encode(image_path.stem)
        label = torch.from_numpy(vector)
        return image, label, order


def loaders(folder: str, size: int) -> object:
    # 包装数据和目标张量的数据集
    objects = ImageDataSet(folder)
    return DataLoader(objects, batch_size=size, shuffle=True)
