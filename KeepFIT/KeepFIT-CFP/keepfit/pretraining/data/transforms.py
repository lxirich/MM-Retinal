"""
数据预处理 数据增强  Data preprocessing data augmentation
"""

import numpy as np
import random
import torch
import copy
import os

from PIL import Image, ImageFile
from torchvision.transforms import Resize
from keepfit.modeling.dictionary import definitions
from kornia.augmentation import RandomHorizontalFlip, RandomAffine, ColorJitter

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 训练时的数据增强操作  Data augmentation operations during training
augmentations_pretraining = torch.nn.Sequential(RandomHorizontalFlip(p=0.5),
                                                RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1)),      # 线性变换  linear transformation
                                                ColorJitter(p=0.25, brightness=0.2, contrast=0.2))          # HSL变换  HSL


# 适配已经经过预处理和Resize图片  Adapt images that have been preprocessed and resized
class LoadTensor():
    def __init__(self, target='image_path'):
        self.target = target

    def __call__(self, data):
        file_path = os.path.splitext(data[self.target])[0] + '.pt'
        img = torch.load(file_path)
        data[self.target.replace("_path", "")] = img
        return data


# 读取pair中的图片、归一化、改变维度顺序、增加字典元素（img纯数据）
# Read images, normalize, change dimension order, add dictionary elements (img pure data)
class LoadImage():
    def __init__(self, target="image_path"):
        self.target = target,

    def __call__(self, data):
        img = np.array(Image.open(data[self.target]), dtype=float)

        # 归一化  normalization
        if np.max(img) > 1:
            img /= 255

        # To C * W * H
        if len(img.shape) > 2:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)

        if img.shape[0] > 3:
            img = img[1:, :, :]

        if "image" in self.target:
            if img.shape[0] < 3:
                img = np.repeat(img, 3, axis=0)

        data[self.target.replace("_path", "")] = img
        return data


# 图像大小变换  Image resize
class ImageScaling():
    def __init__(self, size=(512, 512), canvas=True, target="image"):
        self.size = size
        self.canvas = canvas                            # 是否需要防止长宽失真  prevent length and width distortion?
        self.target = target

        self.transforms = torch.nn.Sequential(
            Resize(self.size, antialias=True),
        )

    def __call__(self, data):
        img = torch.tensor(data[self.target])

        if not self.canvas or (img.shape[-1] == img.shape[-2]):
            img = self.transforms(img)
        # 长宽比不变 其余填充纯色  The aspect ratio does not change   Fill the rest with black
        else:
            sizes = img.shape[-2:]                      # 提取长宽维度  Extract the length and width dimensions
            max_size = max(sizes)
            scale = max_size/self.size[0]
            img = Resize((int(img.shape[-2]/scale), int(img.shape[-1]/scale)), antialias=True)(img)
            img = torch.nn.functional.pad(img, (0, self.size[0] - img.shape[-1], 0, self.size[1] - img.shape[-2], 0, 0))

        data[self.target] = img
        return data


# 将类标变为原始text   提示模板：A [ATR] fundus photograph of [CLS]    Change the class label to the original text
class ProduceDescription():
    def __init__(self, caption):
        self.caption = caption

    def __call__(self, data):
        atr_sample = random.sample(data['atributes'], 1)[0] if len(data['atributes']) > 0 else ""
        cat_sample = random.sample(data['categories'], 1)[0] if len(data['categories']) > 0 else ""

        data["sel_category"] = cat_sample
        data["report"] = [self.caption.replace("[ATR]",  atr_sample).replace("[CLS]",  cat_sample).replace("  ", " ")]
        return data


# 将类名换成医学描述  Replace the class name with a medical description
class AugmentDescription():
    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, data):
        if self.augment:
            # 对本身不是text监督的数据进行操作  Operate on data that is not supervised by text
            if data["image_name"].split("/")[0] not in ["06_EYENET", "11_STARE", "08_ODIR-5K", "31_JICHI"]:
                if data["sel_category"] in list(definitions.keys()):
                    prompts = [data["sel_category"]] + definitions[data["sel_category"]]    # 原始类别名称+专业词汇  list
                    new_cat = random.sample(prompts, 1)[0]
                    data["report"][0] = data["report"][0].replace(data["sel_category"], new_cat)
                    data["augmented_category"] = new_cat

        return data


# 深拷贝数据pair   Deep copy data pair
class CopyDict():
    def __call__(self, data):
        d = copy.deepcopy(data)
        return d


# 返回训练用到的数据  Returns the data used for training
class SelectRelevantKeys():
    def __init__(self, target_keys=None):
        if target_keys is None:
            target_keys = ['image', 'report', 'sel_category']       # 图像张量 文本 原始类别（用于标签生成）  Image tensor; text raw; category (for label generation)
        self.target_keys = target_keys

    def __call__(self, data):
        d = {key: data[key] for key in self.target_keys}
        return d