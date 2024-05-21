import time
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
from torchvision.transforms import Resize
from local_data.constants import *


def Transform(input_path, output_path):
    # LoadImage
    img = np.array(Image.open(input_path), dtype=float)
    if np.max(img) > 1:
        img /= 255
    if len(img.shape) > 2:
        img = np.transpose(img, (2, 0, 1))
    else:
        img = np.expand_dims(img, 0)
    if img.shape[0] > 3:
        img = img[1:, :, :]
    if img.shape[0] < 3:
        img = np.repeat(img, 3, axis=0)
    
    # ImageScaling
    canvas=True
    size=(512, 512)
    transforms = torch.nn.Sequential(Resize(size, antialias=True),)
    
    img = torch.tensor(img)
    # 直接缩放 Direct scaling
    if not canvas or (img.shape[-1] == img.shape[-2]):
        img = transforms(img)
    # 长宽比不变 其余填充纯色 not change aspect ratio fill the rest with black
    else:
        sizes = img.shape[-2:]                    
        max_size = max(sizes)
        scale = max_size/size[0]
        # 缩放到指定尺寸 Scale to the specified size
        # 返回一个可调用对象/方法，其参数为img  Returns a callable object/method with parameter: img
        img = Resize((int(img.shape[-2]/scale), int(img.shape[-1]/scale)), antialias=True)(img)
        img = torch.nn.functional.pad(img, (0, size[0] - img.shape[-1], 0, size[1] - img.shape[-2], 0, 0))

    while True:
        try:
            # 转换为图片再进行保存 Convert to image and save
            img = img.permute(1, 2, 0)
            img = img * 255
            img = img.byte()
            image = Image.fromarray(img.numpy())
            image.save(output_path)
            break
        except OSError or IOError as e:
            print(f"**Error is {e}, **input path is {input_path}, **output_path is {output_path}")
            time.sleep(1)

    
def Task(csv_file):
    for file in csv_file:
        dataframe = pd.read_csv(file)
        for i, item in dataframe.iterrows():
            input_path = PATH_DATASETS + item['image']
            output_path = PATH_RESIZED_DATASETS + item['image']             # jpg & png
            # output_path = os.path.splitext(output_path)[0] + '.pt'
            folder_path = os.path.dirname(output_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            Transform(input_path, output_path)
            print(output_path,end = '\r', flush=True)

def divide_list(lst, n):
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


Task(["./local_data/dataframes/pretraining/39_MM_Retinal_dataset.csv"])