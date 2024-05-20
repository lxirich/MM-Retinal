"""
自定义dataset
"""

import collections.abc
import numpy as np

from torch.utils.data import Dataset as _TorchDataset
from typing import Any, Callable, Optional, Sequence, Union
from torch.utils.data import Subset


# 根据索引返回数据/子集
class Dataset(_TorchDataset):
    """
    __len__返回数据集长度
    __getitem__ 返回数据/数据子集
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        # data是字典的列表（Sequence）      transform是一系列数据预处理变换（Callable 方法），是可选的 默认为None
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        # 获取指定索引的样本，并进行数据处理
        data_i = self.data[index]

        # 输入字典有很多键值对   输出字典只有：图像张量 文本 原始类别
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        # 给索引返回对应的样本（数据+标签）
        # 如果索引是一个列表/列表切片 返回一个pytorch Subset
        if isinstance(index, slice):
            # 切片
            start, stop, step = index.indices(len(self))    # 获取切片的开始 结束 步长
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)    # self原始数据集对象

        if isinstance(index, collections.abc.Sequence):
            # 列表
            return Subset(dataset=self, indices=index)

        return self._transform(index)                       # 标量


class UniformDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data=data, transform=transform)
        self.datasetkey = []        # 数据集名称
        self.data_dic = []          # 数据集名称：此数据集的 数据字典列表
        self.datasetnum = []        # 每个数据集样本个数列表
        self.datasetlen = 0         # 数据集个数
        self.dataset_split(data)

    def dataset_split(self, data):
        keys = []
        for img in data:
            keys.append(img["image_name"].split("/")[0])    # 数据集名称（有重复）
        self.datasetkey = list(np.unique(keys))

        data_dic = {}                                       # 数据集名称：此数据集的 数据字典列表
        for iKey in self.datasetkey:
            data_dic[iKey] = [data[iSample] for iSample in range(len(keys)) if keys[iSample]==iKey]
        self.data_dic = data_dic

        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the data {key} has no data'
            self.datasetnum.append(len(item))               # 每个数据集样本个数

        self.datasetlen = len(self.datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]         # 字典列表=》数据字典
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index):
        # 输入的index只是完成接口任务 不会返回对应index的数据
        set_index = index % self.datasetlen                 # 数据集index
        set_key = self.datasetkey[set_index]                # 选出的数据集名称

        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]       # 随机出这个数据集的一个索引
        return self._transform(set_key, data_index)