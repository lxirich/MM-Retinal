"""
自定义dataset
define dataset
"""

import collections.abc
import numpy as np

from torch.utils.data import Dataset as _TorchDataset
from typing import Any, Callable, Optional, Sequence, Union
from torch.utils.data import Subset


# 根据索引返回数据/子集  Returns data/subsets according to index
class Dataset(_TorchDataset):
    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        # data：是字典的列表   List of dictionaries
        # transform：是一系列数据预处理变换  A series of data preprocessing transformations
        self.data = data
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        # 获取指定索引的样本，并进行数据处理  Gets a sample of the specified index and performs data processing
        data_i = self.data[index]

        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        # 给索引返回对应的样本（数据+标签）  Return the corresponding sample to the index (data + label)
        # 如果索引是一个列表/列表切片 返回一个pytorch Subset   Return a pytorch Subset if the index is a list/list slice
        if isinstance(index, slice):
            # 切片  slice
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)

        if isinstance(index, collections.abc.Sequence):
            # 列表 list
            return Subset(dataset=self, indices=index)

        return self._transform(index)


class UniformDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__(data=data, transform=transform)
        self.datasetkey = []        # 数据集名称  Data set name
        self.data_dic = []
        self.datasetnum = []        # 每个数据集样本个数列表  List of the number of samples per dataset
        self.datasetlen = 0         # 数据集个数  Number of data sets
        self.dataset_split(data)

    def dataset_split(self, data):
        keys = []
        for img in data:
            keys.append(img["image_name"].split("/")[0])    # 数据集名称（有重复）  Data set name (duplicate)
        self.datasetkey = list(np.unique(keys))

        data_dic = {}
        for iKey in self.datasetkey:
            data_dic[iKey] = [data[iSample] for iSample in range(len(keys)) if keys[iSample]==iKey]
        self.data_dic = data_dic

        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the data {key} has no data'
            self.datasetnum.append(len(item))               # 每个数据集样本个数  Number of samples per dataset

        self.datasetlen = len(self.datasetkey)

    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return self.transform(data_i) if self.transform is not None else data_i

    def __getitem__(self, index):
        # 输入的index只是完成接口任务 不会返回对应index的数据
        # The entered index completes the interface task and does not return the corresponding index data
        set_index = index % self.datasetlen                 # 数据集index   dataset index
        set_key = self.datasetkey[set_index]                # 选出的数据集名称  data set name

        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]       # 随机出这个数据集的一个索引  randomly select an index of the data set
        return self._transform(set_key, data_index)