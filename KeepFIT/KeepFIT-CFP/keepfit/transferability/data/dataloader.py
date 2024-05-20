"""
迁移部分    数据变换（预处理）；分割训练、验证、测试集；创建dataloader；平衡数据集类别
"""

import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from keepfit.pretraining.data.dataset import Dataset
from keepfit.pretraining.data.transforms import LoadImage, ImageScaling, CopyDict


# 数据变换（预处理）；分割训练、验证、测试集；创建dataloader
def get_dataloader_splits(dataframe_path, data_root_path, targets_dict, shots_train="80%", shots_val="0%",
                          shots_test="20%", balance=False, batch_size=8, num_workers=0, seed=0, task="classification",
                          size=(512, 512), resize_canvas=False, batch_size_test=1, knowledge_dict=False):
    '''
    dataframe_path 各个数据集（统一格式）路径        data_root_path 原始数据路径         targets_dict 类别名称
    seed=第K折实验 用于随机数生成
    '''

    # 数据变换（预处理）操作集合
    if task == "classification":
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling(size=size)])
    elif task == "segmentation":
        transforms = Compose([CopyDict(), LoadImage(target="image_path"), LoadImage(target="mask_path"),
                              ImageScaling(size=size, target="image", canvas=resize_canvas),
                              ImageScaling(size=size, target="mask", canvas=resize_canvas)])
    else:
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling()])

    # 读取数据字典中有用的部分（数据&标签）    输出：字典列表
    # TAOP数据训练部分
    data = []
    dataframe = pd.read_csv(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()                                      # 将每行变为字典  image,attributes,categories

        data_i = {"image_path": data_root_path + sample_df["image"]}                   # 构造需要的数据    路径
        if task == "classification":
            data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]           # 类别名称=》编号  所有多类转成1类
        if task == "segmentation":
            data_i["mask_path"] = data_root_path + sample_df["mask"]
            data_i["label"] = 1
        data.append(data_i)

    # TAOP数据测试部分================================
    # 原始数据集的测试集
    # data_test = []
    # dataframe = pd.read_csv(dataframe_path.replace("train", "test"))
    # for i in range(len(dataframe)):
    #     sample_df = dataframe.loc[i, :].to_dict()  # 将每行变为字典  image,attributes,categories
    #
    #     data_i = {"image_path": data_root_path + sample_df["image"]}  # 构造需要的数据    路径
    #     if task == "classification":
    #         data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]  # 类别名称=》编号  所有多类转成1类
    #     if task == "segmentation":
    #         data_i["mask_path"] = data_root_path + sample_df["mask"]
    #         data_i["label"] = 1
    #     data_test.append(data_i)

    # 手工打乱数据 每折顺序不一样
    random.seed(seed)
    random.shuffle(data)

    # 领域知识图像文本对数据
    if knowledge_dict:
        data_KD = []
        dataframe_KD = pd.read_csv("./local_data/dataframes/pretraining/39_MM_Retinal_dataset.csv")
        for i in range(len(dataframe_KD)):
            sample_df = dataframe_KD.loc[i, :].to_dict()
            data_i = {"image_path": data_root_path + sample_df["image"]}
            data_i["caption"] = sample_df["caption"]
            data_KD.append(data_i)

    # 分割训练、验证、测试集       希望每个集合的类别分布一致
    labels = [data_i["label"] for data_i in data]                                       # 数据集标签列表
    unique_labels = np.unique(labels)
    data_train, data_val, data_test = [], [], []
    # 遍历每个类别
    for iLabel in unique_labels:
        idx = list(np.squeeze(np.argwhere(labels == iLabel)))                           # 获取数据集中该类的样本序号

        # 各个集合该类的样本个数
        train_samples = get_shots(shots_train, len(idx))
        val_samples = get_shots(shots_val, len(idx))
        test_samples = get_shots(shots_test, len(idx))

        [data_test.append(data[iidx]) for iidx in idx[:test_samples]]
        [data_train.append(data[iidx]) for iidx in idx[test_samples:test_samples+train_samples]]
        [data_val.append(data[iidx]) for iidx in idx[test_samples+train_samples:test_samples+train_samples+val_samples]]

    if balance:                                                                         
        data_train = balance_data(data_train)

    train_loader = get_loader(data_train, transforms, "train", batch_size, num_workers)
    val_loader = get_loader(data_val, transforms, "val", batch_size_test, num_workers)
    test_loader = get_loader(data_test, transforms, "test", batch_size_test, num_workers)

    # TAOP===========================================
    # train_loader = get_loader(data, transforms, "train", batch_size, num_workers)
    # val_loader = None
    # test_loader = get_loader(data_test, transforms, "test", batch_size_test, num_workers)

    KD_loader = None
    if knowledge_dict:
        KD_loader = get_loader(data_KD, transforms, "KD", batch_size, num_workers)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader, "KD":KD_loader}
    return loaders


# pytorch dataset + dataloader操作
def get_loader(data, transforms, split, batch_size, num_workers):
    '''
    data：多个元素的字典        transforms：只有对图像的处理
    输出：具有冗余信息的数据字典 迁移阶段数据量小
    '''

    if len(data) == 0:
        loader = None
    else:
        dataset = Dataset(data=data, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle = split == "train", num_workers=num_workers, drop_last=False)
    return loader


# 平衡训练集中不同类别的分布情况
def balance_data(data):
    labels = [iSample["label"] for iSample in data]                     # 所有样本的label
    unique_labels = np.unique(labels)                                   
    counts = np.bincount(labels)                                        # 统计每个类别的个数
    N_max = np.max(counts)

    data_out = []
    for iLabel in unique_labels:
        idx = list(np.argwhere(np.array(labels) == iLabel)[:, 0])       # 获取数据集中该类的样本序号

        # 如果当前类别在训练集中较少 随机选一些样本（该类）进行重复，使得所有类别的数量一样
        if N_max-counts[iLabel] > 0:
            idx += random.choices(idx, k=N_max-counts[iLabel])
        [data_out.append(data[iidx]) for iidx in idx]

    return data_out


# 返回样本个数
def get_shots(shots_str, N):
    # 输入百分比
    if "%" in str(shots_str):
        shots_int = int(int(shots_str[:-1]) / 100 * N)
    # 直接输入个数
    else:
        shots_int = int(shots_str)
    return shots_int
