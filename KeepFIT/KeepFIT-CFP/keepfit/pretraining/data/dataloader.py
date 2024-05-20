"""
Dataset and Dataloader主函数
"""
import random
import pandas as pd
import torch

from torchvision.transforms import Compose
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from keepfit.pretraining.data.dataset import Dataset, UniformDataset
from keepfit.pretraining.data.transforms import LoadImage, ImageScaling, SelectRelevantKeys, CopyDict,\
    ProduceDescription, AugmentDescription, LoadTensor


def get_loader(dataframes_path, data_root_path, datasets, balance=False, batch_size=8, num_workers=0,
               banned_categories=None, caption="A fundus photograph of [CLS]", augment_description=True,
               knowledge_dict=False):
    """
    Dataloaders创建
    集成所有数据集为统一数据格式

    datasets为列表
    banned_categories为列表
    """

    # Torch vision数据变换（预处理）操作集合   对应的方法来自\pretraining\data\transforms.py
    transforms = Compose([
        CopyDict(),                                             # 深拷贝数据  数据以字典形式存储
        LoadImage(),                                            # 读取图片、预处理、图片张量存入字典
        # ImageScaling(),                                       # 图像大小变换为模型输入大小【宽高比不变】
        # LoadTensor(),
        ProduceDescription(caption=caption),                    # 生成原始text
        AugmentDescription(augment=augment_description),        # 将类名换成医学描述
        SelectRelevantKeys()                                    # 返回训练用到的数据
    ])
    if knowledge_dict:
        # KD_transforms = Compose([CopyDict(), LoadImage()])      # MM数据
        KD_transforms = Compose([CopyDict(), LoadImage(), ProduceDescription(caption=caption),
                                AugmentDescription(augment=False),SelectRelevantKeys()])    # 百度数据


    # 集成所有数据集
    print("Setting assembly data...")
    data = []
    for iDataset in datasets:
        print("Processing data: " + iDataset)

        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv")

        # 只获取50%的数据
        # id_list = list(range(len(dataframe)))
        # selected_id_list = random.sample(id_list, len(id_list)//2)      # 50%数据
        selected_id_list = range(len(dataframe))                      # 100%数据

        for i in selected_id_list:
            data_i = dataframe.loc[i, :].to_dict()              # 将每行变为字典  image,attributes,categories
            data_i["categories"] = eval(data_i["categories"])
            data_i["atributes"] = eval(data_i["atributes"])

            # 去除被ban的类别
            banned = False
            if banned_categories is not None:
                for iCat in data_i["categories"]:
                    if iCat in banned_categories:
                        banned = True
            if banned:
                continue

            # 没被ban的 加入总数据集
            data_i["image_name"] = data_i["image"]
            data_i["image_path"] = data_root_path + data_i["image"]
            data.append(data_i)
    print('Total assembly data samples: {}'.format(len(data)))
    # 领域知识图像文本对数据
    if knowledge_dict:
        # MM数据集
        # data_KD = []
        # dataframe_KD = pd.read_csv("./local_data/dataframes/pretraining/39_MM_Retinal_dataset.csv")
        # for i in range(len(dataframe_KD)):
        #     sample_df = dataframe_KD.loc[i, :].to_dict()
        #     data_i = {"image_path": data_root_path + sample_df["image"]}
        #     data_i["caption"] = sample_df["caption"]
        #     data_KD.append(data_i)

        # 百度数据集
        print("process baidu...")
        data_KD = []
        dataframe_KD = pd.read_csv("/mnt/data/")
        for i in range(len(dataframe_KD)):
            sample_df = dataframe_KD.loc[i, :].to_dict()
            sample_df["categories"] = eval(sample_df["categories"])  
            sample_df["atributes"] = eval(sample_df["atributes"])

            # 去除被ban的类别
            banned = False
            if banned_categories is not None:
                for iCat in sample_df["categories"]:  # 多标签数据集
                    if iCat in banned_categories:
                        banned = True
            if banned:
                continue

            # 没被ban的 加入总数据集
            sample_df["image_name"] = sample_df["image"]
            sample_df["image_path"] = "/mnt/data/jlzhang/Dataset/Resized/" + sample_df["image"]
            data_KD.append(sample_df)


    # 训练数据
    if balance:
        train_dataset = UniformDataset(data=data, transform=transforms)
    else:
        train_dataset = Dataset(data=data, transform=transforms)        # 自定义pytorch dataset
    train_sampler = DistributedSampler(train_dataset)                   # 分布式训练
    # 知识字典
    KD_loader = None
    if knowledge_dict:
        KD_dataset = Dataset(data=data_KD, transform=KD_transforms)


    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)         # 单卡
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)  # 多卡
    if knowledge_dict:
        # 将KD和训练数据一起训练 需要对数据抽取机制进行调整    【百度也用这个】
        weights = torch.ones(len(KD_dataset))  # 均匀分布
        weightedRandomSampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=batch_size * len(train_loader))    # 采样出来的数量和训练集一致
        KD_loader = DataLoader(KD_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weightedRandomSampler)  # KD一起训练 或只参考部分

        # 使用flair数据集填充KD
        # weights = torch.ones(len(train_dataset))  # 均匀分布
        # weightedRandomSampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=batch_size * len(train_loader))  # 采样出来的数量和训练集一致
        # KD_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weightedRandomSampler)


        # KD_loader = DataLoader(KD_dataset, batch_size=batch_size, shuffle = False, num_workers=num_workers, drop_last=False)    # KD仅查询

    dataloaders = {"train": train_loader, "KD":KD_loader}
    return dataloaders

