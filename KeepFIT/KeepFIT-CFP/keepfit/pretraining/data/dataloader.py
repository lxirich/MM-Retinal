"""
Dataset and Dataloader主函数   Dataset and Dataloader main function
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
    datasets为列表 list
    banned_categories为列表 list
    """

    # 数据变换（预处理）操作集合  Set of data transformation (preprocessing) operations
    transforms = Compose([
        CopyDict(),                                             # 深拷贝数据  数据以字典形式存储  Deep copy data. Data is stored in dictionary
        LoadImage(),                                            # 读取图片、预处理、图片张量存入字典  Read pictures, preprocess, and store picture tensors into dictionaries
        # ImageScaling(),                                       # 图像大小变换为模型输入大小【宽高比不变】  Image resize to model input size [unchanged aspect ratio]
        # LoadTensor(),
        ProduceDescription(caption=caption),                    # 生成原始text  Generate raw text
        AugmentDescription(augment=augment_description),        # 将类名换成医学描述  change the class name into medical description
        SelectRelevantKeys()                                    # 返回训练用到的数据  Returns the data used for training
    ])
    if knowledge_dict:
        # KD_transforms = Compose([CopyDict(), LoadImage()])      # MM数据  MM dataset
        KD_transforms = Compose([CopyDict(), LoadImage(), ProduceDescription(caption=caption),
                                AugmentDescription(augment=False),SelectRelevantKeys()])    # 百度数据  baidu dataset


    # 集成所有数据集  assembly all the dataset
    print("Setting assembly data...")
    data = []
    for iDataset in datasets:
        print("Processing data: " + iDataset)

        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv")

        # 只获取50%的数据  50% data
        # id_list = list(range(len(dataframe)))
        # selected_id_list = random.sample(id_list, len(id_list)//2)      # 50%数据  50% data
        selected_id_list = range(len(dataframe))                      # 100%数据   100% data

        for i in selected_id_list:
            data_i = dataframe.loc[i, :].to_dict()              # 将每行变为字典  image,attributes,categories   Turn each line into a dictionary
            data_i["categories"] = eval(data_i["categories"])
            data_i["atributes"] = eval(data_i["atributes"])

            # 去除被ban的类别   Removes the banned category
            banned = False
            if banned_categories is not None:
                for iCat in data_i["categories"]:
                    if iCat in banned_categories:
                        banned = True
            if banned:
                continue

            # 没被ban的 加入总数据集   The total data set that is not banned
            data_i["image_name"] = data_i["image"]
            data_i["image_path"] = data_root_path + data_i["image"]
            data.append(data_i)
    print('Total assembly data samples: {}'.format(len(data)))
    # 领域知识图像文本对数据   Domain knowledge image text pair data
    if knowledge_dict:
        # MM数据集  MM dataset
        # data_KD = []
        # dataframe_KD = pd.read_csv("./local_data/dataframes/pretraining/39_MM_Retinal_dataset.csv")
        # for i in range(len(dataframe_KD)):
        #     sample_df = dataframe_KD.loc[i, :].to_dict()
        #     data_i = {"image_path": data_root_path + sample_df["image"]}
        #     data_i["caption"] = sample_df["caption"]
        #     data_KD.append(data_i)

        # 百度数据集    baidu dataset
        print("process baidu...")
        data_KD = []
        dataframe_KD = pd.read_csv("/mnt/data/")
        for i in range(len(dataframe_KD)):
            sample_df = dataframe_KD.loc[i, :].to_dict()
            sample_df["categories"] = eval(sample_df["categories"])  
            sample_df["atributes"] = eval(sample_df["atributes"])

            banned = False
            if banned_categories is not None:
                for iCat in sample_df["categories"]:
                    if iCat in banned_categories:
                        banned = True
            if banned:
                continue

            sample_df["image_name"] = sample_df["image"]
            sample_df["image_path"] = "/mnt/data/jlzhang/Dataset/Resized/" + sample_df["image"]
            data_KD.append(sample_df)


    # 训练数据   train set
    if balance:
        train_dataset = UniformDataset(data=data, transform=transforms)
    else:
        train_dataset = Dataset(data=data, transform=transforms)
    train_sampler = DistributedSampler(train_dataset)                   # 分布式训练  distributed training
    # 知识字典   knowledge dictionary
    KD_loader = None
    if knowledge_dict:
        KD_dataset = Dataset(data=data_KD, transform=KD_transforms)


    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)         # 单卡  1 gpu
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)  # 多卡  mutil gpus
    if knowledge_dict:
        # 将KD/MM和训练数据一起训练 需要对数据抽取机制进行调整    【百度也用这个】   Training KD/MM with training data requires adjustments to the data extraction
        weights = torch.ones(len(KD_dataset))  # 均匀分布  uniform distribution
        weightedRandomSampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=batch_size * len(train_loader))    # 采样出来的数量和训练集一致  The amount sampled is consistent with the training set
        KD_loader = DataLoader(KD_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weightedRandomSampler)

        # weights = torch.ones(len(train_dataset))
        # weightedRandomSampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=batch_size * len(train_loader))
        # KD_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weightedRandomSampler)


        # KD_loader = DataLoader(KD_dataset, batch_size=batch_size, shuffle = False, num_workers=num_workers, drop_last=False)

    dataloaders = {"train": train_loader, "KD":KD_loader}
    return dataloaders

