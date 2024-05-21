"""
迁移实验主函数
"""

import argparse
import torch

from keepfit.modeling.model import KeepFITModel
from keepfit.transferability.data.dataloader import get_dataloader_splits
from keepfit.utils.metrics import evaluate, average_folds_results, save_results
from keepfit.modeling.misc import set_seeds
from keepfit.transferability.modeling.adapters import LinearProbe, ClipAdapter, ZeroShot, TipAdapter, KDAdapter

from local_data.constants import *
from local_data.experiments import get_experiment_setting

import warnings
warnings.filterwarnings("ignore")

set_seeds(42, use_cuda=torch.cuda.is_available())


# 下游任务适配器初始化  initialize downstream task adapter
def init_adapter(model, args):
    # linear probe
    if args.method == "lp":
        print("Transferability by Linear Probing...", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)
    # CLIP适配器  clip adapter
    elif args.method == "clipAdapter":
        print("Transferability by CLIP Adapter...", end="\n")
        adapter = ClipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge)
    # TipAdapter
    elif args.method == "tipAdapter":
        print("Transferability by TIP-Adapter Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge, train=False)
    # TipAdapter-f
    elif args.method == "tipAdapter-f":
        print("Transferability by TIP-Adapter-f Adapter...", end="\n")
        adapter = TipAdapter(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge, train=True)
    # ZS  zero shot
    elif args.method == "zero_shot":
        print("Zero-shot classification...", end="\n")
        adapter = ZeroShot(model, args.setting["targets"], tta=args.tta, fta=args.fta, domain_knowledge=args.domain_knowledge)
    # 默认采用lp  By default Linear probe
    else:
        print("Adapter not implemented... using LP", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)

    return adapter


# 获取实验名称  Get experiment name
def generate_experiment_id(args):
    id = args.experiment + '_名称_' + args.architecture + '_方法_' + args.method +\
         '_shots_train_' + args.shots_train + '_shots_test_' + args.shots_test + \
         '_平衡_' + str(args.balance) + '_domain knowledge_' + str(args.domain_knowledge) + \
         '_proj_' + str(args.project_features)
    return id


def process(args):
    # metrics_test：测试集评价指标列表  Test set evaluation metric list
    # metrics_external：额外数据集评价指标列表，列表元素是n个列表  Additional data set evaluation metric list
    # weights：适配器权重列表  Adapter weight list
    args.metrics_test, args.metrics_external, args.weights = [], [[] for i in range(len(args.experiment_test))], []     # experiment_test：数据集名称列表  List of data set names

    experiment_id = generate_experiment_id(args)
    print(experiment_id)

    # K折交叉验证  K-fold cross-validation
    for iFold in range(args.folds):
        print("\nTransferability (fold : " + str(iFold + 1) + ")", end="\n")
        args.iFold = iFold

        # 数据  data
        args.setting = get_experiment_setting(args.experiment)                                                          # 获取实验配置 路径、迁移任务、类别  Gets the experiment configuration: path, task, category
        args.loaders = get_dataloader_splits(args.setting["dataframe"], args.data_root_path, args.setting["targets"],
                                             shots_train=args.shots_train, shots_val=args.shots_val,
                                             shots_test=args.shots_test, balance=args.balance,
                                             batch_size=args.batch_size, num_workers=args.num_workers, seed=iFold,
                                             task=args.setting["task"], size=args.size,
                                             resize_canvas=args.resize_canvas, batch_size_test=args.batch_size_test,
                                             knowledge_dict= args.knowledge_dict)                                       # 数据变换（预处理）；分割训练、验证、测试集；创建dataloader
                                                                                                                        # Data transformation (preprocessing); Split training, valid and test sets; Create a dataloader

        # 模型  model
        model = KeepFITModel(from_checkpoint=args.load_weights, weights_path=args.weights_path,
                      projection=args.project_features, norm_features=args.norm_features,
                      vision_pretrained=args.init_imagenet)
        adapter = init_adapter(model, args)                                                                             # 初始化迁移适配器  Initialize adapter

        # 适配器训练  training
        adapter.fit(args.loaders)

        # 预测阶段  prediction
        if args.loaders["test"] is not None:
            refs, preds = adapter.predict(args.loaders["test"])
            metrics_fold = evaluate(refs, preds, args.setting["task"])
            args.metrics_test.append(metrics_fold)

        # 输出，保存适配器权重  Output, save adapter weight
        args.weights.append(adapter.model.state_dict())

        # OOD实验【一个数据集训练 用于另一个数据集】  OOD Experiment Training on one data set and test for another Dataset
        # experiment_test：输入的数据   input dataset    ZS模式
        if args.experiment_test[0] != "":
            # 遍历数据集  Traverse the dataset
            for i_external in range(len(args.experiment_test)):
                print("External testing: " + args.experiment_test[i_external])

                # 数据  data
                setting_external = get_experiment_setting(args.experiment_test[i_external])
                loaders_external = get_dataloader_splits(setting_external["dataframe"], args.data_root_path,
                                                         args.setting["targets"], shots_train="0%", shots_val="0%",
                                                         shots_test="100%", balance=False,
                                                         batch_size=args.batch_size_test,
                                                         batch_size_test=args.batch_size_test,
                                                         num_workers=args.num_workers, seed=iFold,
                                                         task=args.setting["task"], size=args.size,
                                                         resize_canvas=args.resize_canvas)
                # 测试数据预测 评估  Test data evaluation
                refs, preds = adapter.predict(loaders_external["test"])
                metrics = evaluate(refs, preds, args.setting["task"])
                args.metrics_external[i_external].append(metrics)

    # 常规实验K折结果平均  average K-fold
    if args.loaders["test"] is not None:
        print("\nTransferability (cross-validation)", end="\n")
        args.metrics = average_folds_results(args.metrics_test, args.setting["task"])
    else:
        args.metrics = None

    # 保存评估指标 适配器权重  Save the evaluation metric and adapter weight
    save_results(args.metrics, args.out_path, id_experiment=generate_experiment_id(args),
                 id_metrics="metrics", save_model=args.save_model, weights=args.weights)

    # OOD实验 K折交叉验证平均    ood experiment  average K-fold
    if args.experiment_test[0] != "":
        for i_external in range(len(args.experiment_test)):
            print("External testing: " + args.experiment_test[i_external])
            metrics = average_folds_results(args.metrics_external[i_external], args.setting["task"])
            save_results(metrics, args.out_path, id_experiment=generate_experiment_id(args),
                         id_metrics=args.experiment_test[i_external], save_model=False)


def main():
    parser = argparse.ArgumentParser()

    # 数据相关 data
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')
    parser.add_argument('--experiment_description', default=None)
    parser.add_argument('--save_model', default=False, type=lambda x: (str(x).lower() == 'true'))       # 是否保存迁移权重  save the adaptation weight?
    parser.add_argument('--shots_train', default="80%", type=lambda x: (str(x)))                        # 用于训练的数据比例  proportion of data used for training
    parser.add_argument('--shots_val', default="0%", type=lambda x: (str(x)))                           # 验证集  data used for validation
    parser.add_argument('--shots_test', default="20%", type=lambda x: (str(x)))                         # 默认ZS  data used for testing
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))          # 是否平衡数据集  Balanced dataset?
    parser.add_argument('--folds', default=5, type=int)                                                 # K折交叉验证  K-fold cross validation
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_test', default=8, type=int)
    parser.add_argument('--size', default=(512, 512), help="(512, 512) | (2048, 4096) ")
    parser.add_argument('--resize_canvas', default=False, type=lambda x: (str(x).lower() == 'true'))    

    # 实验相关参数  Experimental parameter
    parser.add_argument('--experiment', default='AMD',
                        help='02_MESSIDOR - 13_FIVES - 25_REFUGE - 08_ODIR200x3 - 05_20x3 - AMD - TAOP')            # 实验使用的数据集  data set used in the experiment
    parser.add_argument('--experiment_test', default='',
                        help='02_MESSIDOR, 37_DeepDRiD_online_test',
                        type=lambda s: [item for item in s.split(',')])                                             # OOD实验
    parser.add_argument('--method', default='lp',
                        help='lp - tipAdapter - tipAdapter-f - clipAdapter'
                             'FT - FT_last - LP_FT -LP_FT_bn_last - FT_freeze_all'
                             'zero_shot -'
                             'KDAdapter - KDAdapter-f')                                                             # 模型迁移方式 ZS、FT、LP  adaptation method
    parser.add_argument('--num_workers', default=24, type=int, help='workers number for DataLoader')
    parser.add_argument('--test_from_folder', default=[], type=list)

    parser.add_argument('--epochs', default=50, type=int)                                               # 以下为FT的设置  FT setting
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--update_bn', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--freeze_classifier', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--last_lp', default=False, type=lambda x: (str(x).lower() == 'true'))          # FT中增加LP  FT with lp
    parser.add_argument('--save_best', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--patience', default=10, type=int)

    # 模型架构与权重  Model architecture and weight
    parser.add_argument('--weights_path', default='./results/pretraining/',
                        help='./keepfit/modeling/')                                                                  # 本地模型权重 默认采用下载  local weights, downloaded by default
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))      # 是否加载预训练权重 迁移True   load pre-trained weight?
    parser.add_argument('--init_imagenet', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--architecture', default='keepfit', help='resnet_v1 -- efficientnet')
    parser.add_argument('--project_features', default=False, type=lambda x: (str(x).lower() == 'true')) # 是否投影
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))     # 是否归一化特征
    parser.add_argument('--domain_knowledge', default=False, type=lambda x: (str(x).lower() == 'true')) # 是否迁移时使用领域知识  use domain knowledge when adaptation?
    parser.add_argument('--fta', default=False, type=lambda x: (str(x).lower() == 'true'))              # 训练时数据增强  train time data augmentation
    parser.add_argument('--tta', default=False, type=lambda x: (str(x).lower() == 'true'))              # 测试时数据增强  test time data augmentation
    parser.add_argument('--knowledge_dict', default=False, type=lambda x: (str(x).lower() == 'true'))   # 是否启用知识caption  use domain knowledge caption?
    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()
