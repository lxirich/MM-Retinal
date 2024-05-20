"""
下游任务适配器 初始化、训练、预测
Zero-shot, Linear Probe (LP), ClipAdapter, TipAdapter, TipAdapter-f
"""

import copy
import random
import torch
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from flair.pretraining.data.transforms import augmentations_pretraining


device = 'cuda:3' if torch.cuda.is_available() else 'cpu'


"""
纯视觉适配器 只使用图像编码器
"""
# 适配器的父类    初始化；实现训练接口；训练、预测虚函数；抽取视觉特征和标签
class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        '''
        model：网络编码器      targets：类别      tta：测试阶段数据增强     fta：训练适配器是否用数据增强扩充数据集（LP）
        '''
        self.model = copy.deepcopy(model)
        self.model.eval()                               # 冻结编码器参数
        self.num_targets = len(targets)                 # 类别个数
        self.tta = tta                                  # 测试增强策略 扩充样本 进行集成（投票/均值）
        self.fta = fta                                  # 是否训练增强策略 扩充训练数据
        self.number_augmentations = 20                  # 训练增强次数

    # 获取视觉特征和标签
    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()
        epoch_iterator = tqdm(data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

        # 对于后续的适配器 输入是CLIP视觉编码器得到的特征向量 输出是类别号
        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():
                if transforms is not None:
                    images = transforms(images)         # 有增强过一遍增强

                x = self.model.vision_model(images)     # 只使用图像编码器

            X.extend(x.cpu().detach().numpy())          # detach()从计算图分离 不算梯度
            Y.extend(batch["label"].numpy())            # 类别是序号

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # 训练的接口：进行数据增强、抽视觉特征、调用训练函数     最后得到的LP参数会加入FLAIR模型当中
    def fit(self, loaders, transforms=None):
        '''
        loaders：字典 包含训练集、验证集、测试集的pytorch dataloader
        transforms：数据增强 不使用
        '''
        # 读数据
        data_loader = loaders["train"]                                                         # 训练集  img path 标签/mask

        # 是否使用训练增强策略 增加训练数据
        if self.fta:
            transforms = augmentations_pretraining
        # 获取视觉特征
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)      # 一次增强返回一个列表
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)                                                       # 合并成一维
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)            # transforms为None

        # 训练
        self.train(X, Y)

    # 训练用虚函数
    def train(self, X, Y):
        """
        虚函数 由具体适配器实现
        """
        return

    # 预测用虚函数
    def predict(self, loader, transforms=None):
        """
        虚函数 由具体适配器实现
        """
        return


# LP纯逻辑回归适配器        逻辑回归训练；预测
class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        '''
        targets：类别      tta：测试阶段数据增强     fta：训练适配器是否用数据增强扩充数据集        c逻辑回归的正则化系数
        '''
        super().__init__(model, targets, tta=tta, fta=fta)
        self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0, class_weight="balanced")

    # 训练得到LP的参数 并加入模型当中
    def train(self, X, Y):
        '''
        X是图片经过图像编码器的特征   Y是类标（数字）
        '''
        self.classifier.fit(X, Y)

        # 将训练好的逻辑回归加入FLAIR模型
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)                               # LP就是一个线性层  对于2分类，逻辑回归参数是1维 会广播 但输出就是1维
        self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.classifier.coef_).to(torch.float32))
        self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.classifier.intercept_).to(torch.float32))
        self.model.classifier.to(device)

    # 预测阶段
    def predict(self, loader, transforms=None):
        '''
        loader：测试集的数据
        '''
        self.model.eval()
        # 测试时数据增强  对一个样本进行多次增强 喂到模型中并进行投票（均值）
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        with torch.no_grad():
            refs, preds = [], []                                            # refs是真实标签     preds是预测的标签
            for step, batch in enumerate(epoch_iterator):
                # 数据
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                # 前向过程 输出logits
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))               # 增加一个维度 bs * K * 1
                    score = torch.concat(preds_tta, -1).mean(-1)            # bs * K * number_augmentations==》bs * K     对logits计算均值
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)

                # 激活函数
                if score.shape[-1] == 1:                                    # 二分类情况  输出维度是1
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)     # 转为2维的情况      bs * 2
                else:                                                       # 多类分布情况
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
多模态适配器
"""
# 多模态适配器父类      继承适配器父类；增加文本特征的抽取
class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # 输入类别名称    输出对应类别的文本特征（有/无领域知识）
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()), domain_knowledge=domain_knowledge)


# ZS适配器     不需要训练，只有预测
class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    # 清空训练接口的操作
    def fit(self, loaders, transforms=None):
        """
        ZS没有训练，重写父类训练接口，清空所有操作      主函数都会运行fit
        """
        return

    # 预测阶段
    def predict(self, loader, transforms=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)                # 获取图像特征和标签
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()    # 计算相似度     X:N * embd      text_embeds：K * embd
                scores.append(score_i.unsqueeze(-1))                                                                # score_i.unsqueeze(-1)：N * K * 1
            score = torch.concat(scores, -1).mean(-1)                                                               # score就是logits     N * K
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        preds = torch.softmax(score, dim=-1)                                                                        # logits过softmax变为概率分布
        preds = preds.detach().cpu().numpy()
        return refs, preds


# CLIP适配器   视觉空间到文本空间的残差适配器；训练；预测
class ClipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim     # 特征输出维度
        self.reduction = 4                              # few shot适配器下降尺度
        self.ratio = 0.2                                # adapter采用了残差的结构   ratio * adapter(x) + (1-ratio)  * x

        # 视觉特征与文本特征
        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

    # 预测阶段
    def predict(self, loader, transforms=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    X = self.residual_adapter(X)        # 和ZS的唯一区别是多了一个视觉空间到文本空间的对齐
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                X = self.residual_adapter(X)
                score = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()

        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds

    # 适配器的训练
    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)                 # 标签

        # 训练
        epochs, lr, bs = 40, 0.001, 1       # bs=1（这个参数每用到）
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])
        indexes = np.arange(0, X.shape[0])  # X.shape[0]样本个数
        random.shuffle(indexes)

        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                # 比较简易 没用dataloader bs=1

                # 数据
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)                           # 1 * embd
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)                               # 1，

                # 前向过程
                X_batch = self.residual_adapter(X_batch)                                            # 视觉特征过适配器      残差、归一化
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)   # 计算相似度
                loss = torch.nn.functional.cross_entropy(logits, target)                            # bs=1 所以没用CLIP损失 而是直接用CE损失

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()                                                                    # 这边的学习率调整都是按照iteration来

                loss_epoch += loss.item()/X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")

    # 视觉空间到文本空间的残差适配器
    def residual_adapter(self, X):
        # 输入backbone的视觉特征
        X_res = self.adapter(X)
        X = self.ratio * X_res + (1 - self.ratio) * X                           # 残差结构的adapter
        X = X / X.norm(dim=-1, keepdim=True)                                    # 归一化特征
        return X


class TipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        '''
        train：是否训练
        '''
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        # 相似度计算用到的数据
        self.beta = 5                   # 超参数
        self.alpha = 1                  # 超参数
        self.cache_keys = []            # 存视觉特征向量
        self.cache_values = []          # 存标签 one-hot形式

        # 是否训练适配器
        self.train_tip = train
        self.adapter_layer = []         # 存放适配器

    # 预测阶段
    def predict(self, loader, transforms=None):
        # 计算相似度的在adapter方法里 其他和ZS无区别
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)               # X: N * embed 测试集
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = self.adapter(X)

        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)              # embed * N'  自己和自己算相似度（训练集）
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)                 # N' * K

        if self.train_tip:
            epochs, lr, bs = 40, 0.001, 1
            # 适配器是线性结构      X：N' * embed     线性层：embed ==》 N'
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())                              # 适配器的权重结构为 N' * embed  初始化为X，这与不训练的一样
            adapter_layer = adapter_layer.to(device)

            # 训练
            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=lr, eps=1e-4)
            indexes = np.arange(0, self.cache_keys.shape[1])                                            # self.cache_keys.shape[1] 样本数
            random.shuffle(indexes)
            for i_epoch in range(epochs):
                loss_epoch = 0.0
                for i_sample in range(self.cache_keys.shape[1]):
                    # 数据
                    image = self.cache_keys[:, indexes[i_sample]].unsqueeze(0).to(device)               # self.cache_keys：embed * N'  ==》 1 * embed
                    target = self.cache_values[indexes[i_sample], :].argmax().unsqueeze(0).to(device)   # self.cache_values：N' * K       取类标索引

                    # 前向过程
                    clip_logits = self.model.logit_scale.exp() * (image @ self.text_embeds.t())
                    affinity = adapter_layer(image)                                                     # 1 * embed ==》 1 * N'  （以训练集为参照 初始化权重）
                    cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values   # 1 * K （以训练集为参照）
                    cache_logits /= X.shape[0]
                    cache_logits *= self.model.logit_scale.exp()
                    tip_logits = clip_logits + cache_logits * self.alpha
                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()/self.cache_keys.shape[1]

                print('loss=%2.5f' % loss_epoch, end="\n")

            # Storage trained adapter
            self.adapter_layer = adapter_layer

    # 适配器 + 相似度计算
    def adapter(self, X):
        # X：N * embed   测试集的视觉特征
        clip_logits = 100 * (X @ self.text_embeds.t().to(device))           # N * K  相似度计算

        # 不训练
        if not self.train_tip:
            affinity = X @ self.cache_keys                                  # N * embed * embed * N' ==》 N * N'     联合训练数据和测试数据
        # 训练
        else:
            affinity = self.adapter_layer(X)                                # 适配器将embed ==》 N'   输出形状为N * N'

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values       # N * N' * N' * K ==》 N * K  这里用了训练集的标签作为提示
        logits = clip_logits + cache_logits * self.alpha

        return logits