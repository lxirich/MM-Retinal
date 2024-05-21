"""
下游任务适配器 初始化、训练、预测  Downstream task adapter initialization, training, prediction
Zero-shot, Linear Probe (LP), ClipAdapter, TipAdapter, TipAdapter-f
"""

import copy
import random
import torch
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from flair.pretraining.data.transforms import augmentations_pretraining


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


"""
纯视觉适配器 只使用图像编码器
pure vision adapter uses only the image encoder
"""
# 适配器的父类    初始化；实现训练接口；训练、预测虚函数；抽取视觉特征和标签
# Initialization of the adapter's parent class;
# Implement training interface; Training and predicting virtual functions; Extract visual features and labels
class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        '''
        targets：类别  category     tta：测试阶段数据增强  test time data augmentation   fta：训练时数据增强  train/fit time data augmentation
        '''
        self.model = copy.deepcopy(model)
        self.model.eval()                               # 冻结编码器参数  Freezing encoder parameter
        self.num_targets = len(targets)                 # 类别个数  Number of classes
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20                  # 训练增强次数  enhancement times for fta/tta

    # 获取视觉特征和标签  Get visual features and labels
    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()
        epoch_iterator = tqdm(data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

        # 对于后续的适配器 输入是CLIP视觉编码器得到的特征向量 输出是类别号
        # adapter input is the feature vector of CLIP visual encoder and the output is the class number
        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():
                if transforms is not None:
                    images = transforms(images)

                x = self.model.vision_model(images)

            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    # 训练的接口：进行数据增强、抽视觉特征、调用训练函数
    # Training interface:  data augmentation, extract visual features, call training functions
    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]                                                         # 训练集  img path 标签/mask

        # 是否使用训练增强策略 增加训练数据  use augmentation strategies to increase training data?
        if self.fta:
            transforms = augmentations_pretraining
        # 获取视觉特征  get visual features
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)                                                       # 合并成一维  Merge into one dimension
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        self.train(X, Y)

    # 训练用虚函数  Virtual functions for training
    def train(self, X, Y):
        """
        虚函数 由具体适配器实现  Implemented by a specific adapter
        """
        return

    # 预测用虚函数  Virtual functions for prediction
    def predict(self, loader, transforms=None):
        """
        虚函数 由具体适配器实现  Implemented by a specific adapter
        """
        return


# LP适配器        逻辑回归训练；预测  Logistic regression training; forecast   LP
class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        '''
        c逻辑回归的正则化系数  Regularization coefficient of logistic regression
        '''
        super().__init__(model, targets, tta=tta, fta=fta)
        self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0, class_weight="balanced")

    # 训练得到LP的参数 并加入模型当中   The parameters of LP are trained and added to the model
    def train(self, X, Y):
        '''
        X是图片经过图像编码器的特征  image features  Y是类标（数字） label
        '''
        self.classifier.fit(X, Y)

        # 将训练好的逻辑回归加入FLAIR模型   Add the trained logistic regression to the FLAIR model
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)
        self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.classifier.coef_).to(torch.float32))
        self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.classifier.intercept_).to(torch.float32))
        self.model.classifier.to(device)

    # 预测阶段  prediction
    def predict(self, loader, transforms=None):
        self.model.eval()
        # 测试时数据增强  对一个样本进行多次增强 喂到模型中并进行投票（均值）
        # Data augmentation at test
        # Multiple augmentation to a sample and feed into the model to vote (mean)
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        with torch.no_grad():
            refs, preds = [], []                                            # refs是真实标签 gt     preds是预测的标签  prediction
            for step, batch in enumerate(epoch_iterator):
                # 数据  data
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                # 前向过程 输出logits  forward pass outputs logits
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))               # bs * K * 1
                    score = torch.concat(preds_tta, -1).mean(-1)            # bs * K * number_augmentations==》bs * K
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)

                # 激活函数  activation function
                if score.shape[-1] == 1:
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)     # 转为2维的情况      bs * 2
                else:
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
多模态适配器   Multimodal adapter
"""
# 多模态适配器父类      继承适配器父类；增加文本特征的抽取
# Multimodal adapter parent class inherits the adapter parent class; Increase the extraction of text features
class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # 输入类别名称    输出对应类别的文本特征（有/无领域知识）
        # Input category name. Output text characteristics corresponding to the category (with/without domain knowledge)
        self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()), domain_knowledge=domain_knowledge)


# ZS
class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    # 清空训练接口的操作  Clear the operation of the training interface
    def fit(self, loaders, transforms=None):
        return

    def predict(self, loader, transforms=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)                # 获取图像特征和标签  Get image features and labels
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()    # 计算相似度 compute similarity     X:N * embd      text_embeds：K * embd
                scores.append(score_i.unsqueeze(-1))                                                                # score_i.unsqueeze(-1)：N * K * 1
            score = torch.concat(scores, -1).mean(-1)                                                               # score=logits     N * K
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        preds = torch.softmax(score, dim=-1)                                                                        # softmax
        preds = preds.detach().cpu().numpy()
        return refs, preds


# CLIP适配器   视觉空间到文本空间的残差适配器；训练；预测   CLIP adapter: residuals adapter from visual space to text space; Training; forecast
class ClipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.c_in = self.model.vision_model.out_dim     # 特征输出维度  Feature output dimension
        self.reduction = 4
        self.ratio = 0.2                                # adapter采用了残差的结构  residuals structure    ratio * adapter(x) + (1-ratio)  * x

        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.c_in, self.c_in // self.reduction, bias=False),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Linear(self.c_in // self.reduction, self.c_in, bias=False),
                                           torch.nn.ReLU(inplace=True)).to(device)

    def predict(self, loader, transforms=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    X = self.residual_adapter(X)
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

    def train(self, X, Y):
        X = torch.tensor(X)
        Y = torch.tensor(Y)

        epochs, lr, bs = 40, 0.001, 1
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * X.shape[0])
        indexes = np.arange(0, X.shape[0])
        random.shuffle(indexes)

        for i_epoch in range(epochs):
            loss_epoch = 0.0
            for i_sample in range(X.shape[0]):
                X_batch = X[indexes[i_sample], :].unsqueeze(0).to(device)                           # 1 * embd
                target = Y[indexes[i_sample]].unsqueeze(0).to(device)                               # 1，

                X_batch = self.residual_adapter(X_batch)
                logits = self.model.logit_scale.exp() * X_batch @ self.text_embeds.t().to(device)   # 计算相似度  compute similarity
                loss = torch.nn.functional.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_epoch += loss.item()/X.shape[0]

            print('loss=%2.5f' % loss_epoch, end="\n")

    # 视觉空间到文本空间的残差适配器   Residuals adapter for visual space to text space
    def residual_adapter(self, X):
        X_res = self.adapter(X)
        X = self.ratio * X_res + (1 - self.ratio) * X
        X = X / X.norm(dim=-1, keepdim=True)                                    # 归一化特征   Normalize features
        return X


class TipAdapter(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False, train=False):
        '''
        train：是否训练
        '''
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

        self.beta = 5
        self.alpha = 1
        self.cache_keys = []            # 存视觉特征向量  visual feature vectors
        self.cache_values = []          # 存标签 one-hot形式  label one-hot vector

        # 是否训练适配器  train?
        self.train_tip = train
        self.adapter_layer = []         # 存放适配器  adapter

    def predict(self, loader, transforms=None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = self.adapter(X)               # X: N * embed 测试集  test set data
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

        self.cache_keys = torch.transpose(X, 1, 0).to(torch.float32).to(device)              # embed * N'  自己和自己算相似度（训练集）  count similarity with itself (training set)
        self.cache_values = torch.nn.functional.one_hot(Y).to(torch.float32).to(device)                 # N' * K

        if self.train_tip:
            epochs, lr, bs = 40, 0.001, 1
            # 适配器是线性结构      X：N' * embed     线性层 linear layer：embed ==》 N'
            adapter_layer = torch.nn.Linear(self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(device)
            adapter_layer.weight = torch.nn.Parameter(self.cache_keys.t())                              # 适配器的权重结构为 N' * embed  初始化为X，这与不训练的一样
            adapter_layer = adapter_layer.to(device)

            optimizer = torch.optim.AdamW(adapter_layer.parameters(), lr=lr, eps=1e-4)
            indexes = np.arange(0, self.cache_keys.shape[1])                                            # self.cache_keys.shape[1] 样本数  number of sample
            random.shuffle(indexes)
            for i_epoch in range(epochs):
                loss_epoch = 0.0
                for i_sample in range(self.cache_keys.shape[1]):
                    image = self.cache_keys[:, indexes[i_sample]].unsqueeze(0).to(device)               # self.cache_keys：embed * N'  ==》 1 * embed
                    target = self.cache_values[indexes[i_sample], :].argmax().unsqueeze(0).to(device)   # self.cache_values：N' * K       取类标索引   label index

                    clip_logits = self.model.logit_scale.exp() * (image @ self.text_embeds.t())
                    affinity = adapter_layer(image)                                                     # 1 * embed ==》 1 * N'  （以训练集为参照 初始化权重）  Initialize the weights with reference to the training set
                    cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values   # 1 * K （以训练集为参照）  Use the training set as a reference
                    cache_logits /= X.shape[0]
                    cache_logits *= self.model.logit_scale.exp()
                    tip_logits = clip_logits + cache_logits * self.alpha
                    loss = torch.nn.functional.cross_entropy(tip_logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()/self.cache_keys.shape[1]

                print('loss=%2.5f' % loss_epoch, end="\n")

            # Storage trained adapter
            self.adapter_layer = adapter_layer

    # 适配器 + 相似度计算  Adapter + similarity calculation
    def adapter(self, X):
        # X：N * embed   测试集的视觉特征  Visual feature of the test set
        clip_logits = 100 * (X @ self.text_embeds.t().to(device))           # N * K  相似度计算  similarity calculation

        if not self.train_tip:
            affinity = X @ self.cache_keys                                  # N * embed * embed * N' ==》 N * N'     联合训练数据和测试数据   Combine training data and test data
        else:
            affinity = self.adapter_layer(X)                                # 适配器将embed ==》 N'   输出形状为N * N'  output shape

        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values       # N * N' * N' * K ==》 N * K  这里用了训练集的标签作为提示  training sets' label as reference
        logits = clip_logits + cache_logits * self.alpha

        return logits