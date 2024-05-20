"""
模型主函数
"""
import math
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path

from .dictionary import definitions
from . import constants
from .misc import wget_gdrive_secure

import torch
import torchvision
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, logging           # 自然语言处理的包

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"    # 禁用分词器的并行处理 防止并发错误

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 模型初始化、架构、训练、推理
class KeepFITModel(torch.nn.Module):
    #########################################
    # 模型初始化模块
    #########################################
    def __init__(self, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="A fundus photograph of [CLS]", projection=True, norm_features=True):
        super().__init__()
        # 输入格式、参数   输出
        self.image_size = image_size
        self.caption = caption                                  # 预测阶段的提示模板
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path

        # transfer阶段是否正则化logit
        self.projection = projection
        self.norm_features = norm_features
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias

        # pretrain网络结构
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained              # 是否pretrain
        self.logit_scale_init_value = logit_scale_init_value    # 损失函数可学习温度参数初始值
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # withKD交叉注意力、混合3loss v2
        # self.atte_TD2KD = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
        #                                     num_hiddens=self.proj_dim, num_heads=1, dropout=0.5)
        # self.atte_KD2TD = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
        #                                     num_hiddens=self.proj_dim, num_heads=1, dropout=0.5)


        # KD Attention方法参考  用于v1 v2 混合3loss v1 v2
        self.attention = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
                                            num_hiddens=self.proj_dim, num_heads=2, dropout=0.5)

        # 下游任务迁移读取预训练参数
        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)

        self.to(device)

    # 下游任务迁移读取/下载预训练模型参数
    def load_from_pretrained(self, weights_path=None):
        # 权重路径本地不存在 则创建该路径 并且从云端下载
        if weights_path is None:
            import zipfile
            input_dir = constants.PATH_PRETRAINED_WEIGHTS           # 权重路径
            pretrained_id = constants.ID_FLAIR_RESNET_V1            # 权重文件名称
            pretrained_url_id = constants.URL_ID_FLAIR_RESNET_V1
            weights_path = input_dir + pretrained_id
            if not os.path.exists(input_dir + pretrained_id):       # 创建路径
                if not os.path.exists(input_dir):
                    # 父路径不存在会创建  如果目录存在不会发生异常
                    Path(input_dir).mkdir(parents=True, exist_ok=True)

                # 下载并解压模型权重
                # wget_gdrive_secure(pretrained_url_id, input_dir, filename="weights.zip")
                zipf = zipfile.ZipFile(input_dir + "flair_resnet.zip")
                zipf.extractall(input_dir)
                zipf.close()
                print('\n Download model to:', input_dir + pretrained_id)

        state_dict = torch.load(weights_path, map_location="cuda:0")
        self.load_state_dict(state_dict, strict=True)
        print('load model weight from:', weights_path)

    #########################################
    # 模型训练模块
    #########################################
    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        # 计算相似度
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)          # 防止温度参数爆炸
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text

    def reduce_tensor(self, tensor: torch.Tensor):
        # 分布式计算loss合并
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()                                                # 总进程数
        return rt

    # 模型训练的主函数
    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None, local_rank=None, knowledge_dict=False):
        # transforms预训练数据增强操作   store_num存储周期

        # 优化器 lr调整机制
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # tensorboard记录
        # if not os.path.isdir("./results/train_records"):
        #     os.mkdir("./results/train_records")
        # write = SummaryWriter(log_dir='../../local_data/results/train_records', flush_secs=60)

        if scheduler:
            # lr线性预热
            from keepfit.pretraining.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
        else:
            scheduler = None

        # 训练
        epoch = 15
        while epoch <= epochs:
            # 参考系列
            # loss_epoch = self.train_epoch_KDAtte_s(datalaoders["train"], optimizer, scheduler, transforms, epoch,
            #                               datalaoders["KD"])                                # KD作为参考 采样 注意力

            # 3loss系列 参考+共同训练
            loss_epoch = self.train_epoch_with_KD_loss_Atte_s(datalaoders["train"], optimizer, scheduler, transforms, epoch,
                                          datalaoders["KD"])                                # KD作为参考 采样 注意力 一起训练
            # loss_epoch = self.train_epoch_with_KD_Atte_KDAtte_s(datalaoders["train"], optimizer, scheduler, transforms,
            #                                                   epoch, datalaoders["KD"])  # KD作为参考 采样 注意力 一起训练 v2

            # 共同训练系列
            # loss_epoch = self.train_epoch_with_KD(datalaoders["train"], optimizer, scheduler, transforms, epoch,
            #                               datalaoders["KD"])                                # KD一起训练
            # loss_epoch = self.train_epoch_with_KD_Atte(datalaoders["train"], optimizer, scheduler, transforms, epoch,
            #                               datalaoders["KD"])                                # KD一起训练 注意力

            if local_rank==0:
                print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))
                # write.add_scalar("train_loss", loss_epoch, epoch)

            # 定期保存  路径给出若不存在会创建路径
            if (epoch % store_num == 0) & (local_rank==0):
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.mkdir(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')
            epoch += 1

    # 训练一个epoch的函数 返回损失  KD数据（采样）生成loss的一部分  Attention  采样过程参与训练
    def train_epoch_KDAtte_s(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        # loader为dataloader["train"]

        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  # 梯度范数的最大值（用于防止梯度爆炸）  混合精度训练
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)                                         # 不同进程随机打乱使用相同随机种子

        # 使用进度条包装可迭代对象，调用和没包装前一样      desc为输出的描述      dynamic_ncols为自适应宽度
        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            # 这里batch中每个数据是字典（包括图片数据、文本、str类型的类别）

            # 模型输入
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    # 对text进行词元化tokenize 得到一个字典（每个词的embd编码 和 attention mask）
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            # 构造label
            # 一个batch内，一张图片对应的类别，应该和所有具有相同类别的文本为正样本，反之亦然【对称矩阵】
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)       # 对每列归一化

            # 前向过程
            with autocast():                                                    # 混合精度训练
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)                                 # 数据增强
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)   # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                loss = self.softce_clip_loss(logits_per_text, target).to(device)# 计算CLIP损失  选择哪一个都一样（对称）

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                # 使用注意力机制的方式计算像素度
                KD_embed = self.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)

                mse_loss = torch.nn.MSELoss()
                KD_loss = mse_loss(text_embeds, KD_embed)
                loss += KD_loss

                loss = self.reduce_tensor(loss)                                 # 多线程loss合并

            # 反向传播 混合精度
            scaler.scale(loss).backward()                                       # 缩放梯度减少梯度数值范围并反向传播 防止上溢下溢
            scaler.unscale_(optimizer)                                          # 反缩放 用于更新优化器的模型参数 才能进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)    # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()                                                     # 更新缩放器的参数 看是否出现inf
            optimizer.zero_grad()

            # 输出
            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # 学习率调整 这里是自定义实现的 对iteration进行的学习率调整 所以在epoch里
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)                                           # 返回这个epoch的loss

    # ====================================================================================================

    # 训练一个epoch的函数 返回损失  KD数据和常规数据一起参与训练
    def train_epoch_with_KD(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        # loader为dataloader["train"]

        self.c = 1                                                              # 两个loss的权重

        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  # 梯度范数的最大值（用于防止梯度爆炸）  混合精度训练
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)                                         # 不同进程随机打乱使用相同随机种子

        # 使用进度条包装可迭代对象，调用和没包装前一样      desc为输出的描述      dynamic_ncols为自适应宽度
        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            # 这里batch中每个数据是字典（包括图片数据、文本、str类型的类别）

            # 模型输入
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    # 对text进行词元化tokenize 得到一个字典（每个词的embd编码 和 attention mask）
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            # 构造label
            # 一个batch内，一张图片对应的类别，应该和所有具有相同类别的文本为正样本，反之亦然【对称矩阵】
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)       # 对每列归一化

            KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            # 前向过程
            with autocast():                                                    # 混合精度训练
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)                                 # 数据增强
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)   # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                loss = self.softce_clip_loss(logits_per_text, target).to(device)# 计算CLIP损失  选择哪一个都一样（对称）

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds)  # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)  # 计算CLIP损失  选择哪一个都一样（对称）

                loss = loss + KD_loss * self.c
                loss = self.reduce_tensor(loss)                                 # 多线程loss合并

            # 反向传播 混合精度
            scaler.scale(loss).backward()                                       # 缩放梯度减少梯度数值范围并反向传播 防止上溢下溢
            scaler.unscale_(optimizer)                                          # 反缩放 用于更新优化器的模型参数 才能进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)    # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()                                                     # 更新缩放器的参数 看是否出现inf
            optimizer.zero_grad()

            # 输出
            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # 学习率调整 这里是自定义实现的 对iteration进行的学习率调整 所以在epoch里
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)                                           # 返回这个epoch的loss

    # 训练一个epoch的函数 返回损失  KD数据和常规数据一起参与训练 交叉注意力
    def train_epoch_with_KD_Atte(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        # loader为dataloader["train"]

        self.c = 1                                                              # 两个loss的权重

        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  # 梯度范数的最大值（用于防止梯度爆炸）  混合精度训练
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)                                         # 不同进程随机打乱使用相同随机种子

        # 使用进度条包装可迭代对象，调用和没包装前一样      desc为输出的描述      dynamic_ncols为自适应宽度
        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            # 这里batch中每个数据是字典（包括图片数据、文本、str类型的类别）

            # 模型输入
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    # 对text进行词元化tokenize 得到一个字典（每个词的embd编码 和 attention mask）
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            # 构造label
            # 一个batch内，一张图片对应的类别，应该和所有具有相同类别的文本为正样本，反之亦然【对称矩阵】
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)       # 对每列归一化

            KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            # 前向过程
            with autocast():                                                    # 混合精度训练
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)                                 # 数据增强
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)
                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                # 交叉注意力
                # 训练数据作为query
                KD_embed = self.atte_TD2KD(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0))
                # KD数据作为query
                TD_embed = self.atte_KD2TD(KD_img_embeds.unsqueeze(0), img_embeds.unsqueeze(0), text_embeds.unsqueeze(0))
                text_embeds += KD_embed.squeeze(0)
                KD_text_embeds += TD_embed.squeeze(0)

                # 损失计算
                logits_per_text= self.compute_logits(img_embeds, text_embeds)   # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                loss = self.softce_clip_loss(logits_per_text, target).to(device)# 计算CLIP损失  选择哪一个都一样（对称）
                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds)  # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)  # 计算CLIP损失  选择哪一个都一样（对称）
                loss = loss + KD_loss * self.c
                loss = self.reduce_tensor(loss)                                 # 多线程loss合并

            # 反向传播 混合精度
            scaler.scale(loss).backward()                                       # 缩放梯度减少梯度数值范围并反向传播 防止上溢下溢
            scaler.unscale_(optimizer)                                          # 反缩放 用于更新优化器的模型参数 才能进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)    # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()                                                     # 更新缩放器的参数 看是否出现inf
            optimizer.zero_grad()

            # 输出
            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # 学习率调整 这里是自定义实现的 对iteration进行的学习率调整 所以在epoch里
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)                                           # 返回这个epoch的loss

    # ====================================================================================================

    # 训练一个epoch的函数 返回损失  KD数据（采样）生成loss的一部分  Attention  采样过程参与训练   混合训练
    def train_epoch_with_KD_loss_Atte_s(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        # loader为dataloader["train"]

        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  # 梯度范数的最大值（用于防止梯度爆炸）  混合精度训练
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)                                         # 不同进程随机打乱使用相同随机种子

        # 使用进度条包装可迭代对象，调用和没包装前一样      desc为输出的描述      dynamic_ncols为自适应宽度
        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            # 这里batch中每个数据是字典（包括图片数据、文本、str类型的类别）

            # 模型输入
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    # 对text进行词元化tokenize 得到一个字典（每个词的embd编码 和 attention mask）
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            # KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])          # MM数据
            KD_text_tokens = self.text_model.tokenize(list(KD_batch["report"][0]))  # 百度 flair数据
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            # 构造label
            # 一个batch内，一张图片对应的类别，应该和所有具有相同类别的文本为正样本，反之亦然【对称矩阵】
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)       # 对每列归一化

            # MM数据
            # KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            # KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            # flair 百度数据
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in KD_batch["sel_category"]] for iiDesc in KD_batch["sel_category"]], np.float32)
            KD_target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)  # 对每列归一化

            # 前向过程
            with autocast():                                                    # 混合精度训练
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)                                 # 数据增强
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)   # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                loss = self.softce_clip_loss(logits_per_text, target).to(device)# 计算CLIP损失  选择哪一个都一样（对称）

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds)  # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)  # 计算CLIP损失  选择哪一个都一样（对称）

                # 使用注意力机制的方式计算像素度
                KD_embed = self.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)

                mse_loss = torch.nn.MSELoss()
                KD_norm_loss = mse_loss(text_embeds, KD_embed)
                loss = loss + KD_loss + KD_norm_loss * 100

                loss = self.reduce_tensor(loss)                                 # 多线程loss合并

            # 反向传播 混合精度
            scaler.scale(loss).backward()                                       # 缩放梯度减少梯度数值范围并反向传播 防止上溢下溢
            scaler.unscale_(optimizer)                                          # 反缩放 用于更新优化器的模型参数 才能进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)    # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()                                                     # 更新缩放器的参数 看是否出现inf
            optimizer.zero_grad()

            # 输出
            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # 学习率调整 这里是自定义实现的 对iteration进行的学习率调整 所以在epoch里
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)                                           # 返回这个epoch的loss

    # 训练一个epoch的函数 返回损失  KD数据（采样）生成loss的一部分  交叉注意力混合 共同训练  采样过程参与训练
    def train_epoch_with_KD_Atte_KDAtte_s(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        # loader为dataloader["train"]

        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  # 梯度范数的最大值（用于防止梯度爆炸）  混合精度训练
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)                                         # 不同进程随机打乱使用相同随机种子

        # 使用进度条包装可迭代对象，调用和没包装前一样      desc为输出的描述      dynamic_ncols为自适应宽度
        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            # 这里batch中每个数据是字典（包括图片数据、文本、str类型的类别）

            # 模型输入
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    # 对text进行词元化tokenize 得到一个字典（每个词的embd编码 和 attention mask）
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            # 构造label
            # 一个batch内，一张图片对应的类别，应该和所有具有相同类别的文本为正样本，反之亦然【对称矩阵】
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)       # 对每列归一化

            KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            # 前向过程
            with autocast():                                                    # 混合精度训练
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)                                 # 数据增强
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                # 先计算KDAtte的MSE损失
                KD_embed = self.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)
                mse_loss = torch.nn.MSELoss()
                KD_norm_loss = mse_loss(text_embeds, KD_embed)

                # 进行数据集间的交叉注意力
                # 训练数据作为query
                KD_embed = self.atte_TD2KD(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0))
                # KD数据作为query
                TD_embed = self.atte_KD2TD(KD_img_embeds.unsqueeze(0), img_embeds.unsqueeze(0), text_embeds.unsqueeze(0))
                text_embeds_new = KD_embed.squeeze(0) + text_embeds
                KD_text_embeds_new = TD_embed.squeeze(0) + KD_text_embeds

                # 融合后 计算两个数据集的损失
                logits_per_text= self.compute_logits(img_embeds, text_embeds_new)   # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                loss = self.softce_clip_loss(logits_per_text, target).to(device)# 计算CLIP损失  选择哪一个都一样（对称）
                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds_new)  # 计算相似度     矩阵每一行表示一个图片和所有text的相似度
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)  # 计算CLIP损失  选择哪一个都一样（对称）

                loss = loss + KD_loss + KD_norm_loss * 100

                loss = self.reduce_tensor(loss)                                 # 多线程loss合并

            # 反向传播 混合精度
            scaler.scale(loss).backward()                                       # 缩放梯度减少梯度数值范围并反向传播 防止上溢下溢
            scaler.unscale_(optimizer)                                          # 反缩放 用于更新优化器的模型参数 才能进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)    # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()                                                     # 更新缩放器的参数 看是否出现inf
            optimizer.zero_grad()

            # 输出
            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # 学习率调整 这里是自定义实现的 对iteration进行的学习率调整 所以在epoch里
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)                                           # 返回这个epoch的loss

    #########################################
    # 模型推理模块
    #########################################
    def forward(self, image, text):
        self.eval()
        # 输入数据预处理
        image = self.preprocess_image(image)
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        # 模型前向推理
        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)
            logits = self.compute_logits(img_embeds, text_embeds).t()
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy(), logits.cpu().numpy()

    # 图像预处理
    def preprocess_image(self, image):
        if image.dtype != np.float32:
            image = np.float32(image)

        # 归一化
        if image.max() > 0:
            image /= 255
        # 通道变换
        if len(image.shape) > 2:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)                                    # 扩充bs维度
        # 尺度不变缩放
        image = torch.tensor(image)
        sizes = image.shape[-2:]
        max_size = max(sizes)
        scale = max_size / self.image_size
        image = torchvision.transforms.Resize((int(image.shape[-2] / scale), int(image.shape[-1] / scale)))(image)
        image = torch.nn.functional.pad(image, (0, self.image_size - image.shape[-1], 0, self.image_size - image.shape[-2], 0, 0))

        image = image.to(torch.float32).to(device)
        return image

    # 文本预处理（简易版）
    def preprocess_text(self, text):
        # 提示模板
        prompts = [self.caption.replace("[CLS]", category) for category in text]
        # 词元化 产生文本编码器输入
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

    # 迁移时使用的文本预处理（领域知识）
    def compute_text_embeddings(self, categories, domain_knowledge=False):
        '''
        categories：类别列表
        domain_knowledge：是否使用领域知识
        '''
        text_embeds_dict = {}                                                       # 类别名称：文本特征向量
        for iKey in range(len(categories)):
            # 使用领域知识 且领域描述存在
            if domain_knowledge and categories[iKey] in list(definitions.keys()):   # [描述+类名]
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]                                   # [类名]

            # 生成文本特征
            with torch.no_grad():
                print(descriptions)
                # 提示模板
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                # 词元化 产生文本编码器输入     truncation：截断（超最大长度）       padding：补齐（低于最大长度）      return_tensors：返回类型（pytorch）
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)
                # 生产文本特征
                text_embeds = self.text_model(input_ids, attention_mask)            # 领域描述个数 * 特征维度

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)   # 将不同描述的文本特征平均（文中方法）==》1 * 特征维度

        text_embeds = torch.concat(list(text_embeds_dict.values()))                 # 文本特征向量

        return text_embeds_dict, text_embeds

#########################################
# 模型架构模块
#########################################
# 视觉信息编码网络
class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        # 框架选择 默认为ResNet v1
        if vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet']:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # 获取预训练权重
        if vision_type == "resnet_v1" or vision_type == "resnet_v2":
            # ResNet 获取其在ImageNet的权重作为初始化（预训练）  获取预训练权重（迁移）
            if vision_type == "resnet_v1":
                weights = 'IMAGENET1K_V1' if pretrained else None
            elif vision_type == "resnet_v2":
                weights = 'IMAGENET1K_V2' if pretrained else None
            else:
                weights = 'IMAGENET1K_V1' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            self.vision_dim = 2048                                      # 投影前特征向量的维度
            self.model.fc = torch.nn.Identity()

        elif vision_type == "efficientnet":
            # EfficientNet 获取其在ImageNet的权重作为初始化（预训练）  获取预训练权重（迁移）
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096

        # 投影
        if projection:
            self.out_dim = self.proj_dim
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,bias=proj_bias)
                                                      ,projection=projection, norm=norm)

    def forward(self, pixel_values):
        embed = self.model(pixel_values)
        embed = self.projection_head_vision(embed)
        return embed


# 文本信息编码网路
class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 256                            # 输入最大长度

        # 文本编码器，输出所有层的隐状态，模型的输出是列表
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        # 投影
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        # text：待编码的序列      truncation：截断（超最大长度）       padding：补齐（低于最大长度）      return_tensors：返回类型（pytorch）
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        # tokenize之后每个token对应一个id（字典索引）
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        embed = self.projection_head_text(embed)
        return embed


# MLP投影层
class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()
        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):
        # 投影+norm     norm+投影+norm
        # 只投影        投影
        # 只norm       无操作
        # 不投影不norm  无操作
        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)        # 除以L2范数
        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x


# 多头注意力机制
class MultiHeadAttention(torch.nn.Module):
    """并行实现多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        # num_hiddens用于设置最后的输出
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q = torch.nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = torch.nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = torch.nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = torch.nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def transpose_qkv(self, X, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        """逆转transpose_qkv函数的操作"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.nn.functional.softmax(scores)
        output = torch.bmm(self.dropout(self.attention_weights), values)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(torch.nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = torch.nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        # 输入的是batch、times、numhid  time对应了不同时间点的输出


class AddNorm(torch.nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.ln = torch.nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)  # y是transformer块的输出，x是输入