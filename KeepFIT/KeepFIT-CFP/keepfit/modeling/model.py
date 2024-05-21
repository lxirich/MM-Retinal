"""
模型主函数
model main function
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 模型初始化、架构、训练、推理  Model initialization, architecture, training, inference
class KeepFITModel(torch.nn.Module):
    #########################################
    # 模型初始化模块  Model initialization module
    #########################################
    def __init__(self, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="A fundus photograph of [CLS]", projection=True, norm_features=True):
        super().__init__()
        # 输入格式、参数   输出   Input format, parameter output
        self.image_size = image_size
        self.caption = caption                                  # 预测阶段的提示模板  Prompt template for prediction phase
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path

        # transfer阶段是否正则化logit      norm logit?
        self.projection = projection
        self.norm_features = norm_features
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias

        # pretrain网络结构  model architecture
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained              # 是否pretrain  pretrain?
        self.logit_scale_init_value = logit_scale_init_value    # 损失函数可学习温度参数初始值  learnable scale parameter
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        # mixed train with cross Attention 、keepfit with cross Attention feature fusion  混合训练+特征融合  本文方法+特征融合（见消融实验）
        # self.atte_TD2KD = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
        #                                     num_hiddens=self.proj_dim, num_heads=1, dropout=0.5)
        # self.atte_KD2TD = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
        #                                     num_hiddens=self.proj_dim, num_heads=1, dropout=0.5)


        # 领域知识参考结构   Domain knowledge reference
        self.attention = MultiHeadAttention(key_size=self.proj_dim, query_size=self.proj_dim, value_size=self.proj_dim,
                                            num_hiddens=self.proj_dim, num_heads=2, dropout=0.5)

        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)

        self.to(device)

    # 下游任务迁移读取/下载预训练模型参数  Downstream task adaptation read/download pre-trained model parameters
    def load_from_pretrained(self, weights_path=None):
        # 权重路径本地不存在 则创建该路径 并且从云端下载
        # If the weight path does not exist locally, the weight path is created and downloaded from the cloud
        if weights_path is None:
            import zipfile
            input_dir = constants.PATH_PRETRAINED_WEIGHTS           # 权重路径  weight path
            pretrained_id = constants.ID_FLAIR_RESNET_V1            # 权重文件名称  Weight file name
            pretrained_url_id = constants.URL_ID_FLAIR_RESNET_V1
            weights_path = input_dir + pretrained_id
            if not os.path.exists(input_dir + pretrained_id):
                if not os.path.exists(input_dir):
                    Path(input_dir).mkdir(parents=True, exist_ok=True)

                # wget_gdrive_secure(pretrained_url_id, input_dir, filename="weights.zip")
                zipf = zipfile.ZipFile(input_dir + "flair_resnet.zip")
                zipf.extractall(input_dir)
                zipf.close()
                print('\n Download model to:', input_dir + pretrained_id)

        state_dict = torch.load(weights_path, map_location="cuda:0")
        self.load_state_dict(state_dict, strict=True)
        print('load model weight from:', weights_path)

    #########################################
    # 模型训练模块  model train
    #########################################
    def softce_clip_loss(self, logits_per_text, target_pseudo):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        return (caption_loss + image_loss) / 2.0

    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss

    def compute_logits(self, img_emb, text_emb):
        # similarity compute
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text

    def reduce_tensor(self, tensor: torch.Tensor):
        # 分布式计算loss合并  reduce loss
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt

    # 模型训练的主函数  model train main function
    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None, local_rank=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # tensorboard记录
        # if not os.path.isdir("./results/train_records"):
        #     os.mkdir("./results/train_records")
        # write = SummaryWriter(log_dir='../../local_data/results/train_records', flush_secs=60)

        # lr  scheduler
        if scheduler:
            # lr线性预热  linear warmup
            from keepfit.pretraining.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
        else:
            scheduler = None

        epoch = 15
        while epoch <= epochs:
            # 领域知识参考  EK reference
            # loss_epoch = self.train_epoch_KDAtte_s(datalaoders["train"], optimizer, scheduler, transforms, epoch,
            #                               datalaoders["KD"])

            # 领域知识参考与混合训练  EK reference + mixed train
            loss_epoch = self.train_epoch_with_KD_loss_Atte_s(datalaoders["train"], optimizer, scheduler, transforms, epoch,
                                          datalaoders["KD"])
            # 领域知识参考与混合训练+特征融合  EK reference + mixed train + feature fusion
            # loss_epoch = self.train_epoch_with_KD_Atte_KDAtte_s(datalaoders["train"], optimizer, scheduler, transforms,
            #                                                   epoch, datalaoders["KD"])

            # 混合训练   mixed train
            # loss_epoch = self.train_epoch_with_KD(datalaoders["train"], optimizer, scheduler, transforms, epoch,
            #                               datalaoders["KD"])
            # 混合训练+特征融合  mixed train + feature fusion
            # loss_epoch = self.train_epoch_with_KD_Atte(datalaoders["train"], optimizer, scheduler, transforms, epoch,
            #                               datalaoders["KD"])

            if local_rank==0:
                print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))
                # write.add_scalar("train_loss", loss_epoch, epoch)

            # 定期保存  路径给出若不存在会创建路径   save
            if (epoch % store_num == 0) & (local_rank==0):
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.mkdir(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')
            epoch += 1

    # 领域知识参考   EK reference
    def train_epoch_KDAtte_s(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):


        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):

            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)


            with autocast():
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)
                loss = self.softce_clip_loss(logits_per_text, target).to(device)

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)


                KD_embed = self.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)

                mse_loss = torch.nn.MSELoss()
                KD_loss = mse_loss(text_embeds, KD_embed)
                loss += KD_loss

                loss = self.reduce_tensor(loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)

    # ====================================================================================================

    # 混合训练  mixed train
    def train_epoch_with_KD(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):

        self.c = 1

        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)

            KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            with autocast():
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)
                loss = self.softce_clip_loss(logits_per_text, target).to(device)

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds)
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)

                loss = loss + KD_loss * self.c
                loss = self.reduce_tensor(loss)


            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)
    # end

    # 混合训练+特征融合  mixed train + feature fusion
    def train_epoch_with_KD_Atte(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        self.c = 1

        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)

            KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            with autocast():
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)
                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                KD_embed = self.atte_TD2KD(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0))
                TD_embed = self.atte_KD2TD(KD_img_embeds.unsqueeze(0), img_embeds.unsqueeze(0), text_embeds.unsqueeze(0))
                text_embeds += KD_embed.squeeze(0)
                KD_text_embeds += TD_embed.squeeze(0)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)
                loss = self.softce_clip_loss(logits_per_text, target).to(device)
                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds)
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)
                loss = loss + KD_loss * self.c
                loss = self.reduce_tensor(loss)


            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)

    # ====================================================================================================

    # 领域知识参考与混合训练  EK reference + mixed train
    def train_epoch_with_KD_loss_Atte_s(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()                  # 混合精度训练  amp
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))    
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            # KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])          # MM数据   MM data
            KD_text_tokens = self.text_model.tokenize(list(KD_batch["report"][0]))  # 百度 flair数据  baidu/flair data
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            # 一个batch内，一张图片对应的类别，应该和所有具有相同类别的文本为正样本，反之亦然【对称矩阵】
            # In a batch, the corresponding category of a picture should be a positive sample with all texts of
            # the same category, and vice versa.
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)       # 对每列归一化  norm

            # MM数据  MM dataset
            # KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            # KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            # flair 百度数据  flair/baidu dataset
            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in KD_batch["sel_category"]] for iiDesc in KD_batch["sel_category"]], np.float32)
            KD_target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)  # norm  正则化

            with autocast():                                                    # amp 混合精度
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)                                 # data augmentation  数据增强
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)   # 计算相似度  Similarity
                loss = self.softce_clip_loss(logits_per_text, target).to(device)# 计算CLIP损失  clip loss

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds)  # 计算相似度   Similarity
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)  # 计算CLIP损失 clip loss

                # 使用注意力机制的方式计算相似度  Similarity is calculated using the attention mechanism
                KD_embed = self.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)

                mse_loss = torch.nn.MSELoss()
                KD_norm_loss = mse_loss(text_embeds, KD_embed)
                loss = loss + KD_loss + KD_norm_loss * 100

                loss = self.reduce_tensor(loss)                                 # 多线程loss合并   Multithreaded loss merge

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            # 学习率调整 这里是自定义实现的 对iteration进行的学习率调整 所以在epoch里
            # Learning rate scheduler. Here is a custom implementation of learning rate scheduler on iteration so in epoch for loop
            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)

    # 领域知识参考与混合训练+特征融合  EK reference + mixed train + feature fusion
    def train_epoch_with_KD_Atte_KDAtte_s(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, KD_loader=None):
        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, (batch, KD_batch) in enumerate(zip(epoch_iterator, KD_loader)):
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            KD_images = KD_batch["image"].to(device).to(torch.float32)
            KD_text_tokens = self.text_model.tokenize(KD_batch["caption"])
            KD_input_ids = KD_text_tokens["input_ids"].to(device).to(torch.long)
            KD_attention_mask = KD_text_tokens["attention_mask"].to(device).to(torch.long)

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)

            KD_target = np.eye(len(KD_batch["caption"]), dtype=np.float32)
            KD_target = torch.tensor(KD_target).to(device).to(torch.float32)

            with autocast():
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                print("\nExtracting KD features...")
                KD_img_embeds = self.vision_model(KD_images)
                KD_text_embeds = self.text_model(KD_input_ids, KD_attention_mask)

                KD_embed = self.attention(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0)).squeeze(0)
                mse_loss = torch.nn.MSELoss()
                KD_norm_loss = mse_loss(text_embeds, KD_embed)

                KD_embed = self.atte_TD2KD(img_embeds.unsqueeze(0), KD_img_embeds.unsqueeze(0), KD_text_embeds.unsqueeze(0))

                TD_embed = self.atte_KD2TD(KD_img_embeds.unsqueeze(0), img_embeds.unsqueeze(0), text_embeds.unsqueeze(0))
                text_embeds_new = KD_embed.squeeze(0) + text_embeds
                KD_text_embeds_new = TD_embed.squeeze(0) + KD_text_embeds

                logits_per_text= self.compute_logits(img_embeds, text_embeds_new)
                loss = self.softce_clip_loss(logits_per_text, target).to(device)
                KD_logits_per_text = self.compute_logits(KD_img_embeds, KD_text_embeds_new)
                KD_loss = self.softce_clip_loss(KD_logits_per_text, KD_target).to(device)

                loss = loss + KD_loss + KD_norm_loss * 100

                loss = self.reduce_tensor(loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)
    #

    #########################################
    # 模型推理模块  prediction
    #########################################
    def forward(self, image, text):
        self.eval()
        # 输入数据预处理  pre process
        image = self.preprocess_image(image)
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)
            logits = self.compute_logits(img_embeds, text_embeds).t()
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy(), logits.cpu().numpy()

    # 图像预处理  img preprocess
    def preprocess_image(self, image):
        if image.dtype != np.float32:
            image = np.float32(image)

        # 归一化  norm
        if image.max() > 0:
            image /= 255
        if len(image.shape) > 2:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        # 尺度不变缩放  Scale-invariant scaling
        image = torch.tensor(image)
        sizes = image.shape[-2:]
        max_size = max(sizes)
        scale = max_size / self.image_size
        image = torchvision.transforms.Resize((int(image.shape[-2] / scale), int(image.shape[-1] / scale)))(image)
        image = torch.nn.functional.pad(image, (0, self.image_size - image.shape[-1], 0, self.image_size - image.shape[-2], 0, 0))

        image = image.to(torch.float32).to(device)
        return image

    # 文本预处理（简易版）  text preprocess
    def preprocess_text(self, text):
        prompts = [self.caption.replace("[CLS]", category) for category in text]
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

        return input_ids, attention_mask

    # 迁移时使用的文本预处理（领域知识）  text preprocess
    def compute_text_embeddings(self, categories, domain_knowledge=False):
        text_embeds_dict = {}                                                       # 类别名称：文本特征向量  Category name: text feature vector
        for iKey in range(len(categories)):
            if domain_knowledge and categories[iKey] in list(definitions.keys()):   # [描述+类名]  [Description + class name]
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]

            # 生成文本特征  get text embed
            with torch.no_grad():
                print(descriptions)
                # 提示模板  prompt template
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)
                # 生产文本特征  get text feature
                text_embeds = self.text_model(input_ids, attention_mask)            # 领域描述个数 * 特征维度   Number of domain descriptions * feature dimension

            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)   # 将不同描述的文本特征平均（文中方法）==》1 * 特征维度   Average the text features of different descriptions

        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds

#########################################
# 模型架构模块  model architecture
#########################################
# 视觉信息编码网络  visual encoder
class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        if vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet']:
            print("Vision model should be one of resnet/efficientnet... using resnet.")
            vision_type = "resnet_v1"

        # 获取预训练权重  pre-train weight
        if vision_type == "resnet_v1" or vision_type == "resnet_v2":
            if vision_type == "resnet_v1":
                weights = 'IMAGENET1K_V1' if pretrained else None
            elif vision_type == "resnet_v2":
                weights = 'IMAGENET1K_V2' if pretrained else None
            else:
                weights = 'IMAGENET1K_V1' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            self.vision_dim = 2048
            self.model.fc = torch.nn.Identity()

        elif vision_type == "efficientnet":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096

        # 投影  projection
        if projection:
            self.out_dim = self.proj_dim
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,bias=proj_bias)
                                                      ,projection=projection, norm=norm)

    def forward(self, pixel_values):
        embed = self.model(pixel_values)
        embed = self.projection_head_vision(embed)
        return embed


# 文本信息编码网路  text encoder
class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 256                            # 输入最大长度  max input length

        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        # 投影  Projection
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)

        embed = self.projection_head_text(embed)
        return embed


# MLP投影层  ProjectionLayer
class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()
        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):
        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)        # 除以L2范数  L2 norm
        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)

        return x


# 多头注意力机制  Multi-head attention mechanism
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q = torch.nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = torch.nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = torch.nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = torch.nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        # output shape: (batch_size*num_heads，num of queries，num_hiddens/num_heads)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.nn.functional.softmax(scores)
        output = torch.bmm(self.dropout(self.attention_weights), values)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        # output_concat shape:(batch_size，num of queries，num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(torch.nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = torch.nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        # x: batch、times、numhid


class AddNorm(torch.nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.ln = torch.nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)