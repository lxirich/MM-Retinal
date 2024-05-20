"""
自定义预训练学习率调整机制
"""

import math
import warnings
import torch

from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        # typing数据类型标注
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,                     # 预热iteration次数
        max_epochs: int,                        # 最大预热次数
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,                   # 最小学习率
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    # 返回当前的lr   LRScheduler.step会调用这个方法
    def get_lr(self) -> List[float]:
        '''
        pytorch 优化器里的lr是分组的 所以这里返回的是float 列表
        '''
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        # 第0个iteration
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)                          # 一组lr 创建base_lrs长度的新lr列表
        # 中间的iteration
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)   # base_lr是训练超参数传入的lr（最大值）  group["lr"]当前某一组参数的lr
            ]                                                                           # 线性增加
        # 最后一个iteration
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        # 余弦调整  CosineAnnealingLR
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (1 +math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)))
            * (group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        step方法接收到epoch参数的时候
        """
        # 线性预热部分
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        # 余弦调整的部分
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


# 预训练阶段lr调整机制
def get_scheduler_per_iteration(optimizer, lr, warmup_epoch, its_per_epoch):
    # its_per_epoch：每个epoch iteration的个数 传入的参数是 len(dataloader)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epoch * its_per_epoch,max_epochs=100 * its_per_epoch,
                                              warmup_start_lr=lr / (warmup_epoch * its_per_epoch))
    return scheduler
