import torch
import numpy as np
import random
import os


# 随机种子 复现实验  Random seed   reproduction experiment
def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)      # 多GPU  MULTI-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 下载模型预训练权重   Download model pre-training weights
def wget_gdrive_secure(fileid, input_dir, filename):
    os.system("wget --save-cookies COOKIES_PATH 'https://docs.google.com/uc?export=download&id='$fileid -O- | "
              "sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > CONFIRM_PATH".
              replace("$fileid", fileid).replace("COOKIES_PATH", input_dir + "cookies.txt").
              replace("CONFIRM_PATH", input_dir + "confirm.txt"))

    os.system("wget --load-cookies COOKIES_PATH -O $filename"
              " 'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<CONFIRM_PATH)"
              .replace("$fileid", fileid).replace("$filename", input_dir + filename).
              replace("COOKIES_PATH", input_dir + "cookies.txt").
              replace("CONFIRM_PATH", input_dir + "confirm.txt"))