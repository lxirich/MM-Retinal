import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from .datasets import FFAIRDataset, CombinedDataset, MM_Retinal_Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np


class FFAIRDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size * args.n_gpu
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.args.MM_Retinal and self.split == 'train':
            self.dataset1 = FFAIRDataset(self.args, self.tokenizer, self.split, transform=self.transform)
            self.dataset2 = MM_Retinal_Dataset(self.args, self.tokenizer, transform=self.transform)
            self.dataset = CombinedDataset(self.dataset1, self.dataset2, transform=self.transform, testing=self.args.testing)
        else:
            ## load the FFAIRDataset
            self.dataset = FFAIRDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
        }
        super().__init__(**self.init_kwargs)

    
    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        # To adapt to model training, a dimension is added so that the shape of the vector changes from [batch size, 3(channels), 224(H), 224(W)] to [batch size, 1(num of image in one section), 3(channels), 224(H), 224(W)]
        images = torch.unsqueeze(images, dim=1)
        max_seq_length = max(seq_lengths)
        
        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_mask in enumerate(reports_masks):
            targets_masks[i, :len(report_mask)] = report_mask
        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)
