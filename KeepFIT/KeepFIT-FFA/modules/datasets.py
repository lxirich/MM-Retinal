import os
import ast
import pandas as pd
import json
import torch
from PIL import Image
import argparse
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_json_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.load(open(self.ann_path))
        self.examples = self.ann[self.split]
        self.masks = []
        self.reports = []
        self.image_path = []
        self.case_ids = []
        
        for each in self.examples.keys():
            self.image_path += self.examples[each]['Image_path']
            En_Report = self.tokenizer(self.examples[each]['En_Report'])[:self.max_seq_length]
            mask = [1]*len(En_Report)
            for _ in range(len(self.examples[each]['Image_path'])):
                self.reports.append(En_Report)
                self.masks.append(mask)
                self.case_ids.append(each)
        # Buliding subset for testing the code
        if args.testing:
            self.case_ids = self.case_ids[:10]
            self.image_path = self.image_path[:10]
            self.reports = self.reports[:10]
            self.masks = self.masks[:10]
            
    def __len__(self):
        return len(self.image_path)

class FFAIRDataset(BaseDataset):
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        image_id = case_id
        image_path = self.image_path[idx]
        image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = self.reports[idx]
        report_masks = self.masks[idx]
        seq_length = len(report_masks)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
    
    def get_all_item(self):
        return [self.case_ids, self.image_path, self.reports, self.masks]

class MM_Retinal_Dataset(Dataset):
    def __init__(self, args, tokenizer, transform=None):
        dataframe = pd.read_csv(args.csv_path)
        self.image = []
        self.report = []
        self.mask = []
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        for i, item in dataframe.iterrows():
            # Only save image path
            image_path = item['image'].replace('FFA', 'FFA_resized_jpg')
            if 'png' in image_path:
                image_path = image_path.replace('png', 'jpg')
            image_path = args.data_path + image_path
            if os.path.exists(image_path):
                self.image.append(image_path)
            else:
                print(f"Image path is not existed:{image_path}")
                continue
            reports = [self.tokenizer(report)[:self.max_seq_length] for report in ast.literal_eval(item['categories'])]
            self.report.append(reports[0])
            self.mask.append([1]*len(reports[0]))
            
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        image_id = idx
        image = Image.open(self.image[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        report_ids = self.report[idx]
        report_masks = self.mask[idx]
        seq_length = len(report_masks)
        
        return (image_id, image, report_ids, report_masks, seq_length)
            
    def get_all_item(self):
        return [range(len(self.image)), self.image, self.report, self.mask]

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2, transform, testing=False):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transform = transform
        self.data = []
        for i in range(max(len(dataset1), len(dataset2))):
            data1 = [data[i%len(dataset1)] for data in dataset1.get_all_item()]
            data1[1] = os.path.join(self.dataset1.image_dir, data1[1])
            self.data.append(tuple(data1[i] for i in range(len(data1))))
            self.data.append(tuple(data[i%len(dataset2)] for data in dataset2.get_all_item()))
        print(f'FFA-IR Dataset has {len(self.dataset1)} images, MM_Retinal Dataset has {len(self.dataset2)} images, CombninedDataset has {len(self.data)} images')
        # Building subset for testing the code
        if testing:
            self.data = self.data[:20]
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id, image_path, report, report_mask = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return (image_id, image, report, report_mask, seq_length)
        