import os
import tokenize
from abc import ABC
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
import random

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 64
NUM_WORKERS = 1
CSV_DIR = '/Users/bhuwandutt/PycharmProjects/BSProject/csv/'


class OpenAIDataset(Dataset):

    def __init__(self,
                 file_name: str,
                 batch_size: int = BATCH_SIZE,
                 transform=None):

        super().__init__()
        self.filename = file_name
        self.txt_len = []
        self.csv_text = pd.read_csv(CSV_DIR + "/" + file_name+'.csv')
        self.csv_images = pd.read_csv(CSV_DIR + "/" + 'indiana_projections.csv')
        self.transform = transform
        self.data_dir = PATH_DATASETS
        self.batch_size = batch_size
        self.num_workers = NUM_WORKERS

        self.findings = []
        self.impression = []
        self.image_L = []
        self.image_F = []
        self.subject_ids = []
        self.max_len_finding = self.csv_text['findings'].str.len().max()
        self.max_len_impression = self.csv_text['impression'].str.len().max()
        self.num_tokens = 0
        for index, row in tqdm(self.csv_text.iterrows()):
            subject_id = row['uid']
            self.subject_ids.append(subject_id)

            finding = row['findings']
            impression = row['impression']

            # self.impression.append(impression)
            image_s = self.csv_images.loc[(self.csv_images['uid'] == subject_id)]
            self.image_L.append(image_s.loc[image_s['projection'] == 'Lateral'])
            self.image_F.append(image_s.loc[image_s['projection'] == 'Frontal'])

            # Process to pad the findings and impression to make them of similar length
            # Change Findings and Impression to numpy
            if pd.notnull(finding):
                text_len = len(finding)
                self.num_tokens = self.num_tokens + len(finding.split())
            else:
                text_len = 11
            # print(text_len)
            txt_finding = np.pad(np.array(finding),
                                 (int(self.max_len_finding - text_len), 0),
                                 'constant',
                                 constant_values=0)
            self.findings.append(txt_finding)

            if pd.notnull(impression):
                text_len = len(impression)
                self.num_tokens = self.num_tokens + len(impression.split())
            else:
                text_len = 13
            txt_impression = np.pad(np.array(impression),
                                    (int(self.max_len_impression - text_len), 0),
                                    'constant',
                                    constant_values=0)
            self.impression.append(txt_impression)

            # Find the matching image for this report
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.307,), (0.3081,)),
        #     ]
        # )

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # For png data, load data and normalize
        img_name_F = self.image_F[idx]
        img_name_L = self.image_L[idx]

        chest_img_F = np.array(cv2.cvtColor(cv2.imread(str(img_name_F)), cv2.COLOR_BGR2GRAY))
        chest_img_L = np.array(cv2.cvtColor(cv2.imread(str(img_name_L)), cv2.COLOR_BGR2GRAY))

        if self.transform:
            chest_img_F = self.transform(chest_img_F)
            chest_img_L = self.transform(chest_img_L)

        sample = {
                'subject_id': torch.tensor(self.subject_ids[idx], dtype=torch.long),
                'finding': torch.tensor(self.findings[idx], dtype=torch.long),
                'impression': torch.tensor(self.impression[idx], dtype=torch.long),
                'image_F': torch.tensor(chest_img_F, dtype=torch.float),
                'image_L': torch.tensor(chest_img_L, dtype=torch.float)
        }
        return sample

    def __len__(self):
        return len(self.csv_text)


class OpeniDataset_Siamese(Dataset):
    """View consistency dataset for Open-i"""

    def __init__(self,
                 csv_txt,
                 csv_img,
                 root,
                 transform=None):
        """
        Args:
            csv_txt (string): Path to the csv file with Input txt.
            cvs_img (string): Path to the csv file with Label Images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.text_csv = pd.read_csv(csv_txt)
        self.img_csv = pd.read_csv(csv_img)
        self.root = root
        self.transform = transform
        self.image_L = []
        self.image_F = []
        print("Processing data.....")
        for index, row in tqdm(self.text_csv.iterrows()):
            subject_id = self.text_csv.iloc[index]['subject_id']
            # Find the matching image for this report
            subject_imgs = self.img_csv[self.img_csv.subject_id == subject_id]

            img_name_L = subject_imgs[subject_imgs.direction == 'L'].iloc[0]['path']
            # For png data, load data and normalize
            self.image_L.append(img_name_L)

            # Find the matching image for this report
            img_name_F = subject_imgs[subject_imgs.direction == 'F'].iloc[0]['path']
            # For png data, load data and normalize

            self.image_F.append(img_name_F)

    def __len__(self):
        return len(self.text_csv)

    def get_one_data(self, idx):

        # chest_img_L = np.array(read_png(self.image_L[idx]))
        chest_img_L = np.array(cv2.cvtColor(cv2.imread(str(self.image_L[idx])), cv2.COLOR_BGR2GRAY))
        # chest_img_F = np.array(read_png(self.image_F[idx]))
        chest_img_F = np.array(cv2.cvtColor(cv2.imread(str(self.image_F[idx])), cv2.COLOR_BGR2GRAY))
        if self.transform:
            chest_img_F = self.transform(chest_img_F)
            chest_img_L = self.transform(chest_img_L)

        sample = {
            'image_F': torch.tensor(chest_img_F, dtype=torch.float),
            'image_L': torch.tensor(chest_img_L, dtype=torch.float),
        }
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.get_one_data(idx)
        p = random.uniform(0, 1)
        if p > 0.5:
            # Randomly choose a sample idx!=n_idx
            n_idx = idx
            while idx == n_idx:
                n_idx = random.randint(0, self.__len__() - 1)
            negative = self.get_one_data(n_idx)
            sample['image_L'] = negative['image_L']
            sample['label'] = torch.ones(1).float()
        else:
            sample['label'] = torch.zeros(1).float()
        return sample

