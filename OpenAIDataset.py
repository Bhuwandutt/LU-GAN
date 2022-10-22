import os
from abc import ABC
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.has_mps.is_available() else 64
NUM_WORKERS = 1
CSV_DIR = '/Users/bhuwandutt/PycharmProjects/BSProject/csv/'


class OpenAIDataset(Dataset):

    def __init__(self,
                 batch_size: int = BATCH_SIZE,
                 transform=None):

        super().__init__()

        self.txt_len = []
        self.csv_text = pd.read_csv(CSV_DIR + "/" + 'indiana_reports.csv')
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

        for index, row in tqdm(self.csv_text.iterrows()):
            subject_id = row['uid']
            self.subject_ids.append(subject_id)

            finding = row['findings']
            # self.findings.append(finding)

            impression = row['impression']
            # self.impression.append(impression)

            image_s = self.csv_images.loc[(self.csv_images['uid'] == subject_id)]
            self.image_L.append(image_s.loc[image_s['projection'] == 'Lateral'])
            self.image_F.append(image_s.loc[image_s['projection'] == 'Frontal'])

            # Process to pad the findings and impression to make them of similar length
            # Change Findings and Impression to numpy
            txt_finding = np.array(finding)
            txt_impression = np.array(impression)

            self.max_len_finding = self.csv_text[self.csv_text['findings']].max()
            text_len = len(txt_finding)
            txt_finding = np.pad(txt_finding, (self.max_len_finding - text_len, 0), 'constant', constant_values=0)
            self.findings.append(txt_finding)

            # text_len = len(impression)
            # self.txt_len.append(text_len)

            self.max_len_impression = self.csv_text[self.csv_text['impression']].max()
            txt_impression = np.pad(txt_impression, (self.max_len_impression - text_len, 0), 'constant',
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
        chest_img_F = np.array(cv2.cvtColor(cv2.imread(img_name_F), cv2.COLOR_BGR2GRAY))
        chest_img_L = chest_img_F = np.array(cv2.cvtColor(cv2.imread(img_name_L), cv2.COLOR_BGR2GRAY))

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






