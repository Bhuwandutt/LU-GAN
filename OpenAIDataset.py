import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torchvision.transforms as transforms
import random
from utils.utils import *
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 32
NUM_WORKERS = 1
CSV_DIR = os.getcwd()+'/csv/'
IMAGE_DIR = os.getcwd()+'/data/imgs/'


class OpenAIDataset(Dataset):

    def __init__(self,
                 file_name: str,
                 transform=transforms.Compose([Rescale((256, 256)),
                                               ToTensor()])):

        super().__init__()
        self.filename = file_name
        self.txt_len = []
        self.csv_text = pd.read_csv(CSV_DIR + file_name + '.csv')
        self.csv_images = pd.read_csv(CSV_DIR + 'indiana_projections.csv')

        self.data_dir = PATH_DATASETS
        self.num_workers = NUM_WORKERS
        self.transform = transform
        self.device = 'mps'

        # Tokenizer for BioLinkBERT
        # The sentence would be tokenized using pretrained Tokenizer to create a dictionary which maps the id's
        # to the text and attention mask
        self.tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
        #

        self.findings = []
        self.impression = []
        self.image_L = []
        self.image_F = []
        self.subject_ids = []
        # self.word_dict = os.getcwd()+'/utils/dict.json'
        self.max_len_impression, self.max_len_finding = self.get_maximum_length()

        for index, row in tqdm(self.csv_text.iterrows()):
            subject_id = row['uid']

            self.subject_ids.append(subject_id)
            image_ = self.csv_images.loc[
                (self.csv_images['uid'] == subject_id)].values
            self.image_F.append(image_[0][1])
            self.image_L.append(image_[1][1])

            fi = row['findings']
            im = row['impression']

            self.findings.append(fi)
            self.impression.append(im)

    def __getitem__(self, idx):
        # The function is called to retrieve items from the Dataset. Items are called using index.
        # The items are then converted (from numpy arrays) to torch object and returned as a dictionary of torch values
        # The size of numpy/torch are as follows :-
        #                                     finding[i] = torch.Size([1, n+1] ) n =Maximum length of Finding (165)
        #                                     impression[i] = torch.size([1, k+1] ) k = Max length of Impression(124)
        #                                     image_f[i] = torch.Size([1, 1, 256, 256])
        #                                     image_l[i] = torch.size([1, 1, 256, 256])
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # For png data, load data and normalize
        img_name_F = self.image_F[idx]
        img_name_L = self.image_L[idx]

        chest_img_F = np.array(read_png(IMAGE_DIR + str(img_name_F)))
        chest_img_L = np.array(read_png(IMAGE_DIR + str(img_name_L)))

        chest_img_F = self.transform(chest_img_F)
        # print("After Transformation", chest_img_F.shape)
        chest_img_L = self.transform(chest_img_L)

        tokenized_finding = self.tokenizer(self.findings[idx],
                                           padding='max_length',
                                           max_len=self.max_len_finding,
                                           return_tensors="pt")
        tokenized_impression = self.tokenizer(self.impression[idx],
                                              padding='max_length',
                                              max_len=self.max_len_impression,
                                              return_tensors="pt")

        dict_finding = {'input_ids': tokenized_finding.input_ids.to(device=self.device),
                        'attention_mask':  tokenized_finding.attention_mask.to(device=self.device)}

        dict_impression = {'input_ids': tokenized_impression.input_ids.to(device=self.device),
                           'attention_mask': tokenized_impression.attention_mask.to(device=self.device)}

        # print(self.findings[idx].shape)

        # print(self.findings[idx])

        sample = {
            'subject_id': torch.tensor(self.subject_ids[idx], dtype=torch.long),
            'finding': torch.tensor(dict_finding, dtype=torch.long),
            'impression': torch.tensor(dict_impression, dtype=torch.long),
            'image_F': torch.tensor(chest_img_F, dtype=torch.float),
            'image_L': torch.tensor(chest_img_L, dtype=torch.float)
        }
        return sample

    def __len__(self):
        return len(self.csv_text)

    def get_maximum_length(self):

        len_finding = []
        len_impression = []

        for index, row in tqdm(self.csv_text.iterrows()):
            fi = row['findings'].split()
            im = row['impression'].split()
            len_impression.append(len(im))
            len_finding.append(len(fi))

        max_len_im, max_len_fi = max(len_impression), max(len_finding)

        print("Totally {} medical report".format(self.__len__()))
        print("Max Finding: Sentence Length {}  ".format(max_len_fi))
        print("Max Impression: Sentence length {} ".format(max_len_im))
        return max_len_im, max_len_fi


class ViewConsistencyDataset:

    def __init__(self, file_name: str,
                 transform=transforms.Compose([Rescale((256, 256)),
                                               ToTensor()])):

        self.image_L = []
        self.image_F = []
        self.subject_ids = []
        self.csv_text = pd.read_csv(CSV_DIR + file_name + '.csv')
        self.csv_images = pd.read_csv(CSV_DIR + 'indiana_projections.csv')
        self.data_dir = PATH_DATASETS
        self.num_workers = NUM_WORKERS
        self.transform = transform

        for index, row in tqdm(self.csv_text.iterrows()):
            subject_id = row['uid']
            self.subject_ids.append(subject_id)
            image_ = self.csv_images.loc[
                (self.csv_images['uid'] == subject_id)].values
            self.image_F.append(image_[0][1])
            self.image_L.append(image_[1][1])

    def __len__(self):
        return len(self.csv_text)

    def get_a_pair(self, idx):

        img_name_F = self.image_F[idx]
        img_name_L = self.image_L[idx]

        chest_img_F = np.array(read_png(IMAGE_DIR + str(img_name_F)))
        chest_img_L = np.array(read_png(IMAGE_DIR + str(img_name_L)))

        chest_img_F = self.transform(chest_img_F)
        # print("After Transformation", chest_img_F.shape)
        chest_img_L = self.transform(chest_img_L)
        # print(self.findings[idx])

        sample = {
            'image_F': torch.tensor(chest_img_F, dtype=torch.float),
            'image_L': torch.tensor(chest_img_L, dtype=torch.float)
        }
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.get_a_pair(idx)
        p = random.uniform(0, 1)
        if p > 0.5:
            # Randomly choose a sample idx!=n_idx
            n_idx = idx
            while idx == n_idx:
                n_idx = random.randint(0, self.__len__() - 1)
            negative = self.get_a_pair(n_idx)
            sample['image_L'] = negative['image_L']
            sample['label'] = torch.ones(1).float()
        else:
            sample['label'] = torch.zeros(1).float()
        return sample