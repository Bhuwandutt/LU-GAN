import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torchvision.transforms as transforms
import random
from utils.utils import *


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 32
NUM_WORKERS = 1
CSV_DIR = '/Users/bhuwandutt/Documents/GitHub/LU-GAN/csv/'
IMAGE_DIR = '/Users/bhuwandutt/Documents/GitHub/LU-GAN/data/imgs/'


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

        self.findings = []
        self.impression = []
        self.image_L = []
        self.image_F = []
        self.subject_ids = []
        self.word_dict = '/Users/bhuwandutt/Documents/GitHub/LU-GAN/utils/dict.json'
        if os.path.exists(self.word_dict):
            with open(self.word_dict) as f:
                self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding, \
                self.max_word_len_impression, self.max_word_len_finding = json.load(f)
        else:
            self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding, \
                self.max_word_len_impression, self.max_word_len_finding = self.get_word_by_index()
            with open(self.word_dict, 'w') as f:
                json.dump([self.word_to_idx, self.vocab_size, self.max_len_impression, self.max_len_finding,
                           self.max_word_len_impression, self.max_word_len_finding], f)

        for index, row in tqdm(self.csv_text.iterrows()):
            subject_id = row['uid']
            self.subject_ids.append(subject_id)
            image_ = self.csv_images.loc[
                (self.csv_images['uid'] == subject_id)].values
            self.image_F.append(image_[0][1])
            self.image_L.append(image_[1][1])

            fi = row['findings']
            im = row['impression']
            txt_finding = []
            txt_impression = []

            for w in fi.split():
                txt_finding_sen = self.word_to_idx[w]
                txt_finding_sen = np.pad(txt_finding_sen,
                                         (self.max_word_len_finding - len(str(txt_finding_sen)), 0),
                                         'constant', constant_values=0)
                txt_finding.append(txt_finding_sen)

            for w in im.split():
                txt_impression_sen = self.word_to_idx[w]
                txt_impression_sen = np.pad(txt_impression_sen,
                                            (self.max_word_len_impression - len(str(txt_impression_sen)), 0),
                                            'constant', constant_values=0)
                txt_impression.append(txt_impression_sen)

            txt_finding = np.pad(np.array(txt_finding),
                                 (self.max_len_finding - len(txt_finding), 0),
                                 'constant',
                                 constant_values=0)
            self.findings.append(txt_finding)

            txt_impression = np.pad(np.array(txt_impression),
                                    (self.max_len_impression - len(txt_impression), 0),
                                    'constant',
                                    constant_values=0)
            self.impression.append(txt_impression)

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
        # print(self.findings[idx].shape)

        # print(self.findings[idx])

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

    def get_word_by_index(self):
        print("Counting Vocabulary....")
        wordbag = []
        len_finding = []
        len_impression = []
        word_len_finding = []
        word_len_impression = []

        for index, row in tqdm(self.csv_text.iterrows()):
            fi = row['findings'].split()
            im = row['impression'].split()

            # print(fi)

            len_impression.append(len(im))
            len_finding.append(len(fi))

            for word in fi:
                word_len_finding.append(len(word))
                wordbag.append(word)
            for word in im:
                word_len_impression.append(len(word))
                wordbag.append(word)

        vocab = set(wordbag)
        word_to_idx = {}
        count = 1
        # print(wordbag)

        for word in vocab:
            if word in word_to_idx.keys():
                pass
            else:
                word_to_idx[word] = count
                count += 1

        # print(word_to_idx)
        vocab_len = count
        max_len_im, max_len_fi = max(len_impression), max(len_finding)
        max_word_len_im, max_word_len_fi = max(word_len_impression), max(word_len_finding)
        # print(type(word_to_idx))
        # print(type(vocab_len))
        # print(type(max_len_im))
        # print(type(max_len_fi))
        # print(type(max_word_len_fi))
        # print(type(max_word_len_im))

        print("Totally {} medical report".format(self.__len__()))
        print("Totally {} vocabulary".format(vocab_len))
        print("Max Finding: Sentence Length {} \t Word Length {} ".format(max_len_fi, max_word_len_fi))
        print("Max Impression: Sentence length {} \t Word Length {}".format(max_len_im, max_word_len_im))
        return word_to_idx, vocab_len, max_len_im, max_len_fi, max_word_len_im, max_word_len_fi


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