import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import nn
import tqdm as tqdm
import numpy as np
from torchvision.utils import save_image
from OpenAIDataset import OpenAIDataset
from Encoder import Encoder
from Decoder import Decoder
import pytorch_lightning as pl
from torch.utils.data import DataLoader
class OPENIDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int = 12, shuffle : bool = False):
        super(OPENIDataModule, self).__init__()

        self.train_set = None
        self.t2i_dataset = None
        self.batch_size = batch_size

    def setup(self, stage = None):

        self.t2i_dataset = OpenAIDataset(batch_size= 16,
                                         transform= None)
        self.train_set = OpenAIDataset(batch_size=16,
                                       transform=None)
        self.val_set = OpenAIDataset(batch_size=16,transform=None)

        self.test_set = OpenAIDataset(batch_size=16, transform=None)

    def train_dataloader(self):
        return DataLoader(self.train_set)

    def test_dataloader(self) :
        return DataLoader(self.test_set)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set)


class Tester:
    def __init__(self):
        
        super(Tester, self).__init__()



