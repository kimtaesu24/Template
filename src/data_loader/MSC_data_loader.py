import numpy as np
import pandas as pd
import torch
import os

from model import modules
from torch.utils.data import Dataset

class MSC_Dataset(Dataset):
    def __init__(self, data_path, device, args, mode='train'):
        self.data_path = data_path
        self.device = device
        
        self.manual_index = 0
        self.mode = mode

    def __len__(self):
        length = 0
        return length

    def __getitem__(self, idx):
        if idx == 0:
            self.manual_index = 0  # initialize

        inputs = [tokens,
                  audio_feature,
                  visual_feature,
                  input_historys_tokens,
                  ]

        labels = [tokens_labels]
        
        return inputs, labels
