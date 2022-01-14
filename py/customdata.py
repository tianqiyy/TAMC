import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class CustomData(Dataset):
    def __init__(self, datadir):
        label_list = []
        info_list = []

        self.datadir = datadir # datadir contains training, validation or test datasets
        self.all_filenames = os.listdir(self.datadir) #Define dataset filenames

        #Define file labels and moods score for each mpbs
        for filename in self.all_filenames:
          info_list.append("_".join(filename.split("_")[-2:]).split(".")[-2])
          if (filename.split(".")[-2].split("_")[-1] == "bound"):
            label_list.append(1)
          else:
            label_list.append(0)

        self.label_list = label_list
        self.info_list = info_list

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):

        #load input
        selected_filename = self.all_filenames[idx]
        input = np.load(os.path.join(self.datadir, selected_filename), allow_pickle=True)
        input = torch.from_numpy(input)
        input = input.type(torch.FloatTensor)

        #load label
        selected_label = self.label_list[idx]
        label = np.array(selected_label)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        #load info
        info = self.info_list[idx]

        return input, label, info
