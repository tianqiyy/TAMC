import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class default(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=9, out_channels=3, kernel_size=3, stride = 1, padding = 1) #[(W−K+2P)/S]+1
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=3, kernel_size=5, stride = 1, padding = 2)
        self.conv3 = nn.Conv1d(in_channels=9, out_channels=3, kernel_size=7, stride = 1, padding = 3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride = 1, padding = 0)
        self.fc1 = nn.Linear(3000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, input):
        x = input
        start = x.shape[2]//2 - 500
        end = x.shape[2]//2 + 500
        x = x[:, :, start:end] #[100, 9, 1000]

        x1 = F.relu(self.conv1(x)) #[100, 3, 1000]
        x2 = F.relu(self.conv2(x)) #[100, 3, 1000]
        x3 = F.relu(self.conv3(x)) #[100, 3, 1000]

        x1_p = self.maxpool1(x1.permute(0, 2, 1)).permute(0, 2, 1) #[100, 1, 1000]
        x2_p = self.maxpool1(x2.permute(0, 2, 1)).permute(0, 2, 1) #[100, 1, 1000]
        x3_p = self.maxpool1(x3.permute(0, 2, 1)).permute(0, 2, 1) #[100, 1, 1000]

        x_c = torch.cat((x1_p.view(-1, 1000), x2_p.view(-1, 1000), x3_p.view(-1, 1000)),1) #[100, 3000]

        x_l1 = self.fc1(x_c) #[100, 1000]
        x_l2 = self.fc2(x_l1) #[100, 100]
        x_l3 = self.fc3(x_l2) #[100, 2]

        return x_l3
