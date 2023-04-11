import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvLSTMCell, Sign

import numpy as np

class SideCell(nn.Module):
    def __init__(self):
        super(SideCell, self).__init__()

        self.conv = nn.Conv2d(
            3, 64//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.rnn1 = ConvLSTMCell(
            64//2,
            128//2,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            128//2,
            256//2,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            256//2,
            128//2,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn4 = ConvLSTMCell(
            128//2,
            64//2,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.conv1 = nn.Conv2d(
            64//2, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input, hidden1, hidden2,hidden3,hidden4):
        x = self.conv(input)
        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]
        
        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]
        
        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]
      
        hidden4 = self.rnn4(x, hidden4)
        x = hidden4[0]
        x = self.conv1(x)
        
        return x, hidden1, hidden2, hidden3,hidden4
