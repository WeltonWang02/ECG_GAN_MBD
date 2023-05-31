# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:16:34 2019

@author: anne marie delaney
         eoin brophy
         
Module of the GAN model for time series synthesis.

"""

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


""" 
NN Definitions
---------------
Defining the Neural Network Classes to be evaluated in this Notebook

Minibatch Discrimination
--------------------------
Creating a module for Minibatch Discrimination to avoid mode collapse as described:
https://arxiv.org/pdf/1606.03498.pdf
https://torchgan.readthedocs.io/en/latest/modules/layers.html#minibatch-discrimination

"""

class MinibatchDiscrimination(nn.Module):
   def __init__(self,input_features,output_features,minibatch_normal_init, hidden_features=16):
      super(MinibatchDiscrimination,self).__init__()
      
      self.input_features = input_features
      self.output_features = output_features
      self.hidden_features = hidden_features
      self.T = nn.Parameter(torch.randn(self.input_features,self.output_features, self.hidden_features))
      if minibatch_normal_init == True:
        nn.init.normal(self.T, 0,1)
      
   def forward(self,x):
      M = torch.mm(x,self.T.view(self.input_features,-1))
      M = M.view(-1, self.output_features, self.hidden_features).unsqueeze(0)
      M_t = M.permute(1, 0, 2, 3)
      # Broadcasting reduces the matrix subtraction to the form desired in the paper
      out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
      return torch.cat([x, out], 1)

"""
Discriminator Class
-------------------
This discriminator has a parameter num_cv which allows the user to specify if 
they want to have 1 or 2 Convolution Neural Network Layers.

"""
class Discriminator(nn.Module):
    def __init__(self, seq_length, batch_size, minibatch_normal_init, n_features = 1, num_cv = 1, minibatch = 0, cv1_out= 10, cv1_k = 3, cv1_s = 4, p1_k = 3, p1_s = 3, cv2_out = 10, cv2_k = 3, cv2_s = 3 ,p2_k = 3, p2_s = 3):
        super(Discriminator, self).__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.C1 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=3, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.P1 = nn.MaxPool1d(kernel_size=3, stride=1)

        self.C2 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=5, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.P2 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.C3 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.P3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.C4 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=12, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.P4 = nn.MaxPool1d(kernel_size=5, stride=2)

        self.out = nn.Sequential(
            nn.Linear(12 * 2, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.C1(x.view(self.batch_size, self.n_features, self.seq_length))
        x = self.P1(x)

        x = self.C2(x)
        x = self.P2(x)

        x = self.C3(x)
        x = self.P3(x)

        x = self.C4(x)
        x = self.P4(x)

        x = x.view(self.batch_size, -1)
        x = self.out(x)

        return x

"""
Generator Class
---------------
This defines the Generator for evaluation. The Generator consists of two LSTM 
layers with a final fully connected layer.

"""

class Generator(nn.Module):
  def __init__(self,seq_length,batch_size,n_features = 1, hidden_dim = 50, 
               num_layers = 2, tanh_output = False):
      super(Generator,self).__init__()
      self.n_features = n_features
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.seq_length = seq_length
      self.batch_size = batch_size
      self.tanh_output = tanh_output
      

      
      self.layer1 = nn.LSTM(input_size = self.n_features, hidden_size = self.hidden_dim, 
                                  num_layers = self.num_layers,batch_first = True#,dropout = 0.2,
                                 )
      if self.tanh_output == True:
        self.out = nn.Sequential(nn.Linear(self.hidden_dim,1),nn.Tanh()) # to make sure the output is between 0 and 1 - removed ,nn.Sigmoid()
      else:
        self.out = nn.Linear(self.hidden_dim,1) 
      
  def init_hidden(self):
      weight = next(self.parameters()).data
      hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device), weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().to(device))
      return hidden
  
  def forward(self,x,hidden):
      
      x,hidden = self.layer1(x.view(self.batch_size,self.seq_length,1),hidden)
      
      x = self.out(x)
      
      return x #,hidden
