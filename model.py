
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        # Definimos la convolución
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # Usamos ReLU como función de activación
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Aplicamos la convolución y luego la activación ReLU
        x = self.conv(x)
        x = self.relu(x)
        return x
# Definición de la clase Conv para una capa convolucional personalizada con LRN
class ConvLRN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLRN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)  # LRN

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.lrn(x)  # Aplicar LRN después de la activación
        return x
    
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)  # Batch Normalization
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class MaxPooling(nn.Module):
    def __init__(self,kernel_size=3,stride=2,padding=0):
        super(MaxPooling,self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding)
        
    def forward(self,x):
        x=self.maxpool(x)
        return x


class YOLOV1(nn.Module):
    def __init__(self,S,B,num_classes):
        super(YOLOV1,self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        
        self.layer1=Conv(kernel_size=7,in_channels=3,out_channels=64,stride=2)
        self.maxpool1=MaxPooling(kernel_size=2,stride=2)
        
        self.layer2=Conv(kernel_size=3,in_channels=64,out_channels=192,stride=1)
        self.maxpool2=MaxPooling(kernel_size=2,stride=2)
        
        self.layer3=Conv(kernel_size=1,in_channels=192,out_channels=128,stride=1)
        self.layer4=Conv(kernel_size=3,in_channels=128,out_channels=256,stride=1)
        self.layer5=Conv(kernel_size=1,in_channels=256,out_channels=256,stride=1)
        self.layer6=Conv(kernel_size=3,in_channels=256,out_channels=512,stride=1)
        self.maxpool3=MaxPooling(kernel_size=2,stride=2)
        
        
        
        self.layer7=Conv(kernel_size=1,in_channels=512,out_channels=256,stride=1)
        self.layer8=Conv(kernel_size=3,in_channels=256,out_channels=512,stride=1)
        
        self.layer9=Conv(kernel_size=1,in_channels=512,out_channels=256,stride=1)
        self.layer10=Conv(kernel_size=3,in_channels=256,out_channels=512,stride=1)
        
        self.layer11=Conv(kernel_size=1,in_channels=512,out_channels=256,stride=1)
        self.layer12=Conv(kernel_size=3,in_channels=256,out_channels=512,stride=1)
        
        self.layer13=Conv(kernel_size=1,in_channels=512,out_channels=256,stride=1)
        self.layer14=Conv(kernel_size=3,in_channels=256,out_channels=512,stride=1)
        
        self.layer15=Conv(kernel_size=1,in_channels=512,out_channels=512,stride=1)
        self.layer16=Conv(kernel_size=3,in_channels=512,out_channels=1024,stride=1)
        
        
        
        self.maxpool4=MaxPooling(kernel_size=2,stride=2)
        
        
        
        self.layer17=Conv(kernel_size=1,in_channels=1024,out_channels=512,stride=1)
        self.layer18=Conv(kernel_size=3,in_channels=512,out_channels=1024,stride=1)
        
        self.layer19=Conv(kernel_size=1,in_channels=1024,out_channels=512,stride=1)
        self.layer20=Conv(kernel_size=3,in_channels=512,out_channels=1024,stride=1)
        
        self.layer21=Conv(kernel_size=3,in_channels=1024,out_channels=1024,stride=1)
        
        self.layer22=Conv(kernel_size=3,in_channels=1024,out_channels=1024,stride=2)
        
        self.layer23=Conv(kernel_size=3,in_channels=1024,out_channels=1024,stride=1)
        self.layer24=Conv(kernel_size=3,in_channels=1024,out_channels=1024,stride=1)
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 9* 9, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.num_classes))  # Output de 7x7x30
        )
            
    def forward(self, x):
        x=self.layer1(x)
        x=self.maxpool1(x)
        
        x=self.layer2(x)
        x=self.maxpool2(x)
        
        
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)
        x=self.maxpool3(x)
        
        x=self.layer7(x)
        x=self.layer8(x)
        
        x=self.layer9(x)
        x=self.layer10(x)
        
        x=self.layer11(x)
        x=self.layer12(x)
        
        x=self.layer13(x)
        x=self.layer14(x)
        
        x=self.layer15(x)
        x=self.layer16(x)
        x=self.maxpool4(x)
        
        x=self.layer17(x)
        x=self.layer18(x)
        
        x=self.layer19(x)
        x=self.layer20(x)
        
        x=self.layer21(x)
        x=self.layer22(x)
        
        x=self.layer23(x)
        x=self.layer24(x)
        
        # Aplana el tensor para que tenga el tamaño adecuado

        
        x=self.fc1(x)
        
          # Reestructuramos la salida a 7x7x30
        x = x.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        
        return x
        
        
            
